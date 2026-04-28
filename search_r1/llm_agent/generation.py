import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import requests
import torch

from verl import DataProto

from .tensor_helper import TensorConfig, TensorHelper
from search_r1.search.evidence_formatter import format_relevant_evidence


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3
    max_invalid_actions: int = 3
    max_search_result_chars: int = 1700
    terminate_on_observation_overflow: bool = True
    max_evidence_sentences: int = 3
    max_evidence_sentence_chars: int = 220
    require_search_verification: bool = True


class LLMGenerationManager:
    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig, is_validation: bool = False):
        self.tokenizer = tokenizer 
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )
        self.search_guidance = (
            "Do not use the provisional diagnosis as the final answer unless the search confirms it.\n"
            "Search using distinctive clinical or imaging findings, not the suspected diagnosis.\n"
        )
        self.prompt_guidance = (
            "\nImportant rules:\n"
            "1. Do not use the provisional diagnosis as the final answer unless the search confirms it.\n"
            "2. Search using distinctive clinical or imaging findings, not the suspected diagnosis.\n"
        )

    def _normalize_query(self, query: str) -> str:
        query = re.sub(r"[^\w\s']", " ", (query or "").lower())
        return re.sub(r"\s+", " ", query).strip()

    def _batch_tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return self.tokenizer(
            list(texts),
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def _truncate_generated_text(self, text: str) -> str:
        text = text.strip()
        for closing in ("</search>", "</answer>"):
            if closing in text:
                return text.split(closing)[0] + closing
        return text

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        # Keep only the first completed tool action so downstream parsing sees one step at a time.
        responses_str = [
            self._truncate_generated_text(text)
            for text in self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        ]
        return self._batch_tokenize(responses_str), responses_str

    def _append_prompt_guidance(self, input_ids: torch.Tensor) -> torch.Tensor:
        guidance_ids = self._batch_tokenize([self.prompt_guidance] * input_ids.shape[0]).to(input_ids.device)
        augmented = self.tensor_fn.concatenate_with_padding([input_ids, guidance_ids], pad_to_left=True)
        attention_mask = self.tensor_fn.create_attention_mask(augmented)
        max_len = min(self.config.max_start_length, int(attention_mask.sum(dim=1).max().item()))
        return augmented[:, -max_len:]

    def _process_next_obs(self, next_obs: List[str]) -> Tuple[torch.Tensor, List[bool]]:
        tokenized = self.tokenizer(next_obs, add_special_tokens=False, padding=False)["input_ids"]
        overflow_mask, truncated = [], []
        for obs_text, obs_ids in zip(next_obs, tokenized):
            # Clip observations at the environment boundary so the prompt budget stays predictable.
            obs_limit = self.config.max_obs_length * 2 if "<information>" in obs_text else self.config.max_obs_length
            is_overflow = len(obs_ids) > obs_limit
            overflow_mask.append(is_overflow)
            truncated.append(obs_ids[:obs_limit])
        next_obs_ids = self.tokenizer.pad({"input_ids": truncated}, padding="longest", return_tensors="pt")[
            "input_ids"
        ]
        return next_obs_ids, overflow_mask

    def _update_rolling_state(
        self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor
    ) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        max_len = min(self.config.max_prompt_length, int(new_attention_mask.sum(dim=1).max().item()))
        updated = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
            }
        )
        updated.meta_info.update(rollings.meta_info)
        return updated

    def _concat_with_info_mask(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        masked_tensors = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            # Retrieval evidence is present in the visible sequence but masked out for info-aware consumers.
            masked_tensors.append(torch.full_like(info, pad_id))
        merged = torch.cat(tensors, dim=1)
        merged_masked = torch.cat(masked_tensors, dim=1)
        # Re-pack to right padding after concatenation so later slicing stays simple.
        mask = merged == pad_id
        order = mask.to(torch.int64).argsort(dim=1, stable=True)
        return merged.gather(1, order), merged_masked.gather(1, order)

    def _update_right_side(
        self, right_side: Dict[str, torch.Tensor], cur_responses: torch.Tensor, next_obs_ids: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        responses, masked = self._concat_with_info_mask(
            right_side["responses"],
            right_side["responses_with_info_mask"],
            cur_responses,
            next_obs_ids,
        )
        max_len = min(
            self.config.max_prompt_length,
            int(self.tensor_fn.create_attention_mask(responses).sum(dim=1).max().item()),
        )
        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": masked[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        batch_size = active_batch.batch["input_ids"].shape[0]
        if batch_size == 0:
            return None

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if self.config.num_gpus <= 1 or batch_size % self.config.num_gpus == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Some distributed generation paths expect every step batch to divide evenly across GPUs.
        padding_size = self.config.num_gpus - (batch_size % self.config.num_gpus)
        padded_batch = {}
        for key, value in active_batch.batch.items():
            pad_rows = value[:1].repeat(padding_size, *([1] * (value.ndim - 1)))
            padded_batch[key] = torch.cat([value, pad_rows], dim=0)

        output = self.actor_rollout_wg.generate_sequences(DataProto.from_dict(padded_batch))
        output.batch = {key: value[:-padding_size] for key, value in output.batch.items()}
        if getattr(output, "meta_info", None):
            trimmed_meta = {}
            for key, value in output.meta_info.items():
                trimmed_meta[key] = value[:-padding_size] if isinstance(value, torch.Tensor) else value
            output.meta_info = trimmed_meta
        return output

    def postprocess_predictions(self, predictions: Sequence[str]) -> Tuple[List[str], List[str]]:
        actions, contents = [], []
        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            text = prediction.strip()
            answer_match = re.search(r"<answer\b[^>]*>(.*?)</answer>", text, re.DOTALL)
            search_match = re.search(r"<search\b[^>]*>(.*?)</search>", text, re.DOTALL)
            if answer_match:
                actions.append("answer")
                contents.append(answer_match.group(1).strip())
            elif search_match:
                actions.append("search")
                contents.append(search_match.group(1).strip())
            else:
                actions.append(None)
                contents.append("")
        return actions, contents

    def _search(self, queries: List[str]) -> List[str]:
        if not queries:
            return []
        payload = {"queries": queries, "topk": self.config.topk, "return_scores": True}
        try:
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json().get("result", [])
        except Exception as exc:
            # Keep rollouts alive on retriever failure by returning a readable synthetic observation.
            return [f"Search request failed: {exc}"] * len(queries)

        formatted = []
        for query, retrieval_result in zip(queries, results):
            formatted.append(
                format_relevant_evidence(
                    query=query,
                    retrieval_result=retrieval_result,
                    max_total_chars=self.config.max_search_result_chars,
                    max_sentences=self.config.max_evidence_sentences,
                    per_sentence_chars=self.config.max_evidence_sentence_chars,
                )
            )
        if len(formatted) < len(queries):
            formatted.extend(["No evidence retrieved."] * (len(queries) - len(formatted)))
        return formatted

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search: bool = True,
        search_counts: List[int] = None,
        previous_actions: List[str] = None,
        previous_search_queries: List[str] = None,
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        del pad_token
        if active_mask is None:
            active_mask = torch.ones(len(predictions), dtype=torch.bool)

        actions, contents = self.postprocess_predictions(predictions)
        search_counts = search_counts or [0] * len(actions)
        previous_actions = previous_actions or [None] * len(actions)
        previous_search_queries = previous_search_queries or [""] * len(actions)

        queued_queries = []
        for i, (active, action, content) in enumerate(zip(active_mask.tolist(), actions, contents)):
            repeated = (
                active
                and action == "search"
                and previous_actions[i] == "search"
                and self._normalize_query(content) == self._normalize_query(previous_search_queries[i])
            )
            if do_search and active and action == "search" and not repeated: #valid search queries are appended into queued queries 
                queued_queries.append(content)

        # Search once up front, then consume results in trajectory order during the main pass below.
        search_results = iter(self._search(queued_queries)) #call search 
        next_obs, dones, valid_action, is_search = [], [], [], []

        for i, (active, action, content) in enumerate(zip(active_mask.tolist(), actions, contents)):
            if not active: #finish no need append net obseration, mark done, not valid action, not search
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue
            
            # for active trajectories, check current action (answer/search)
            repeated = (
                action == "search" and previous_actions[i] == "search"
                and self._normalize_query(content) == self._normalize_query(previous_search_queries[i])
            )

            #case 1: invalid action - answering without search, invoke search + search guidance (search by findings and symptoms)
            if action == "answer" and self.config.require_search_verification and search_counts[i] <= 0: 
                next_obs.append(
                    "Search required to supporrt your answer. Write your query in <search> </search> to verify your answer.\n"
                    "The searched results are retrieved from <information> and </information> as a compact evidence summary.\n"
                    f"{self.search_guidance}"
                )
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)

            # case 2: invalid action - repeated search query 
            elif repeated:  
                next_obs.append(
                    "Repeated search query. Rewrite the query instead of searching the same thing again.\n"
                    f"{self.search_guidance}"
                )
                dones.append(0) 
                valid_action.append(0) 
                is_search.append(0)
            
            # case 3: answer with search
            elif action == "answer": #answer with search is a valid action, not a search, done, no observation 
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
            
            # case 4: valid search , save search queries to track 
            elif action == "search": 
                evidence = next(search_results, "No evidence retrieved.")
                next_obs.append(f"\n\n<information>{evidence.strip()}</information>\n\n")
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                previous_search_queries[i] = content

            #case 5: no <search> and <answer> tags, might be formatting problem
            else:
                next_obs.append(
                    "Invalid action. Episode terminated. Output must be exactly one action: "
                    "<search>query</search> or <answer>most likely diagnostic</answer>.\n"
                )
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)

            previous_actions[i] = action #save action as prev action to track for repeated search action 

        #return lists of next obs strings, dones binary valid_action and is_search binary
        return next_obs, dones, valid_action, is_search

    def _compose_final_output(
        self, left_side: Dict[str, torch.Tensor], right_side: Dict[str, torch.Tensor], meta_info: Dict
    ) -> DataProto:
        final_output = dict(right_side)
        final_output["prompts"] = left_side["input_ids"]
        final_output["input_ids"] = torch.cat([left_side["input_ids"], right_side["responses"]], dim=1)
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(right_side["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(right_side["responses_with_info_mask"]),
            ],
            dim=1,
        )
        final_output["position_ids"] = self.tensor_fn.create_position_ids(final_output["attention_mask"])
        output = DataProto.from_dict(final_output)
        output.meta_info.update(meta_info)
        return output

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        initial_input_ids = self._append_prompt_guidance(initial_input_ids)
        gen_batch.batch["input_ids"] = initial_input_ids
        gen_batch.batch["attention_mask"] = self.tensor_fn.create_attention_mask(initial_input_ids)
        gen_batch.batch["position_ids"] = self.tensor_fn.create_position_ids(gen_batch.batch["attention_mask"])

        left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length :]}
        right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        batch_size = gen_batch.batch["input_ids"].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        invalid_action_streak = torch.zeros(batch_size, dtype=torch.int)
        previous_actions = [None] * batch_size
        previous_search_queries = [""] * batch_size
        rollings = gen_batch
        meta_info = dict(gen_batch.meta_info)

        for step in range(self.config.max_turns):
            if not active_mask.any():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
            )
            # Generate only for unfinished examples, then pad the decoded results back to full batch shape.
            active_rollings = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(active_rollings)
            if gen_output is None:
                break

            meta_info = dict(gen_output.meta_info)
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            for idx, response in enumerate(responses_str):
                if active_mask[idx]:
                    print(f"[ROLLOUT_STEP] step={step} idx={idx} response={response.replace(chr(10), ' ')[:500]}")

            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask=active_mask,
                search_counts=valid_search_stats.tolist(),
                previous_actions=previous_actions,
                previous_search_queries=previous_search_queries,
            )

            valid_action_tensor = torch.tensor(valid_action, dtype=torch.int)
            invalid_action_streak = torch.where(
                valid_action_tensor.bool(),
                torch.zeros_like(invalid_action_streak),
                invalid_action_streak + active_mask.to(torch.int),
            )
            invalid_limit_hit = invalid_action_streak >= self.config.max_invalid_actions
            dones = [int(done or invalid_limit_hit[i].item()) for i, done in enumerate(dones)]

            current_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & current_active_mask
            turns_stats[current_active_mask] += 1
            valid_action_stats += valid_action_tensor
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids, overflow_mask = self._process_next_obs(next_obs)
            if self.config.terminate_on_observation_overflow:
                for i, is_overflow in enumerate(overflow_mask):
                    if is_overflow:
                        active_mask[i] = False

            # The left side stays fixed; the right side accumulates generated actions plus observations.
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            right_side = self._update_right_side(right_side, responses_ids, next_obs_ids)

        if active_mask.any():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
            )
            # Final pass asks the model to answer from the last state without performing another search.
            active_rollings = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(active_rollings)
            if gen_output is not None:
                meta_info = dict(gen_output.meta_info)
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
                _, dones, valid_action, is_search = self.execute_predictions(
                    responses_str,
                    self.tokenizer.pad_token,
                    active_mask=active_mask,
                    do_search=False,
                    search_counts=valid_search_stats.tolist(),
                    previous_actions=previous_actions,
                    previous_search_queries=previous_search_queries,
                )
                active_mask = active_mask & torch.tensor([not done for done in dones], dtype=torch.bool)
                valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
                valid_search_stats += torch.tensor(is_search, dtype=torch.int)
                right_side = self._update_right_side(right_side, responses_ids)

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["valid_action_stats"] = valid_action_stats.tolist()
        meta_info["valid_search_stats"] = valid_search_stats.tolist()
        return self._compose_final_output(left_side, right_side, meta_info)
