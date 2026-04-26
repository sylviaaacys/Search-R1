# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

#rule-based RL 
INVALID_ACTION_TEXT = "My previous action is invalid."
INVALID_ACTION_MARKERS = (INVALID_ACTION_TEXT, "Invalid action.")
NONSENSE_ANSWER_SET = {
    "", "and", "or", "the", "a", "an", "yes", "no", "maybe", "unknown",
    "unclear", "none", "n/a","...",
}
GENERIC_SEARCH_PATTERNS = (
    "what is the most likely diagnosis",
    "most likely diagnosis for the patient",
    "for the patient described in the case report",
    "patient described in the case report",
    "what is the diagnosis",
)
INSTRUCTION_COMPLETION_PATTERNS = (
    "you can search as many times as you want",
    "write it in <answer></answer>",
    "output the final diagnosis as <answer>",
    "do not write chatty completions",
    "search required before answering",
)
BAD_ANSWER_PHRASES = (
    "based on",
    "clinical presentation",
    "laboratory findings",
    "most likely diagnosis",
    "symptoms",
    "consistent with",
    "support",
    "additionally",
    "therefore",
    "likely a result",
)

def _extract_assistant_content(text):
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    if assistant_match:
        return text[assistant_match.end():]
    return text

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set((string.punctuation.replace("'", "")) + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace("_", " ")
    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def _as_answer_list(golden_answers): #there's multiple answer 
    if isinstance(golden_answers, str):
        return [golden_answers]
    return golden_answers

def em_check(prediction, golden_answers):
    golden_answers = _as_answer_list(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction: 
            score = 1
            break
    return score

def cover_em_check(prediction, golden_answers):
    golden_answers = _as_answer_list(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    prediction_token_set = set(prediction_tokens)
    best_score = 0

    for golden_answer in golden_answers:
        golden_tokens = normalize_answer(golden_answer).split()
        if not golden_tokens:
            continue

        covered_tokens = sum(1 for token in golden_tokens if token in prediction_token_set)
        best_score = max(best_score, covered_tokens / len(golden_tokens))

    return best_score


def text_contains_gold_answer(text: str, golden_answers) -> bool:
    normalized_text = normalize_answer(extract_assistant_text(text))
    for golden_answer in _as_answer_list(golden_answers):
        normalized_golden = normalize_answer(golden_answer)
        if normalized_golden and normalized_golden in normalized_text:
            return True
    return False

def count_valid_search_actions(text):
    content = extract_assistant_text(text)
    return len(re.findall(r"<search\b[^>]*>.*?</search>", content, re.DOTALL))

def count_invalid_actions(text):
    return sum(text.count(marker) for marker in INVALID_ACTION_MARKERS)

def extract_assistant_text(text):
    return _extract_assistant_content(text)

def has_action_tag(text):
    content = extract_assistant_text(text)
    return bool(re.search(r"<(?:search|answer)\b[^>]*>", content))

def extract_first_action(text):
    content = extract_assistant_text(text)
    match = re.search(r"<(search|answer)\b[^>]*>", content)
    if match:
        return match.group(1)
    return None


def extract_actions(text: str) -> list[str]:
    content = extract_assistant_text(text)
    return re.findall(r"<(search|answer)\b[^>]*>", content)

def extract_search_queries(text: str) -> list[str]:
    content = extract_assistant_text(text)
    return [match.strip() for match in re.findall(r"<search\b[^>]*>(.*?)</search>", content, re.DOTALL)]


def count_repeated_same_query(text: str) -> int:
    queries = [normalize_answer(query) for query in extract_search_queries(text)]
    repeats = 0
    previous_query = None
    for query in queries:
        if previous_query is not None and query == previous_query:
            repeats += 1
        previous_query = query
    return repeats

def has_generic_search_query(text: str) -> bool:
    queries = [normalize_answer(query) for query in extract_search_queries(text)]
    for query in queries:
        if not query:
            continue
        if any(pattern in query for pattern in GENERIC_SEARCH_PATTERNS):
            return True
    return False

def has_instruction_completion(text: str) -> bool:
    content = normalize_answer(extract_assistant_text(text))
    return any(pattern in content for pattern in INSTRUCTION_COMPLETION_PATTERNS)

def is_symptoms_only_output(text):
    content = extract_assistant_text(text)
    has_symptoms = bool(re.search(r"<symptoms\b[^>]*>.*?</symptoms>", content, re.DOTALL))
    return has_symptoms and not has_action_tag(text)

def is_overlong_answer(answer, max_words=24, max_chars=160):
    if answer is None:
        return False
    normalized = " ".join(answer.split())
    return len(normalized) > max_chars or len(normalized.split()) > max_words

def validate_answer(ans: str):
    text = ans.strip().lower()

    if "\n" in ans:
        return False, "multiline_answer"

    if len(ans.split()) > 8:
        return False, "answer_too_long"

    if any(phrase in text for phrase in BAD_ANSWER_PHRASES):
        return False, "explanatory_answer"

    if any(ch in ans for ch in [":", ";"]):
        return False, "explanatory_punctuation"

    return True, "ok"


def answer_copies_information(answer, text, min_overlap_tokens=4, overlap_ratio=0.8):
    if answer is None:
        return False

    answer_tokens = normalize_answer(answer).split()
    if len(answer_tokens) < min_overlap_tokens:
        return False

    answer_token_set = set(answer_tokens)
    for info_block in extract_information_blocks(text):
        info_tokens = set(normalize_answer(info_block).split())
        if not info_tokens:
            continue
        overlap = len(answer_token_set & info_tokens) / max(1, len(answer_token_set))
        if overlap >= overlap_ratio:
            return True
    return False


def is_empty_or_nonsense_answer(answer):
    if answer is None:
        return False
    normalized = normalize_answer(answer)
    if normalized in NONSENSE_ANSWER_SET:
        return True
    if len(normalized.split()) == 0:
        return True
    return False

def is_valid_sequence(text):
    content = _extract_assistant_content(text)

    tags_to_check = ["symptoms", "search", "information", "answer", "think"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:symptoms|think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    state = "start"
    saw_action = False

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if re.match(r"</?(?:symptoms|think|search|information|answer)>", part):
            if part == "<symptoms>" and state in ["after_think", "information", "end"]:
                state = "in_symptoms"
            elif part == "</symptoms>" and state == "in_symptoms":
                if saw_action:
                    state = "end"
                else:
                    state = "after_symptoms"
            elif part == "<think>" and state in ["start", "after_symptoms", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state in ["start", "after_symptoms", "after_think", "information"]:
                state = "in_search"
                saw_action = True
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state in ["start", "after_symptoms", "after_think", "information"]:
                state = "in_answer"
                saw_action = True
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_symptoms", "in_think", "in_search", "in_information", "in_answer"]:
                pass
            elif state in ["start", "after_symptoms", "after_think", "after_search", "information", "end"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state not in ["after_search", "information", "end"]:
        return False, f"Incomplete sequence, ended in state {state}"
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    content = extract_assistant_text(solution_str)
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, content, re.DOTALL)
    matches = list(match)
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, extract_assistant_text(text), re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in _as_answer_list(golden_answers):
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_format_reward(
    is_valid_format: bool,
    retrieval_correct: bool,
    structure_format_score: float,
    retrieval_score: float,
) -> float:
    if not is_valid_format:
        return 0.0
    reward = structure_format_score
    if retrieval_correct:
        reward += retrieval_score
    return reward

def compute_score_em(
    solution_str,
    ground_truth,
    method='strict',
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    cover_exact_score=0,
    invalid_action_penalty=0,
    repeated_invalid_penalty=0,
    empty_answer_penalty=0,
    valid_search_bonus=0,
    first_search_bonus=0,
    generic_search_penalty=0,
    instruction_completion_penalty=0,
    answer_format_penalty=0,
    no_search_wrong_penalty=0,
    invalid_format_penalty=0,
    no_action_penalty=0,
    symptoms_only_penalty=0,
    overlong_answer_penalty=0,
    copied_answer_penalty=0,
    repeated_same_query_penalty=0,
    answer_before_search_penalty=0,
    extra_search_penalty=0,
    answer_max_words=24,
    answer_max_chars=160,
    score=1.,
):
    """The scoring function for exact match (EM)"
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    reply_has_gold_answer = text_contains_gold_answer(solution_str, ground_truth['target'])
    tagged_has_em = answer is not None and em_check(answer, ground_truth['target'])
    tagged_cover_em = answer is not None and cover_em_check(answer, ground_truth['target'])
    fallback_cover_em = 0 if answer is not None else cover_em_check(extract_assistant_text(solution_str), ground_truth['target'])
    has_em = bool(tagged_has_em or reply_has_gold_answer)
    has_cover_em = max(tagged_cover_em, fallback_cover_em)
    valid_search_count = count_valid_search_actions(solution_str)
    invalid_action_count = count_invalid_actions(solution_str)
    first_action = extract_first_action(solution_str)
    actions = extract_actions(solution_str)
    total_action_count = len(actions)
    repeated_same_query_count = count_repeated_same_query(solution_str)
    generic_search = has_generic_search_query(solution_str)
    instruction_completion = has_instruction_completion(solution_str)
    empty_or_nonsense = is_empty_or_nonsense_answer(answer)
    valid_answer_format, answer_format_reason = validate_answer(answer) if answer is not None else (True, "no_answer")
    found_answer_content = answer is not None or reply_has_gold_answer or has_cover_em > 0
    answered_without_search = found_answer_content and valid_search_count == 0
    answer_before_search = first_action == "answer"
    missing_action = not has_action_tag(solution_str)
    symptoms_only = is_symptoms_only_output(solution_str)
    overlong_answer = is_overlong_answer(answer, max_words=answer_max_words, max_chars=answer_max_chars)
    copied_answer = answer_copies_information(answer, solution_str)
    normalized_answer = normalize_answer(answer) if answer is not None else None
    extra_search_count = max(valid_search_count - 2, 0)
    format_reward = compute_format_reward(
        is_valid_format=is_valid_format,
        retrieval_correct=retrieval_correct,
        structure_format_score=structure_format_score,
        retrieval_score=retrieval_score,
    )
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Normalized extracted answer: {normalized_answer}")
        print(f"Reply contains gold answer: {reply_has_gold_answer}")
        print(f"Tagged exact match: {tagged_has_em}")
        print(f"Exact match: {has_em}")
        print(f"Tagged cover exact match: {tagged_cover_em}")
        print(f"Cover exact match: {has_cover_em}")
        print(f"Actions: {actions}")
        print(f"Total actions: {total_action_count}")
        print(f"Valid search count: {valid_search_count}")
        print(f"Repeated same query count: {repeated_same_query_count}")
        print(f"Extra search count: {extra_search_count}")
        print(f"Invalid action count: {invalid_action_count}")
        print(f"First action: {first_action}")
        print(f"Answer before search: {answer_before_search}")
        print(f"Generic search: {generic_search}")
        print(f"Instruction completion: {instruction_completion}")
        print(f"Answer format valid: {valid_answer_format} ({answer_format_reason})")
        print(f"Retrieval contains answer: {retrieval_correct}")
        print(f"Missing action: {missing_action}")
        print(f"Symptoms only: {symptoms_only}")
        print(f"Overlong answer: {overlong_answer}")
        print(f"Copied answer: {copied_answer}")
        print(f"Solution string: {solution_str}")

    reward = 0.0

    if invalid_action_count > 0:
        reward -= invalid_action_penalty
    if invalid_action_count > 1:
        reward -= repeated_invalid_penalty
    if generic_search:
        reward -= generic_search_penalty
    if instruction_completion:
        reward -= instruction_completion_penalty
    if not is_valid_format:
        reward -= invalid_format_penalty
    if missing_action:
        reward -= no_action_penalty
    if symptoms_only:
        reward -= symptoms_only_penalty
    reward -= repeated_same_query_penalty * repeated_same_query_count
    reward -= extra_search_penalty * extra_search_count
    if answer_before_search:
        reward -= answer_before_search_penalty

    if tagged_has_em:
        if is_valid_format:
            reward += score
        else:
            reward += score - structure_format_score
    elif answer is not None and tagged_cover_em:
        if is_valid_format:
            reward += max(cover_exact_score, format_reward)
        else:
            reward += max(cover_exact_score, final_format_score)
    elif reply_has_gold_answer:
        reward += cover_exact_score + format_reward
    else:
        if answer is not None and not is_valid_format:
            reward += final_format_score
        else:
            reward += format_reward

    if answer is not None:
        if empty_or_nonsense:
            reward -= empty_answer_penalty
        if not valid_answer_format and not tagged_has_em:
            reward -= answer_format_penalty
        if overlong_answer and not tagged_has_em:
            reward -= overlong_answer_penalty
        if copied_answer and not tagged_has_em:
            reward -= copied_answer_penalty

    if valid_search_count > 0:
        reward += valid_search_bonus
    if first_action == "search":
        reward += first_search_bonus
    if answered_without_search and not has_em and has_cover_em <= 0: #answer without search nd wrong 
        reward -= no_search_wrong_penalty
    if do_print:
        print(f"Reward: {reward}")
    return reward
