# An Evidence-Informed Synthesis Agent for Medical Deep Research

**Author:** Sylvia Chung Yan Shan  
**Institution:** Zhejiang University–University of Illinois Urbana-Champaign Institute  
**Year:** 2026
These notes cover the training setup, results, and limitations of my senior thesis.

## Research questions
1. Does retrieved evidence improve diagnosis compared with direct inference?
2. How do retriever choice and retrieved context size affect performance?
3. Which search and reasoning patterns appear in successful trajectories?

## Training pipeline

![Training pipeline](../public/medical_pipeline.png)

I initialize the agent with medical SFT, then train it using GRPO. Each case generates a group of candidate trajectories that alternate between search actions and retrieved evidence before producing a final diagnosis. Group-relative rewards provide advantages; KL regularization limits drift from the reference policy. Retrieved text is excluded from policy-gradient and KL-loss terms during policy optimization.

### SFT settings

| Parameter | Thesis appendix |
| --- | --- |
| Base model | Qwen2.5-3B-Instruct |
| Dataset | MedCaseReasoning |
| Adaptation | LoRA, rank 32, alpha 16, dropout 0.1 |
| LoRA targets | Q/K/V/O projections |
| Learning rate | 2 × 10⁻⁵ |
| Epochs | 1 |
| Per-device batch / gradient accumulation | 1 / 16 |
| Effective batch | 64 across 4 GPUs |
| Optimizer / schedule | AdamW / linear, warmup ratio 0.1 |

### GRPO settings

| Parameter | Thesis methodology / appendix |
| --- | --- |
| Initialization | Med-Qwen2.5-3B-Instruct-SFT-merged |
| Training strategy | FSDP |
| Batch / actor mini-batch / micro-batch | 32 / 16 / 4 |
| Prompt / response / observation limit | 5,000 / 500 / 500 tokens |
| Samples per case | 6 |
| Temperature / top-p / repetition penalty | 0.6 / 0.9 / 1.1 |
| Rollout dtype | float16 |
| KL coefficient / clip ratio / gradient clip | 0.001 / 0.2 / 1.0 |
| Maximum turns / invalid-action threshold | 3 / 4 |
| Retrieved passages | Top-3 |
| Evidence formatting | 1,200 characters, up to 3 sentences |
| Retrieval endpoint | `http://127.0.0.1:8000/retrieve` |
| Save / validation interval | 15 / 10 steps |
| Hardware | 4 GPUs |

I describe a 1,000-sample, 3-epoch run in Chapter 4. My appendix also lists a 5-epoch, 300-step trainer configuration; these settings should be kept separate when comparing runs.

### Reward design

I combine answer correctness, retrieval support, format rewards, and penalties:

- Exact-match answer with valid formatting and a prior search: correctness reward 2.0; exact match with valid formatting: 1.0.
- Partial token coverage: coefficient 0.6; gold answer contained in the response: 0.8.
- Gold answer present in retrieved information: retrieval reward 1.0.
- Valid tag sequence: format reward 0.2; first action is search: bonus 0.2.
- Penalties discourage invalid actions, repeated or generic queries, empty or copied answers, and format violations.

These are the reward settings described in my thesis. The current launcher still needs its reward terms connected consistently; see the training notes below. The case example in the README has a logged reward of 1.7.

## Results

- [Diagnostic ablation, Table 4.1](results/diagnosis.csv): six settings, each evaluated on 897 cases. Percentages are stored on a 0–100 scale.
- [Retriever ablation, Table 4.3](results/retrieval.csv): all 12 retriever/top-k settings. E5 uses k=24 in its second row; BM25 and MedCPT use k=20. My original MAP and NDCG calculations include values above 1; I therefore use Hit Rate, MRR, and Precision for comparison.
- [Retrieved sections, Table 4.4](results/sections.csv): section distribution within the 5,849 perfect exact-match trajectories. Blank counts indicate values I did not report. Case/`case`/`CASE` are merged for the section distribution.

I found an average of 1.53 unique sections per successful case before normalizing section names. Successful patterns include a single discriminative search, symptom-first and diagnosis-second confirmation, outcome-oriented follow-up, case-to-concept verification, imaging-driven search, and differential-diagnosis comparisons.

## Implementation status

This repository includes my medical preprocessing, corpus files, retrieval components, and experimental GRPO launcher. A few parts of the training setup still need to be aligned:

- `scripts/data_process/create_train_data.py` currently samples 500 training and 200 test cases; my thesis experiments use 1,000 training samples and 897 evaluation cases.
- `train_grpo.sh` currently uses 6 GPUs, 8 sampled agents, temperature 0.2, 4 turns, and 2,048/512/512 prompt/response/observation limits. Use the settings above when referring to my thesis experiments.
- The launcher invokes `verl.trainer.main_ppo_format`, whose reward dispatcher does not currently accept the `medcase` data-source identifier emitted by the medical preprocessing script.
- Several launcher overrides are absent from `verl/trainer/config/ppo_trainer.yaml`; Hydra configuration and reward-term wiring need reconciliation before a medical training run.
- `infer.py` defaults to an upstream general-QA checkpoint and question. Thesis SFT/RL checkpoints, the diagnostic evaluation runner, and the complete 5,849-trajectory release are not yet included in this repository.

The tables above record my thesis settings. The current scripts also contain configurations from other experiments.

## Discussion and future work

The main limitations I observed were limited corpus coverage, dependence on one retriever, and reduced trajectory diversity later in RL training. I would like to extend this work with complementary retrievers, conflict resolution between sources, live literature search, and multimodal diagnosis. The current agent works with a fixed case-report corpus and textual descriptions of clinical findings.
