# MedSearch-R1: An Evidence-Informed Synthesis Agent for Medical Deep Research 🩺🧠

<div align="center">
  <img src="public/logo.png" alt="MedSearch-R1 logo" width="300"/>
</div>

<p align="center">
  <a href="docs/thesis.md">Thesis</a> •
  <a href="#preliminary-results">Results</a> •
  <a href="#case-example">Case Example</a> •
  <a href="https://github.com/PeterGriffinJin/Search-R1">Search-R1</a>
</p>

**MedSearch-R1** is my senior thesis project on training a medical diagnosis agent to **search for evidence, refine its reasoning, and answer**. I built it on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/volcengine/verl).

My thesis, **“An Evidence-Informed Synthesis Agent for Medical Deep Research”**, was supervised by **Prof. Zuozhu Liu** at the **ZJU-UIUC Institute** in 2026.

I start with **Qwen2.5-3B-Instruct**, apply **LoRA supervised fine-tuning** on MedCaseReasoning, and then train with **GRPO** and a medical retriever. Rather than retrieving once before answering, the agent can search again as its diagnostic hypothesis develops.

![Medical deep research training pipeline](public/medical_pipeline.png)

*Six sampled rollouts interact with the retriever. Correctness, evidence matching, and format rewards guide GRPO updates, with KL regularization against a reference policy.*

## Links

- [Installation](#installation)
- [Quick start](#quick-start)
- [Preliminary results](#preliminary-results)
- [Case example](#case-example)
- [Inference](#inference)
- [Use your own dataset](#use-your-own-dataset)
- [Use your own search engine](#use-your-own-search-engine)
- [Features](#features)
- [Acknowledge](#acknowledge)
- [Citations](#citations)

## Installation

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment (optional)
For local retrieval, use a separate environment:
```bash
conda create -n retriever python=3.10
conda activate retriever

# install torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

## Quick start

I use **MedCaseReasoning** and a **PubMed case-report corpus**, with **BM25** as the main retriever.

(1) Prepare the medical QA data.

```bash
python scripts/data_process/create_train_data.py --local_dir ./data/medcase
```

The preprocessing script defaults to 500 training and 200 test examples. For my thesis experiments, I used 1,000 training samples and evaluated on 897 cases. See [training details](docs/thesis.md).

(2) Build a Pyserini BM25 index for `corpus/medcase200.jsonl`, then launch the retrieval server with your index path.

```bash
conda activate retriever
python search_r1/search/retrieval_server.py \
    --index_path /path/to/medical/bm25-index \
    --corpus_path corpus/medcase200.jsonl \
    --retriever_name bm25 \
    --topk 3 \
    --host 127.0.0.1 \
    --port 8000
```

(3) Configure the medical GRPO training entry point.

Set the model and data paths in [train_grpo.sh](train_grpo.sh). The medical setup uses `Med-Qwen2.5-3B-Instruct-SFT-merged` and `data/medcase/{train,test}.parquet`. The current script includes settings from other experiments; check the [training notes](docs/thesis.md#implementation-status) before running it.

## Preliminary results

### Medical diagnosis

I evaluated the base model, SFT model, and RL checkpoints on **897 cases**, extracting the final diagnosis from `<answer>...</answer>` tags.

| Setting | Retriever | Answer tag | Normalized EM | Flexible Match |
| --- | --- | ---: | ---: | ---: |
| Base model | None | 98.77% | 3.01% | 7.25% |
| Base model | BM25 | 100.00% | 7.36% | 8.34% |
| SFT model | None | 99.55% | 2.79% | 7.47% |
| SFT model | BM25 | 99.78% | 1.34% | 7.92% |
| Multi-turn RL agent (checkpoint 75) | BM25 | 99.66% | 6.02% | 11.04% |
| Multi-turn RL agent (checkpoint 90) | BM25 | 97.65% | **9.70%** | **17.73%** |

My best checkpoint (90) reaches **9.70% Normalized EM** and **17.73% Flexible Match**, improving over the base model without retrieval by **6.69** and **10.48 percentage points**, respectively. Single-pass retrieval does not consistently improve the SFT model.

I use **Normalized EM** as the main metric and **Flexible Match** for relaxed answer matching. The answer-tag column tracks format adherence. [Full results](docs/results/diagnosis.csv)

### Retriever comparison

I compared BM25 with MedCPT and E5. Below are the top-10 retrieval results.

| Retriever | Hit Rate | MRR | Precision |
| --- | ---: | ---: | ---: |
| BM25 | **0.997** | **0.995** | **0.231** |
| MedCPT | 0.790 | 0.680 | 0.135 |
| E5 | 0.826 | 0.677 | 0.122 |

BM25 works best in these experiments, where distinctive clinical terms provide strong retrieval signals. Increasing top-k from 10 to 64 barely changes its hit rate (0.997 → 0.998), while precision drops from 0.231 to 0.046. I use top-3 retrieval during agent rollouts.

[Full top-k results](docs/results/retrieval.csv). My original MAP and NDCG calculations include values above 1, so I use Hit Rate, MRR, and Precision for the comparison.

### Successful trajectories

I collected **5,849 perfect exact-match trajectories** and examined how the agent searches. Case-oriented sections make up **64.1%** of retrievals, followed by treatment (**12.6%**). The agent often starts with symptoms, then searches again to confirm a candidate diagnosis. Later in training, the sampled trajectories became increasingly similar, showing signs of policy collapse.

[Section distribution](docs/results/sections.csv) · [Training details](docs/thesis.md)

## Case example

Here is one successful trajectory. The agent first searches using clinical findings, then adds the suspected diagnosis to its second query before answering **Bell's palsy**.

```text
Search 1: 35-year-old man acute watery diarrhea high fever altered consciousness
          SARS-CoV-2 PCR positive House-Brackmann grade V facial weakness
Search 2: [same findings] Bell's palsy
Answer:   Bell's Palsy
```

This trajectory receives a reward of **1.7**, with **exact match**, **Cover EM = 1.0**, and the gold answer present in the retrieved evidence.

<details>
<summary>View case example</summary>

![Successful medical search-and-answer example](public/case_example.png)

</details>

## Inference

To try your own case, update `model_id`, `question`, the prompt, and the retrieval endpoint in [infer.py](infer.py). It currently defaults to the original Search-R1 general-QA example. The medical checkpoints and evaluation scripts are not yet included.

## Use your own dataset

I use [create_train_data.py](scripts/data_process/create_train_data.py) to load `zou-lab/MedCaseReasoning` and prepare records in the Search-R1 format:

```python
data = {
    "data_source": "medcase",
    "prompt": [{"role": "user", "content": medical_case_prompt}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": final_diagnosis},
    },
    "extra_info": {"split": split, "index": index},
}
```

The policy emits `<search>query</search>` or `<answer>diagnosis</answer>`; the environment returns evidence in `<information>...</information>`. Retrieved tokens are masked from policy optimization. A final answer should follow at least one search.

## Use your own search engine

The retriever runs separately from RL training and exposes `http://127.0.0.1:8000/retrieve`:

```json
{"queries": ["focused clinical findings"], "topk": 3, "return_scores": true}
```

For your own corpus, prepare JSONL records with `id` and `contents` fields. See the [retriever documentation](docs/retriever.md), [retrieval server](search_r1/search/retrieval_server.py), and [evidence formatter](search_r1/search/evidence_formatter.py).

My experiments focus on BM25, E5, and MedCPT over a fixed corpus. Future directions include combining retrievers, searching live medical literature, and incorporating medical images.

## Features

- Medical cold start with Qwen2.5-3B-Instruct and LoRA SFT.
- GRPO with iterative retrieval and structured search/answer actions.
- Rewards for answer correctness, evidence support, and format adherence.
- Retrieved-token masking during policy optimization.
- Medical retriever and context-size ablations.
- Analysis of successful diagnostic search trajectories.

## Acknowledge

I would like to thank **Prof. Zuozhu Liu** for his supervision, support, and computational resources, and **Fucheng Niu** for his guidance and helpful discussions throughout this project.

Thanks to the authors of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/volcengine/verl) for making their work available. This project also benefits from [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), and [RAGEN](https://github.com/ZihanWang314/RAGEN). The original Search-R1 experiment logs are kept [here](docs/experiment_log.md).

## Citations

```bibtex
@misc{chung2026evidence,
  title={An Evidence-Informed Synthesis Agent for Medical Deep Research},
  author={{Sylvia Chung Yan Shan}},
  howpublished={Undergraduate thesis, Zhejiang University--University of Illinois Urbana-Champaign Institute},
  year={2026}
}

@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}

@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```

## License

[Apache License 2.0](LICENSE).
