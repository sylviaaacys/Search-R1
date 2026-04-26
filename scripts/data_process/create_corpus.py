import json
from datasets import load_dataset

dataset = load_dataset("zou-lab/MedCaseReasoning")

train_dataset = dataset['train']
#shuffle(seed=42).select(range(500))


def build_contents(example):
    fields = [
        ("Case Prompt", example["case_prompt"]),
        ("Text", example["text"]),
        ("Diagnostic Reasoning", example["diagnostic_reasoning"]),
        ("Final Diagnosis", example["final_diagnosis"]),
    ]
    parts = [f"{label}: {value.strip()}" for label, value in fields if value and value.strip()]
    return "\n\n".join(parts)


def make_map_fn(split):
    def process_fn(example, idx):
        return {
            "id": example["pmcid"],
            "contents": build_contents(example)
        }
    return process_fn

train_dataset = train_dataset.map(
    function=make_map_fn('train'),
    with_indices=True,
    remove_columns=train_dataset.column_names
)

with open("corpus.jsonl", "w") as f:
    for d in train_dataset:
        f.write(json.dumps(d) + "\n")
