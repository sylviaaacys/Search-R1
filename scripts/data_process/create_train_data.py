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
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse


KEEP_COLUMNS = [
    "case_prompt",
    "text",
    "diagnostic_reasoning",
    "final_diagnosis",
]

'''
prefix = f"""
        From the given medical case, if you need more knowledge, first write your query in <search> </search>. \
        Before giving any final diagnosis in <answer> </answer>, you must search at least once to verify it. \
        The searched results are retrieved from <information> and </information> as a compact evidence summary. \
        Question: {question}\n"""
'''
def make_prefix(dp, template_type):
    question = dp['case_prompt']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """Use a strict action-oriented prompt to reduce chat imitation."""
        prefix = f"""
        Medical diagnosis task. Work one step at a time.
        First, read the case and extract key findings using internal medical knowledge only.
        Then, you must search at least once to verify the points given, output only one short focused search action: <search>query</search>.
        After receiving <information>search results</information>, use the evidence to update your reasoning. You may search again if needed.
        Only when ready, output the final diagnosis as <answer>most likely diagnosis</answer>.
        Rules:
        - Search at least once before the final answer.
        - Do not write chatty completions, instruction restatements, or any free text outside <search>...</search> or <answer>...</answer>.
        - Keep search queries short and specific.
        Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/medcase')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()
    #dataset = datasets.load_dataset('zoulab/FlashRAG_datasets', 'nq')
    dataset = load_dataset("zou-lab/MedCaseReasoning")

    # print(dataset)
    # print(dataset["train"])

    train_dataset = dataset['train'].shuffle(seed=42).select(range(500))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(200))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data_source = "medcase"
            q = "case_prompt"
            golden_answers = "final_diagnosis"

            example[q] = example[q].strip()
            diagnosis_question = "What is the most likely diagnosis? Write it in <answer></answer>"
            if diagnosis_question.lower() not in example[q].lower():
                if example[q] and example[q][-1] not in ".!?":
                    example[q] += "."
                example[q] += f" {diagnosis_question}"
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example[golden_answers],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'case_prompt': example["case_prompt"],
                    'text': example["text"],
                    'diagnostic_reasoning': example["diagnostic_reasoning"],
                    'final_diagnosis': example["final_diagnosis"],
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn('train'),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn('test'),
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"data saved at {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
