from openai import OpenAI
import os
import time
import json
import pandas as pd

from utils_llm import (
    set_seed,
    save_behavior_samples_to_txt,
    get_training_behavior_sequence_samples
)

from functions_for_llm_labeling import (
    get_sample_prompt,
    call_prompts_with_rate_limit,
    parse_llm_result
)

from prompt_design import get_aux_prompt

# API key and base URL for Qwen
api_key = "your api key"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Disable proxy if needed
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Settings
seed = 1226
time_slot_interval = 1
behavior_time_window_len = 15
stride = 1
set_seed(seed)

# Directory to save labeled data
labeled_data_dir = 'llm_labeled_data/'
os.makedirs(labeled_data_dir, exist_ok=True)

# Define dataset list: tuple of (CSV path, output JSON name)
dataset_list = [
    ("datasets/CIC18-DDoS-HOIC/train_set.csv", "HOIC_train.json"),
]

# Main labeling process for one dataset
def label_dataset(csv_path, output_json_name):
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)

    samples = get_training_behavior_sequence_samples(
        df,
        time_slot_interval=time_slot_interval,
        behavior_time_window_len=behavior_time_window_len,
        stride=stride
    )

    sample_prompt_list = get_sample_prompt(samples, number_samples_for_one_prompt=1)
    aux_prompt = get_aux_prompt()

    llm_results = call_prompts_with_rate_limit(
        sample_prompt_list, aux_prompt, client,
        temperature=0,
        max_workers=15,
        max_requests_per_min=15000,
        max_tokens_per_min=1200000
    )

    concept_labels = []
    for result in llm_results:
        parsed = parse_llm_result(result)
        concept_labels.extend(parsed)

    for sample, label in zip(samples, concept_labels):
        sample.pop("flow_indices", None)
        sample["concept_label"] = label

    output_path = os.path.join(labeled_data_dir, output_json_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved labeled data to: {output_path}\n")

# Batch process
for csv_path, output_name in dataset_list:
    try:
        label_dataset(csv_path, output_name)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
