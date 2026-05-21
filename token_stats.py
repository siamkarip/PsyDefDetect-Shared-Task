import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

from prompt_templates import get_prompt, PROMPT_REGISTRY



DATASETS_DIR  = "datasets"
OUTPUT_DIR    = "token_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b-it",
    "google/gemma-2-9b",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/deepseek-llm-7b-chat",
    "utter-project/EuroLLM-9B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "ibm-granite/granite-4.1-8b",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "Langboat/Mengzi3-8B-Chat",
    "zai-org/glm-4-9b-chat-hf",
    "tiiuae/Falcon3-10B-Instruct",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
]

PROMPT_TYPES = ["zero_shot", "few_shot", "cot"]

FINGERPRINT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Natural language processing is a subfield of artificial intelligence."
)

INPUT_EXTRACTORS = {
    "SST2":       lambda r: str(r["sentence"]),
    "SST5":       lambda r: str(r["sentence"]),
    "CoLA":       lambda r: str(r["sentence"]),
    "SNLI":       lambda r: f"{r['premise']} {r['hypothesis']}",
    "MNLI":       lambda r: f"{r['premise']} {r['hypothesis']}",
    "MRPC":       lambda r: f"{r['sentence1']} {r['sentence2']}",
    "SWAG":       lambda r: f"{r['startphrase']} {r['ending0']} {r['ending1']} {r['ending2']} {r['ending3']}",
    "HateXplain": lambda r: str(r["text"]),
    "BoolQ":      lambda r: f"{r['passage']} {r['question']}",
    "TREC":       lambda r: str(r["text"]),
}



print("Loading and deduplicating tokenizers...")

tokenizers   = {}   
fingerprints = {}   
groups       = defaultdict(list)  

for model_id in MODELS:
    print(f"  Loading {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizers[model_id] = tok
        fp = str(tok.encode(FINGERPRINT_TEXT))
        fingerprints[model_id] = fp
        groups[fp].append(model_id)
    except Exception as e:
        print(f"  WARNING: Could not load {model_id}: {e}")

representatives = {}   
tok_name_map    = {}   
duplicates_log  = []   

for fp, model_list in groups.items():
    rep = model_list[0]
    representatives[fp] = rep
    short_name = rep.split("/")[-1]
    tok_name_map[rep] = short_name
    if len(model_list) > 1:
        duplicates_log.append({
            "representative": rep,
            "shared_with": ", ".join(model_list[1:])
        })
        print(f"  Shared tokenizer: {rep} == {', '.join(model_list[1:])}")

unique_tokenizers = {rep: tokenizers[rep] for rep in tok_name_map}
print(f"\n{len(unique_tokenizers)} unique tokenizers from {len(MODELS)} models.\n")


pd.DataFrame(duplicates_log).to_csv(
    os.path.join(OUTPUT_DIR, "tokenizer_deduplication.csv"), index=False
)



def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_stats(values):
    arr = np.array(values)
    return {
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "mean":   round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "std":    round(float(arr.std()), 2),
        "p95":    int(np.percentile(arr, 95)),
        "p99":    int(np.percentile(arr, 99)),
    }



summary_rows = []

for dataset in PROMPT_REGISTRY:
    df = pd.read_csv(os.path.join(DATASETS_DIR, f"{dataset}.csv"))
    input_extractor = INPUT_EXTRACTORS[dataset]
    print(f"Processing {dataset} ({len(df)} rows)...")

    for prompt_type in PROMPT_TYPES:
        print(f"  prompt_type={prompt_type}")

        # Build per-instance records
        instance_records = []

        for idx, row in df.iterrows():
            record = {"row_id": idx}

            input_text  = input_extractor(row)
            full_prompt = get_prompt(dataset, prompt_type, row)

            for rep_id, tok in unique_tokenizers.items():
                short = tok_name_map[rep_id]
                record[f"input_only__{short}"]  = count_tokens(tok, input_text)
                record[f"full_prompt__{short}"] = count_tokens(tok, full_prompt)

            instance_records.append(record)

        instance_df = pd.DataFrame(instance_records)
        out_path = os.path.join(OUTPUT_DIR, f"{dataset}_{prompt_type}_tokens.csv")
        instance_df.to_csv(out_path, index=False)
        print(f"    Saved {out_path}")

        # Build summary rows
        for rep_id, tok in unique_tokenizers.items():
            short = tok_name_map[rep_id]

            for scope in ["input_only", "full_prompt"]:
                col = f"{scope}__{short}"
                stats = compute_stats(instance_df[col].tolist())
                summary_rows.append({
                    "dataset":     dataset,
                    "prompt_type": prompt_type,
                    "tokenizer":   short,
                    "scope":       scope,
                    **stats,
                })


summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
print(f"\nSaved summary.csv — {len(summary_df)} rows.")
print("Done.")