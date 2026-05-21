import os
import json
import time
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from tqdm import tqdm
import shutil


MODEL_PAIRS = [
    {"base": "meta-llama/Llama-3.1-8B",    "instruct": "meta-llama/Llama-3.1-8B-Instruct"},
    {"base": "google/gemma-2-9b",           "instruct": "google/gemma-2-9b-it"},
    {"base": "Qwen/Qwen3.5-9B-Base",        "instruct": "Qwen/Qwen3.5-9B"},
]


MODELS = []
for pair in MODEL_PAIRS:
    MODELS.append({"model_name": pair["base"],     "model_type": "base"})
    MODELS.append({"model_name": pair["instruct"], "model_type": "instruct"})

DATASETS = ["SST2", "SST5", "CoLA", "SNLI", "MNLI",
            "MRPC", "HateXplain", "BoolQ", "TREC", "SWAG"]

DATASETS_DIR     = "datasets"
FEWSHOT_DIR      = "fewshot_examples"
TOKEN_STATS_DIR  = "token_stats"
CHECKPOINT_DIR   = "checkpoints_base_instruct"
OUTPUT_DIR       = "predictions_base_instruct"

SAVE_EVERY       = 200
DEBUG_SAMPLES    = None   
FRESH_START      = False 


NTFY_TOPIC   = ""
NTFY_ENABLED = True

SAMPLE_LOG_N = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATASET_LABELS = {
    "SST2":       ["positive", "negative"],
    "SST5":       ["very negative", "negative", "neutral", "positive", "very positive"],
    "CoLA":       ["acceptable", "unacceptable"],
    "SNLI":       ["entailment", "neutral", "contradiction"],
    "MNLI":       ["entailment", "neutral", "contradiction"],
    "MRPC":       ["paraphrase", "not paraphrase"],
    "SWAG":       ["A", "B", "C", "D"],
    "HateXplain": ["normal", "offensive", "hatespeech"],
    "BoolQ":      ["true", "false"],
    "TREC":       ["abbreviation", "entity", "description", "human", "location", "numeric"],
}

GOLD_LABEL_COL = {
    "SST2":       "label_text",
    "SST5":       "label_text",
    "CoLA":       "label_text",
    "SNLI":       "label_text",
    "MNLI":       "label_text",
    "MRPC":       "label_text",
    "SWAG":       "label_text",
    "HateXplain": "label_text",
    "BoolQ":      "label_text",
    "TREC":       "label_coarse_text",
}

INPUT_FORMATTERS = {
    "SST2":       lambda r: f"Text: {r['sentence']}",
    "SST5":       lambda r: f"Text: {r['sentence']}",
    "CoLA":       lambda r: f"Sentence: {r['sentence']}",
    "SNLI":       lambda r: f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}",
    "MNLI":       lambda r: f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}",
    "MRPC":       lambda r: f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
    "SWAG":       lambda r: (
        f"Beginning: {r['startphrase']}\n"
        f"A: {r['ending0']}\nB: {r['ending1']}\n"
        f"C: {r['ending2']}\nD: {r['ending3']}"
    ),
    "HateXplain": lambda r: f"Text: {r['text']}",
    "BoolQ":      lambda r: f"Passage: {r['passage']}\nQuestion: {r['question']}",
    "TREC":       lambda r: f"Question: {r['text']}",
}

FEWSHOT_LABEL_COL = {
    "SST2":       "label_text",
    "SST5":       "label_text",
    "CoLA":       "label_text",
    "SNLI":       "label_text",
    "MNLI":       "label_text",
    "MRPC":       "label_text",
    "SWAG":       "label_text",
    "HateXplain": "label_text",
    "BoolQ":      "label_text",
    "TREC":       "label_coarse_text",
}

_FEWSHOT_CACHE = {}

def load_fewshot(dataset):
    if dataset not in _FEWSHOT_CACHE:
        _FEWSHOT_CACHE[dataset] = pd.read_csv(
            os.path.join(FEWSHOT_DIR, f"{dataset}.csv")
        )
    return _FEWSHOT_CACHE[dataset]



def build_completion_prompt(dataset, row):
    examples   = load_fewshot(dataset)
    label_col  = FEWSHOT_LABEL_COL[dataset]
    formatter  = INPUT_FORMATTERS[dataset]

    lines = []
    for _, ex in examples.iterrows():
        lines.append(formatter(ex))
        lines.append(f"Label: {ex[label_col]}")
        lines.append("")

    lines.append(formatter(row))
    lines.append("Label:")

    return "\n".join(lines)



def get_checkpoint_path(model_name, dataset):
    short = model_name.split("/")[-1]
    os.makedirs(os.path.join(CHECKPOINT_DIR, short), exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, short, f"{dataset}.csv")


def get_output_path(model_name, dataset):
    short = model_name.split("/")[-1]
    os.makedirs(os.path.join(OUTPUT_DIR, short), exist_ok=True)
    return os.path.join(OUTPUT_DIR, short, f"{dataset}.csv")


def load_progress():
    path = os.path.join(CHECKPOINT_DIR, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    path = os.path.join(CHECKPOINT_DIR, "progress.json")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def get_progress_key(model_name, dataset):
    short = model_name.split("/")[-1]
    return f"{short}__{dataset}"


def get_last_row_id(progress, model_name, dataset):
    key = get_progress_key(model_name, dataset)
    return progress.get(key, -1)


def save_batch(batch, checkpoint_path):
    df = pd.DataFrame(batch)
    if os.path.exists(checkpoint_path):
        df.to_csv(checkpoint_path, mode="a", header=False, index=False)
    else:
        df.to_csv(checkpoint_path, index=False)


def clear_checkpoints():
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    print("Checkpoints cleared.")



def write_sample_log(record, log_path):
    if SAMPLE_LOG_N == 0:
        return
    lines = [
        f"row_id={record['row_id']}",
        f"  gold:    {record['gold_label_text']}",
        f"  pred:    {record['logprob_predicted_label']}  ({'✓' if record['logprob_correct'] else '✗'})",
        "",
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))



def send_ntfy(title, message):
    if not NTFY_ENABLED:
        return
    try:
        import requests
        r = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=5,
        )
        tqdm.write(f"  [ntfy] {r.status_code} — {title}")
    except Exception as e:
        tqdm.write(f"  [ntfy failed] {e}")


def notify_progress(short_name, model_type, dataset,
                    n_processed, total, correct_count, elapsed_min):
    pct = int((n_processed) / total * 100)
    acc = correct_count / n_processed * 100 if n_processed > 0 else 0
    send_ntfy(
        title=f"{short_name} | {dataset} | {pct}%",
        message=(
            f"Type: {model_type}\n"
            f"Progress: {pct}% ({n_processed}/{total})\n"
            f"Acc: {acc:.1f}%\n"
            f"Elapsed: {elapsed_min:.1f}min"
        )
    )


def notify_sample_file(short_name, model_type, dataset, log_path):
    if not os.path.exists(log_path):
        return
    with open(log_path, "r", encoding="utf-8") as f:
        contents = f.read()
    if len(contents.encode("utf-8")) > 3800:
        contents = contents.encode("utf-8")[:3800].decode("utf-8", errors="ignore") + "\n..."
    send_ntfy(
        title=f"Samples | {short_name} | {model_type} | {dataset}",
        message=contents,
    )


def notify_model_complete(short_name, model_type,
                          datasets_done, elapsed_hours, avg_acc):
    send_ntfy(
        title=f"DONE | {short_name} | {model_type}",
        message=(
            f"Datasets: {datasets_done}/10\n"
            f"Avg Acc: {avg_acc:.1f}%\n"
            f"Time: {elapsed_hours:.2f}h"
        )
    )



def load_token_stats():
    return pd.read_csv(os.path.join(TOKEN_STATS_DIR, "summary.csv"))


def get_max_input_tokens(token_stats, dataset, tokenizer_name):
    row = token_stats[
        (token_stats["dataset"]     == dataset) &
        (token_stats["prompt_type"] == "few_shot") &
        (token_stats["tokenizer"]   == tokenizer_name) &
        (token_stats["scope"]       == "full_prompt")
    ]
    return int(row["p99"].values[0]) if len(row) > 0 else 512



def get_debug_subset(df, dataset, n):
    label_col = GOLD_LABEL_COL[dataset]
    labels    = df[label_col].dropna().unique()
    k         = max(1, n // len(labels))
    sampled   = [df[df[label_col] == label].head(k) for label in labels]
    return pd.concat(sampled).reset_index(drop=True)



def get_label_token_sequences(dataset, tokenizer):
    labels = DATASET_LABELS[dataset]
    sequences = {}
    for label in labels:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        sequences[label] = ids
    return sequences



def extract_logprobs(model, prompt_ids, dataset, label_sequences):
    labels     = DATASET_LABELS[dataset]
    prompt_len = len(prompt_ids)
    raw_scores = {}

    for label in labels:
        label_ids  = label_sequences[label]
        full_ids   = prompt_ids + label_ids
        input_tens = torch.tensor([full_ids]).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tens)

        logits    = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        score = 0.0
        for i, tok_id in enumerate(label_ids):
            pos    = prompt_len + i - 1
            score += log_probs[pos, tok_id].item()

        raw_scores[label] = score / len(label_ids)

    score_tensor    = torch.tensor(list(raw_scores.values()))
    probs           = torch.softmax(score_tensor, dim=0).tolist()
    prob_dict       = dict(zip(raw_scores.keys(), probs))

    prob_arr        = np.array(probs)
    entropy         = float(-np.sum(prob_arr * np.log(prob_arr + 1e-10)))

    sorted_probs    = sorted(probs, reverse=True)
    margin          = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    predicted_label = max(prob_dict, key=prob_dict.get)

    result = {}
    for label in labels:
        result[f"logprob_{label.replace(' ', '_')}"] = prob_dict[label]
    result["logprob_predicted_label"] = predicted_label
    result["logprob_label_entropy"]   = entropy
    result["logprob_margin"]          = margin

    return result



def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if FRESH_START:
        clear_checkpoints()

    progress    = load_progress()
    token_stats = load_token_stats()

    model_bar = tqdm(MODELS, desc="Models", position=0)

    for model_entry in model_bar:

        model_name  = model_entry["model_name"]
        model_type  = model_entry["model_type"]
        short_name  = model_name.split("/")[-1]

        model_bar.set_description(f"Model: {short_name} ({model_type})")

        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"Loading: {model_name} [{model_type}]")
        tqdm.write(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, fix_mistral_regex=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        tokenizer_name  = short_name
        model_start     = time.time()
        datasets_done   = 0
        model_acc_sum   = 0.0
        model_acc_count = 0

        dataset_bar = tqdm(DATASETS, desc="  Datasets", position=1, leave=False)

        for dataset in dataset_bar:

            dataset_bar.set_description(f"  Dataset: {dataset}")

            df       = pd.read_csv(os.path.join(DATASETS_DIR, f"{dataset}.csv"))
            gold_col = GOLD_LABEL_COL[dataset]
            df       = df.dropna(subset=[gold_col]).reset_index(drop=True)

            if DEBUG_SAMPLES:
                df = get_debug_subset(df, dataset, DEBUG_SAMPLES)
                tqdm.write(f"  Debug mode: {len(df)} instances")

            label_sequences  = get_label_token_sequences(dataset, tokenizer)
            max_input_tokens = get_max_input_tokens(token_stats, dataset, tokenizer_name)

            checkpoint_path = get_checkpoint_path(model_name, dataset)
            last_row_id     = get_last_row_id(progress, model_name, dataset)

            total  = len(df)
            n_skip = min(last_row_id + 1, total) if last_row_id >= 0 else 0

            batch         = []
            n_processed   = 0
            correct_count = 0
            dataset_start = time.time()

            notify_milestones = set(max(1, int(total * i / 4)) for i in range(1, 5))
            notified_milestones = set()

            log_path = os.path.join(
                OUTPUT_DIR, short_name, f"{dataset}_samples.txt"
            )
            sample_file_is_new = not os.path.exists(log_path)
            if SAMPLE_LOG_N > 0 and (FRESH_START or sample_file_is_new):
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Model: {short_name} [{model_type}] | Dataset: {dataset}\n")
                    f.write("=" * 60 + "\n\n")

            df_todo = df[df.index > last_row_id]
            instance_bar = tqdm(
                df_todo.iterrows(),
                total=len(df_todo),
                desc="      Instances",
                position=2,
                leave=False,
            )

            for idx, row in instance_bar:

                prompt = build_completion_prompt(dataset, row)
                gold   = row[gold_col]

                prompt_ids = tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_input_tokens,
                )

                t0 = time.time()
                try:
                    logprob_results = extract_logprobs(
                        model, prompt_ids, dataset, label_sequences
                    )
                except Exception as e:
                    tqdm.write(f"      logprob error row {idx}: {e}")
                    logprob_results = {
                        f"logprob_{l.replace(' ', '_')}": None
                        for l in DATASET_LABELS[dataset]
                    }
                    logprob_results.update({
                        "logprob_predicted_label": "invalid",
                        "logprob_label_entropy":   None,
                        "logprob_margin":          None,
                    })
                logprob_time = time.time() - t0

                is_correct = int(
                    logprob_results["logprob_predicted_label"] == gold
                )

                record = {
                    "row_id":                 idx,
                    "dataset":                dataset,
                    "model":                  short_name,
                    "model_type":             model_type,
                    **logprob_results,
                    "gold_label":             gold,
                    "gold_label_text":        gold,
                    "logprob_correct":        is_correct,
                    "logprob_inference_time": round(logprob_time, 4),
                }

                batch.append(record)
                n_processed   += 1
                correct_count += is_correct

                if SAMPLE_LOG_N > 0 and n_processed <= SAMPLE_LOG_N:
                    write_sample_log(record, log_path)
                    if n_processed == SAMPLE_LOG_N and sample_file_is_new:
                        notify_sample_file(
                            short_name, model_type, dataset, log_path
                        )

                for milestone in sorted(notify_milestones):
                    if n_skip + n_processed >= milestone and milestone not in notified_milestones:
                        notified_milestones.add(milestone)
                        notify_progress(
                            short_name, model_type, dataset,
                            n_skip + n_processed, total, correct_count,
                            (time.time() - dataset_start) / 60,
                        )

                instance_bar.set_postfix(
                    acc=f"{correct_count / n_processed:.2%}",
                    pred=logprob_results["logprob_predicted_label"][:6],
                )

                if len(batch) % SAVE_EVERY == 0:
                    save_batch(batch, checkpoint_path)
                    key           = get_progress_key(model_name, dataset)
                    progress[key] = idx
                    save_progress(progress)
                    batch = []

            instance_bar.close()

            if batch:
                save_batch(batch, checkpoint_path)
                key           = get_progress_key(model_name, dataset)
                progress[key] = idx
                save_progress(progress)

            if os.path.exists(checkpoint_path):
                shutil.copy(
                    checkpoint_path,
                    get_output_path(model_name, dataset),
                )

            tqdm.write(
                f"      ✓ {short_name} [{model_type}] | {dataset} — "
                f"{n_processed} processed, {n_skip} skipped"
            )

            if n_processed > 0:
                model_acc_sum   += correct_count / n_processed * 100
                model_acc_count += 1

            datasets_done += 1

        dataset_bar.close()

        notify_model_complete(
            short_name, model_type,
            datasets_done,
            (time.time() - model_start) / 3600,
            model_acc_sum / model_acc_count if model_acc_count > 0 else 0.0,
        )

        tqdm.write(f"\n  Unloading {short_name}...")
        del model
        torch.cuda.empty_cache()

    model_bar.close()
    tqdm.write("\nAll models complete.")


if __name__ == "__main__":
    main()