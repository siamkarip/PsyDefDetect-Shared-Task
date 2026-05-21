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

from prompt_templates import get_prompt, PROMPT_REGISTRY



INSTRUCT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen3.5-9B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/deepseek-llm-7b-chat",
    "utter-project/EuroLLM-9B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "ibm-granite/granite-4.1-8b",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "Langboat/Mengzi3-8B-Chat",
    "zai-org/glm-4-9b-chat-hf",
    "tiiuae/Falcon3-10B-Instruct",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
]

DATASETS     = ["SST2", "SST5", "CoLA", "SNLI", "MNLI",
                "MRPC", "SWAG", "HateXplain", "BoolQ", "TREC"]
PROMPT_TYPES = ["zero_shot", "few_shot", "cot"]

DATASETS_DIR     = "datasets"
LABEL_TOKENS_DIR = "label_tokens"
TOKEN_STATS_DIR  = "token_stats"
CHECKPOINT_DIR   = "checkpoints"
OUTPUT_DIR       = "predictions"

SAVE_EVERY          = 50
DEBUG_SAMPLES       = 50
FRESH_START         = False

PROMPT_MODE         = "all"
SAMPLE_LOG_N        = 10

NTFY_TOPIC          = "llma_status"
NTFY_ENABLED        = True

SC_TEMPERATURE      = 0.7
SC_SAMPLES          = 3
COT_MAX_NEW_TOKENS  = 200

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



def get_checkpoint_path(model_name, dataset, prompt_type):
    short = model_name.split("/")[-1]
    os.makedirs(os.path.join(CHECKPOINT_DIR, short), exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, short, f"{dataset}_{prompt_type}.csv")


def get_output_path(model_name, dataset, prompt_type):
    short = model_name.split("/")[-1]
    os.makedirs(os.path.join(OUTPUT_DIR, short), exist_ok=True)
    return os.path.join(OUTPUT_DIR, short, f"{dataset}_{prompt_type}.csv")


def get_ensemble_path(model_name, dataset):
    short = model_name.split("/")[-1]
    os.makedirs(os.path.join(OUTPUT_DIR, short), exist_ok=True)
    return os.path.join(OUTPUT_DIR, short, f"{dataset}_ensemble.csv")


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


def get_progress_key(model_name, dataset, prompt_type):
    short = model_name.split("/")[-1]
    return f"{short}__{dataset}__{prompt_type}"


def get_last_row_id(progress, model_name, dataset, prompt_type):
    key = get_progress_key(model_name, dataset, prompt_type)
    return progress.get(key, -1)


def save_batch(batch, checkpoint_path):
    df = pd.DataFrame(batch)
    if os.path.exists(checkpoint_path):
        df.to_csv(checkpoint_path, mode="a", header=False, index=False)
    else:
        df.to_csv(checkpoint_path, index=False)


def write_sample_log(record, log_path, prompt):
    if SAMPLE_LOG_N == 0:
        return

    sep = "-" * 60
    lines = [
        f"row_id={record['row_id']}",
        f"  gold:      {record['gold_label_text']}",
        f"  logprob:   {record['logprob_predicted_label']}  ({'✓' if record['logprob_correct'] else '✗'})",
        f"  sc_parsed: {record['sc_sample_1']} | {record['sc_sample_2']} | {record['sc_sample_3']}",
        f"  sc_vote:   {record['sc_majority_vote']}  ({'✓' if record['sc_correct'] else '✗'})",
        "",
        "  [FULL PROMPT]",
        sep,
        prompt,
        sep,
        "  [RAW SC OUTPUT 1]",
        record['_raw_sc_1'],
        "  [RAW SC OUTPUT 2]",
        record['_raw_sc_2'],
        "  [RAW SC OUTPUT 3]",
        record['_raw_sc_3'],
        sep,
        "",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def clear_checkpoints():
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    print("Checkpoints cleared.")



def compute_ensemble(model_name, dataset, active_prompt_types):
    short = model_name.split("/")[-1]

    prompt_preds = {}
    for pt in active_prompt_types:
        path = get_output_path(model_name, dataset, pt)
        if not os.path.exists(path):
            tqdm.write(f"      [ensemble] missing {pt} predictions, skipping ensemble")
            return
        df = pd.read_csv(path)
        prompt_preds[pt] = df.set_index("row_id")[["sc_majority_vote",
                                                     "gold_label_text",
                                                     "sc_correct"]].copy()

    common_ids = None
    for pt, df in prompt_preds.items():
        common_ids = df.index if common_ids is None else common_ids.intersection(df.index)

    if len(common_ids) == 0:
        tqdm.write(f"      [ensemble] no common row_ids for {dataset}, skipping")
        return

    rows = []
    for row_id in common_ids:
        stage1_votes = []
        for pt in active_prompt_types:
            vote = str(prompt_preds[pt].loc[row_id, "sc_majority_vote"]).strip().lower()
            stage1_votes.append(vote)

        gold = prompt_preds[active_prompt_types[0]].loc[row_id, "gold_label_text"]
        valid_votes = [v for v in stage1_votes if v != "invalid"]

        if len(valid_votes) == 0:
            ensemble_pred = "invalid"
            ensemble_agreement = 0
        else:
            counts = Counter(valid_votes)
            ensemble_pred      = counts.most_common(1)[0][0]
            ensemble_agreement = counts[ensemble_pred]

        n_types = len(active_prompt_types)
        cy = Counter(stage1_votes)
        entropy = 0.0
        if n_types > 1:
            for label, count in cy.items():
                p = count / n_types
                if p > 0:
                    entropy -= p * np.log(p)
            entropy /= np.log(n_types)   

        rows.append({
            "row_id":             row_id,
            "dataset":            dataset,
            "model":              short,
            **{f"vote_{pt}": prompt_preds[pt].loc[row_id, "sc_majority_vote"]
               for pt in active_prompt_types},
            "ensemble_pred":      ensemble_pred,
            "ensemble_agreement": ensemble_agreement,
            "vote_entropy":       round(float(entropy), 4),
            "gold_label_text":    gold,
            "ensemble_correct":   int(ensemble_pred == str(gold).strip().lower()),
        })

    df_out = pd.DataFrame(rows)
    out_path = get_ensemble_path(model_name, dataset)
    df_out.to_csv(out_path, index=False)

    valid = df_out[df_out["ensemble_pred"] != "invalid"]
    acc   = valid["ensemble_correct"].mean() * 100 if len(valid) > 0 else 0.0
    tqdm.write(
        f"      ✓ Ensemble | {short} | {dataset} — "
        f"{len(df_out)} instances, acc={acc:.1f}%"
    )



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


def notify_progress(short_name, dataset, prompt_type,
                    n_processed, total, correct_count, elapsed_min):
    pct = int(n_processed / total * 100)
    acc = correct_count / n_processed * 100 if n_processed > 0 else 0
    send_ntfy(
        title=f"{short_name} | {dataset} | {pct}%",
        message=(
            f"Prompt: {prompt_type}\n"
            f"Progress: {pct}% ({n_processed}/{total})\n"
            f"Acc: {acc:.1f}%\n"
            f"Elapsed: {elapsed_min:.1f}min"
        )
    )


def notify_sample_file(short_name, dataset, prompt_type, log_path):
    if not os.path.exists(log_path):
        return
    with open(log_path, "r", encoding="utf-8") as f:
        contents = f.read()
    if len(contents.encode("utf-8")) > 3800:
        contents = contents.encode("utf-8")[:3800].decode("utf-8", errors="ignore") + "\n..."
    send_ntfy(
        title=f"Samples | {short_name} | {dataset} | {prompt_type}",
        message=contents,
    )


def notify_model_complete(short_name, prompt_type,
                          datasets_done, elapsed_hours, avg_acc):
    send_ntfy(
        title=f"{short_name} done",
        message=(
            f"Prompt: {prompt_type}\n"
            f"Datasets: {datasets_done}/10\n"
            f"Avg Acc: {avg_acc:.1f}%\n"
            f"Time: {elapsed_hours:.2f}h"
        )
    )



def get_debug_subset(df, dataset, n):
    label_col = GOLD_LABEL_COL[dataset]
    labels    = df[label_col].dropna().unique()
    sampled   = [df[df[label_col] == label].head(n) for label in labels]
    return pd.concat(sampled).reset_index(drop=True)



def load_token_stats():
    return pd.read_csv(os.path.join(TOKEN_STATS_DIR, "summary.csv"))


def load_max_new_tokens():
    df = pd.read_csv(os.path.join(TOKEN_STATS_DIR, "max_new_tokens_global.csv"))
    return dict(zip(df["dataset"], df["global_safe_max_new_tokens"]))


def build_token_lookup(token_stats, dataset, tokenizer_name):
    lookup = {}
    for pt in PROMPT_TYPES:
        row = token_stats[
            (token_stats["dataset"]     == dataset) &
            (token_stats["prompt_type"] == pt) &
            (token_stats["tokenizer"]   == tokenizer_name) &
            (token_stats["scope"]       == "full_prompt")
        ]
        lookup[pt] = int(row["p99"].values[0]) if len(row) > 0 else 512
    return lookup



DISABLE_THINKING   = {"Qwen3.5-9B"}
USE_SYSTEM_MESSAGE = {"Ministral-3-8B-Instruct-2512-BF16"}


def apply_prompt_template(tokenizer, prompt, model_short_name):
    if tokenizer.chat_template is not None:
        if model_short_name in USE_SYSTEM_MESSAGE:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if model_short_name in DISABLE_THINKING:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt


def get_label_token_sequences(dataset, tokenizer):
    labels = DATASET_LABELS[dataset]
    sequences = {}
    for label in labels:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        sequences[label] = ids
    return sequences



def extract_logprobs(model, tokenizer, prompt_ids, dataset, label_sequences):
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



def parse_label(text, labels):
    text = text.strip().lower()
    if "label:" in text:
        text = text.split("label:")[-1].strip()
    for label in labels:
        if text == label.lower(): return label
    for label in labels:
        if text.startswith(label.lower()): return label
    for label in labels:
        if label.lower() in text: return label
    return "invalid"


def run_self_consistency(model, tokenizer, input_ids, dataset,
                         is_cot, sc_max_new_tokens):
    labels  = DATASET_LABELS[dataset]
    max_new = COT_MAX_NEW_TOKENS if is_cot else sc_max_new_tokens
    inputs  = {"input_ids": input_ids.to(DEVICE)}

    samples     = []
    raw_outputs = []
    for _ in range(SC_SAMPLES):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                max_length=None,
                do_sample=True,
                temperature=SC_TEMPERATURE,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        generated  = tokenizer.decode(new_tokens, skip_special_tokens=True)
        label      = parse_label(generated, labels)
        samples.append(label)
        raw_outputs.append(generated)

    valid_samples = [s for s in samples if s != "invalid"]
    if len(valid_samples) == 0:
        majority_vote = "invalid"
        agreement     = 0
    else:
        counts        = Counter(valid_samples)
        majority_vote = counts.most_common(1)[0][0]
        agreement     = counts[majority_vote]

    return {
        "sc_sample_1":      samples[0],
        "sc_sample_2":      samples[1],
        "sc_sample_3":      samples[2],
        "sc_majority_vote": majority_vote,
        "sc_agreement":     agreement,
        "_raw_sc_1":        raw_outputs[0],
        "_raw_sc_2":        raw_outputs[1],
        "_raw_sc_3":        raw_outputs[2],
    }



def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if FRESH_START:
        clear_checkpoints()

    progress        = load_progress()
    token_stats     = load_token_stats()
    max_new_tok_map = load_max_new_tokens()

    all_prompt_types = ["zero_shot", "few_shot", "cot"]
    if PROMPT_MODE == "all":
        active_prompt_types = all_prompt_types
    elif PROMPT_MODE in all_prompt_types:
        active_prompt_types = [PROMPT_MODE]
    else:
        raise ValueError(f"Invalid PROMPT_MODE: '{PROMPT_MODE}'")

    model_bar = tqdm(INSTRUCT_MODELS, desc="Models", position=0)

    for model_name in model_bar:
        short_name = model_name.split("/")[-1]
        model_bar.set_description(f"Model: {short_name}")

        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"Loading: {model_name}")
        tqdm.write(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, fix_mistral_regex=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
            )
        except ValueError:
            from transformers import Mistral3ForConditionalGeneration
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
                quantization_config=None, ignore_mismatched_sizes=True,
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

            label_sequences   = get_label_token_sequences(dataset, tokenizer)
            sc_max_new_tokens = max_new_tok_map.get(dataset, 6)
            token_lookup      = build_token_lookup(token_stats, dataset, tokenizer_name)

            prompt_bar = tqdm(active_prompt_types, desc="    Prompts", position=2, leave=False)

            for prompt_type in prompt_bar:
                prompt_bar.set_description(f"    Prompt: {prompt_type}")

                checkpoint_path  = get_checkpoint_path(model_name, dataset, prompt_type)
                last_row_id      = get_last_row_id(progress, model_name, dataset, prompt_type)
                max_input_tokens = token_lookup[prompt_type] + 100
                is_cot           = (prompt_type == "cot")

                total  = len(df)
                n_skip = min(last_row_id + 1, total) if last_row_id >= 0 else 0

                batch         = []
                n_processed   = 0
                correct_count = 0
                dataset_start = time.time()

                notify_milestones   = set(max(1, int(total * i / 4)) for i in range(1, 5))
                notified_milestones = set()

                log_path = os.path.join(
                    OUTPUT_DIR, short_name,
                    f"{dataset}_{prompt_type}_samples.txt"
                )
                if SAMPLE_LOG_N > 0 and (FRESH_START or not os.path.exists(log_path)):
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"Model: {short_name} | Dataset: {dataset} | Prompt: {prompt_type}\n")
                        f.write("=" * 60 + "\n\n")

                df_todo = df[df.index > last_row_id]
                instance_bar = tqdm(
                    df_todo.iterrows(), total=len(df_todo),
                    desc=f"      Instances", position=3, leave=False,
                )

                for idx, row in instance_bar:
                    prompt = get_prompt(dataset, prompt_type, row)
                    prompt = apply_prompt_template(tokenizer, prompt, short_name)
                    gold   = row[gold_col]

                    prompt_ids = tokenizer.encode(
                        prompt, add_special_tokens=False,
                        truncation=True, max_length=max_input_tokens,
                    )
                    input_ids = torch.tensor([prompt_ids])

                    t0 = time.time()
                    try:
                        logprob_results = extract_logprobs(
                            model, tokenizer, prompt_ids, dataset, label_sequences,
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

                    t1 = time.time()
                    try:
                        sc_results = run_self_consistency(
                            model, tokenizer, input_ids, dataset,
                            is_cot, sc_max_new_tokens,
                        )
                    except Exception as e:
                        tqdm.write(f"      SC error row {idx}: {e}")
                        sc_results = {
                            "sc_sample_1":      "invalid",
                            "sc_sample_2":      "invalid",
                            "sc_sample_3":      "invalid",
                            "sc_majority_vote": "invalid",
                            "sc_agreement":     0,
                            "_raw_sc_1":        "",
                            "_raw_sc_2":        "",
                            "_raw_sc_3":        "",
                        }
                    sc_time = time.time() - t1

                    is_correct = int(
                        logprob_results["logprob_predicted_label"] == gold
                    )

                    record = {
                        "row_id":                 idx,
                        "dataset":                dataset,
                        "model":                  short_name,
                        "prompt_type":            prompt_type,
                        **logprob_results,
                        **sc_results,
                        "gold_label":             gold,
                        "gold_label_text":        gold,
                        "logprob_correct":        is_correct,
                        "sc_correct":             int(
                            sc_results["sc_majority_vote"] == gold
                        ),
                        "logprob_inference_time": round(logprob_time, 4),
                        "sc_inference_time":      round(sc_time, 4),
                    }

                    csv_record = {k: v for k, v in record.items()
                                  if not k.startswith("_raw_")}
                    batch.append(csv_record)
                    n_processed   += 1
                    correct_count += is_correct

                    if SAMPLE_LOG_N > 0 and n_processed <= SAMPLE_LOG_N:
                        write_sample_log(record, log_path, prompt)
                        if n_processed == SAMPLE_LOG_N:
                            notify_sample_file(short_name, dataset, prompt_type, log_path)

                    for milestone in sorted(notify_milestones):
                        if n_processed >= milestone and milestone not in notified_milestones:
                            notified_milestones.add(milestone)
                            notify_progress(
                                short_name, dataset, prompt_type,
                                n_skip + n_processed, total, correct_count,
                                (time.time() - dataset_start) / 60,
                            )

                    instance_bar.set_postfix(
                        acc=f"{correct_count / n_processed:.2%}",
                        pred=logprob_results["logprob_predicted_label"][:6],
                    )

                    if len(batch) % SAVE_EVERY == 0:
                        save_batch(batch, checkpoint_path)
                        key           = get_progress_key(model_name, dataset, prompt_type)
                        progress[key] = idx
                        save_progress(progress)
                        batch = []

                instance_bar.close()

                if batch:
                    save_batch(batch, checkpoint_path)
                    key           = get_progress_key(model_name, dataset, prompt_type)
                    progress[key] = idx
                    save_progress(progress)

                if os.path.exists(checkpoint_path):
                    shutil.copy(checkpoint_path, get_output_path(model_name, dataset, prompt_type))

                tqdm.write(
                    f"      ✓ {short_name} | {dataset} | {prompt_type} — "
                    f"{n_processed} processed, {n_skip} skipped"
                )

                if n_processed > 0:
                    model_acc_sum   += correct_count / n_processed * 100
                    model_acc_count += 1

            prompt_bar.close()


            if len(active_prompt_types) > 1:
                compute_ensemble(model_name, dataset, active_prompt_types)

            datasets_done += 1

        dataset_bar.close()

        notify_model_complete(
            short_name, PROMPT_MODE, datasets_done,
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