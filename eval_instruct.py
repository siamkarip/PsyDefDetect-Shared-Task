import os
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm



PREDICTIONS_DIR = "predictions"
DATASETS_DIR    = "datasets"
OUTPUT_DIR      = os.path.join("eval_results", "instruct")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVAL_PROMPT_MODE = "all"

INSTRUCT_MODELS = [
    "Llama-3.1-8B-Instruct",
    "gemma-2-9b-it",
    "Qwen3.5-9B",
    "Mistral-7B-Instruct-v0.3",
    "deepseek-llm-7b-chat",
    "EuroLLM-9B-Instruct",
    "EXAONE-3.5-7.8B-Instruct",
    "Ministral-3-8B-Instruct-2512-BF16",
    "granite-4.1-8b",
    "kanana-1.5-8b-instruct-2505",
    "Mengzi3-8B-Chat",
    "glm-4-9b-chat-hf",
    "Falcon3-10B-Instruct",
    "SOLAR-10.7B-Instruct-v1.0",
]

DATASETS = ["SST2", "SST5", "CoLA", "SNLI", "MNLI",
            "MRPC", "SWAG", "HateXplain", "BoolQ", "TREC"]

PROMPT_TYPES = ["zero_shot", "few_shot", "cot"]

PROMPT_PREFIX = {
    "zero_shot": "zs",
    "few_shot":  "fs",
    "cot":       "cot",
}

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

N_BOOTSTRAP     = 10000
N_ENTROPY_BINS  = 10
CONFIDENCE_PCT  = 33
ALPHA           = 0.05


def save_csv(df, basename):

    if EVAL_PROMPT_MODE == "all" and "prompt_type" in df.columns:
        for pt, group in df.groupby("prompt_type"):
            prefix = PROMPT_PREFIX.get(pt, pt)
            path = os.path.join(OUTPUT_DIR, f"{prefix}_{basename}")
            group.to_csv(path, index=False)
            print(f"  Saved {prefix}_{basename} — {len(group)} rows")
    else:
        path = os.path.join(OUTPUT_DIR, basename)
        df.to_csv(path, index=False)
        print(f"  Saved {basename} — {len(df)} rows")


def normalize_labels(df, dataset):
    if dataset == "BoolQ":
        for col in ["gold_label_text", "gold_label"]:
            if col in df.columns:
                df[col] = df[col].map(
                    {True: "true", False: "false",
                     "True": "true", "False": "false"}
                ).fillna(df[col].astype(str).str.lower())

    for col in ["gold_label_text", "gold_label",
                "logprob_predicted_label",
                "sc_majority_vote",
                "sc_sample_1", "sc_sample_2", "sc_sample_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("invalid").astype(str).str.strip()
            if dataset != "SWAG":
                df[col] = df[col].str.lower()

    return df


def load_predictions(model, dataset, prompt_type):
    path = os.path.join(PREDICTIONS_DIR, model, f"{dataset}_{prompt_type}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = normalize_labels(df, dataset)
    df["sc_correct"] = (
        df["sc_majority_vote"] == df["gold_label_text"]
    ).astype(int)
    
    df["logprob_correct"] = (
        df["logprob_predicted_label"].astype(str).str.lower() ==
        df["gold_label_text"].astype(str).str.lower()
    ).astype(int)
    
    return df


def get_active_prompt_types():
    if EVAL_PROMPT_MODE == "all":
        return PROMPT_TYPES
    elif EVAL_PROMPT_MODE in PROMPT_TYPES:
        return [EVAL_PROMPT_MODE]
    else:
        raise ValueError(f"Invalid EVAL_PROMPT_MODE: {EVAL_PROMPT_MODE}")



def stratified_bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, alpha=ALPHA):
    classes = np.unique(y_true)
    indices_by_class = {c: np.where(np.array(y_true) == c)[0] for c in classes}

    scores = []
    for _ in range(n):
        resampled_idx = np.concatenate([
            np.random.choice(idx, size=len(idx), replace=True)
            for idx in indices_by_class.values()
        ])
        y_t = np.array(y_true)[resampled_idx]
        y_p = np.array(y_pred)[resampled_idx]
        try:
            scores.append(metric_fn(y_t, y_p))
        except Exception:
            continue

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)



def mcnemar_test(correct_a, correct_b):
    b = np.sum((np.array(correct_a) == 1) & (np.array(correct_b) == 0))
    c = np.sum((np.array(correct_a) == 0) & (np.array(correct_b) == 1))
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return float(p_value)


def compute_accuracy_f1():
    print("Computing accuracy and F1...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                y_true  = df["gold_label_text"].astype(str).tolist()
                y_pred  = df["sc_majority_vote"].astype(str).tolist()
                labels  = [str(l) for l in DATASET_LABELS[dataset]]
                n_total = len(df)

                acc = accuracy_score(y_true, y_pred)

                valid_mask  = [p not in ("invalid", "nan") for p in y_pred]
                y_true_v    = [t for t, v in zip(y_true, valid_mask) if v]
                y_pred_v    = [p for p, v in zip(y_pred, valid_mask) if v]
                n_valid     = sum(valid_mask)
                invalid_rate = 1 - n_valid / n_total

                f1_macro    = f1_score(y_true_v, y_pred_v, average="macro",
                                       labels=labels, zero_division=0) if n_valid > 0 else 0.0
                f1_weighted = f1_score(y_true_v, y_pred_v, average="weighted",
                                       labels=labels, zero_division=0) if n_valid > 0 else 0.0

                rows.append({
                    "model":        model,
                    "dataset":      dataset,
                    "prompt_type":  prompt_type,
                    "accuracy":     round(acc, 4),
                    "f1_macro":     round(f1_macro, 4),
                    "f1_weighted":  round(f1_weighted, 4),
                    "invalid_rate": round(invalid_rate, 4),
                    "n_valid":      n_valid,
                    "n_total":      n_total,
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "accuracy.csv")
    return df_out


def compute_bootstrap_ci():
    print("Computing bootstrap CI...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS, desc="  Models"):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                y_true = df["gold_label_text"].astype(str).tolist()
                y_pred = df["sc_majority_vote"].astype(str).tolist()

                if len(y_true) < 10:
                    continue

                labels = [str(l) for l in DATASET_LABELS[dataset]]

                acc_lower, acc_upper = stratified_bootstrap_ci(
                    y_true, y_pred,
                    lambda t, p: accuracy_score(t, p)
                )

                valid_mask = [p not in ("invalid", "nan") for p in y_pred]
                y_true_v   = [t for t, v in zip(y_true, valid_mask) if v]
                y_pred_v   = [p for p, v in zip(y_pred, valid_mask) if v]

                f1_lower, f1_upper = (0.0, 0.0)
                if len(y_true_v) >= 10:
                    f1_lower, f1_upper = stratified_bootstrap_ci(
                        y_true_v, y_pred_v,
                        lambda t, p: f1_score(t, p, average="macro",
                                              labels=labels, zero_division=0)
                    )

                rows.append({
                    "model":        model,
                    "dataset":      dataset,
                    "prompt_type":  prompt_type,
                    "acc_ci_lower": round(acc_lower, 4),
                    "acc_ci_upper": round(acc_upper, 4),
                    "f1_ci_lower":  round(f1_lower, 4),
                    "f1_ci_upper":  round(f1_upper, 4),
                    "n":            len(y_true),
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "bootstrap_ci.csv")
    return df_out


def compute_mcnemar():
    print("Computing McNemar's tests...")
    rows = []
    active_pts = get_active_prompt_types()
    model_pairs = list(combinations(INSTRUCT_MODELS, 2))

    for dataset in tqdm(DATASETS):
        for prompt_type in active_pts:
            model_preds = {}
            for model in INSTRUCT_MODELS:
                df = load_predictions(model, dataset, prompt_type)
                if df is not None and len(df) > 0:
                    model_preds[model] = df.set_index("row_id")["sc_correct"]

            p_values  = []
            pair_rows = []

            for model_a, model_b in model_pairs:
                if model_a not in model_preds or model_b not in model_preds:
                    continue
                common_idx = model_preds[model_a].index.intersection(
                    model_preds[model_b].index
                )
                if len(common_idx) == 0:
                    continue

                correct_a = model_preds[model_a].loc[common_idx].tolist()
                correct_b = model_preds[model_b].loc[common_idx].tolist()
                p_val     = mcnemar_test(correct_a, correct_b)
                p_values.append(p_val)
                pair_rows.append({
                    "dataset":     dataset,
                    "prompt_type": prompt_type,
                    "model_a":     model_a,
                    "model_b":     model_b,
                    "p_value":     round(p_val, 6),
                    "n_common":    len(common_idx),
                })

            if len(p_values) > 0:
                reject, p_corrected, _, _ = multipletests(
                    p_values, alpha=ALPHA, method="fdr_bh"
                )
                for i, row in enumerate(pair_rows):
                    row["p_value_corrected"] = round(p_corrected[i], 6)
                    row["significant"]       = bool(reject[i])
                    rows.append(row)

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "mcnemar.csv")
    return df_out


def compute_dampening_ratio():
    print("Computing dampening ratio (SST5)...")
    rows = []
    active_pts = get_active_prompt_types()

    extreme_labels  = ["very negative", "very positive"]
    moderate_labels = ["negative",      "positive"]

    for model in tqdm(INSTRUCT_MODELS):
        for prompt_type in active_pts:
            df = load_predictions(model, "SST5", prompt_type)
            if df is None or len(df) == 0:
                continue

            for extreme, moderate in zip(extreme_labels, moderate_labels):
                subset = df[df["gold_label_text"] == extreme]
                if len(subset) == 0:
                    continue

                predicted  = subset["sc_majority_vote"]
                n_correct  = (predicted == extreme).sum()
                n_dampened = (predicted == moderate).sum()
                n_other    = len(subset) - n_correct - n_dampened

                rows.append({
                    "model":           model,
                    "prompt_type":     prompt_type,
                    "gold_label":      extreme,
                    "n_total":         len(subset),
                    "n_correct":       int(n_correct),
                    "n_dampened":      int(n_dampened),
                    "n_other":         int(n_other),
                    "dampening_ratio": round(n_dampened / n_correct, 4) if n_correct > 0 else None,
                    "dampening_rate":  round(n_dampened / len(subset), 4),
                    "correct_rate":    round(n_correct / len(subset), 4),
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "dampening_ratio.csv")
    return df_out


def compute_safety_hallucination():
    print("Computing safety hallucination (HateXplain)...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for prompt_type in active_pts:
            df = load_predictions(model, "HateXplain", prompt_type)
            if df is None or len(df) == 0:
                continue

            normal_mask = df["gold_label_text"] == "normal"
            hate_mask   = df["gold_label_text"] == "hatespeech"
            off_mask    = df["gold_label_text"] == "offensive"

            fpr = (
                df[normal_mask]["sc_majority_vote"]
                .isin(["hatespeech", "offensive"]).mean()
            ) if normal_mask.sum() > 0 else None

            fnr = (
                df[hate_mask]["sc_majority_vote"] == "normal"
            ).mean() if hate_mask.sum() > 0 else None

            safety_collapse = (
                df["sc_majority_vote"] == "hatespeech"
            ).mean()

            rows.append({
                "model":            model,
                "prompt_type":      prompt_type,
                "fpr":              round(float(fpr), 4) if fpr is not None else None,
                "fnr":              round(float(fnr), 4) if fnr is not None else None,
                "safety_collapse":  round(float(safety_collapse), 4),
                "n_normal_gold":    int(normal_mask.sum()),
                "n_hate_gold":      int(hate_mask.sum()),
                "n_offensive_gold": int(off_mask.sum()),
                "n_total":          len(df),
            })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "safety_hallucination.csv")
    return df_out


def compute_entailment_bias():
    print("Computing entailment bias (SNLI/MNLI)...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for nli_dataset in ["SNLI", "MNLI"]:
            for prompt_type in active_pts:
                df = load_predictions(model, nli_dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                for gold_label in ["neutral", "contradiction"]:
                    subset = df[df["gold_label_text"] == gold_label]
                    if len(subset) == 0:
                        continue

                    bias_rate = (
                        subset["sc_majority_vote"] == "entailment"
                    ).mean()

                    rows.append({
                        "model":               model,
                        "dataset":             nli_dataset,
                        "prompt_type":         prompt_type,
                        "gold_label":          gold_label,
                        "entailment_bias_rate": round(float(bias_rate), 4),
                        "n_total":             len(subset),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "entailment_bias.csv")
    return df_out


def compute_entropy_accuracy():
    print("Computing entropy-accuracy relationship...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                df = df.dropna(subset=["logprob_label_entropy"])
                if len(df) == 0:
                    continue

                df["entropy_bin"] = pd.qcut(
                    df["logprob_label_entropy"],
                    q=N_ENTROPY_BINS,
                    labels=False,
                    duplicates="drop",
                )

                for bin_id, group in df.groupby("entropy_bin"):
                    rows.append({
                        "model":        model,
                        "dataset":      dataset,
                        "prompt_type":  prompt_type,
                        "entropy_bin":  int(bin_id),
                        "entropy_min":  round(group["logprob_label_entropy"].min(), 4),
                        "entropy_max":  round(group["logprob_label_entropy"].max(), 4),
                        "entropy_mean": round(group["logprob_label_entropy"].mean(), 4),
                        "accuracy":     round(group["logprob_correct"].mean(), 4),
                        "n":            len(group),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "entropy_accuracy.csv")
    return df_out


def compute_confidence_correctness():
    print("Computing confidence-correctness decomposition...")
    rows = []
    active_pts = get_active_prompt_types()

    for dataset in tqdm(DATASETS):
        for prompt_type in active_pts:
            all_margins = []
            model_dfs   = {}

            for model in INSTRUCT_MODELS:
                df = load_predictions(model, dataset, prompt_type)
                if df is not None and len(df) > 0:
                    df = df.dropna(subset=["logprob_margin"])
                    model_dfs[model] = df
                    all_margins.extend(df["logprob_margin"].tolist())

            if len(all_margins) == 0:
                continue

            threshold = np.percentile(all_margins, CONFIDENCE_PCT)

            for model, df in model_dfs.items():
                high_conf = df["logprob_margin"] >= threshold
                correct   = df["logprob_correct"] == 1

                tp = int((high_conf & correct).sum())
                oe = int((high_conf & ~correct).sum())
                uc = int((~high_conf & correct).sum())
                tn = int((~high_conf & ~correct).sum())
                n  = len(df)

                rows.append({
                    "model":                    model,
                    "dataset":                  dataset,
                    "prompt_type":              prompt_type,
                    "threshold":                round(float(threshold), 4),
                    "true_positive":            tp,
                    "overconfident_error":      oe,
                    "underconfident_correct":   uc,
                    "true_negative":            tn,
                    "overconfident_error_rate": round(oe / n, 4),
                    "n":                        n,
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "confidence_correctness.csv")
    return df_out


def compute_confidence_correctness_sc():
    print("Computing confidence-correctness decomposition (self-consistency)...")
    rows = []
    active_pts = get_active_prompt_types()

    for dataset in tqdm(DATASETS):
        for model in INSTRUCT_MODELS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                df = df.dropna(subset=["sc_agreement", "sc_correct"])
                n = len(df)
                if n == 0:
                    continue

                for threshold in [2, 3]:
                    high_conf = df["sc_agreement"] >= threshold
                    correct   = df["sc_correct"] == 1

                    tp = int((high_conf & correct).sum())
                    oe = int((high_conf & ~correct).sum())
                    uc = int((~high_conf & correct).sum())
                    tn = int((~high_conf & ~correct).sum())

                    rows.append({
                        "model":                    model,
                        "dataset":                  dataset,
                        "prompt_type":              prompt_type,
                        "agreement_threshold":      threshold,
                        "true_positive":            tp,
                        "overconfident_error":      oe,
                        "underconfident_correct":   uc,
                        "true_negative":            tn,
                        "overconfident_error_rate": round(oe / n, 4),
                        "n":                        n,
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "confidence_correctness_sc.csv")
    return df_out


def compute_cross_task_correlation():
    print("Computing cross-task failure correlation...")
    active_pts = get_active_prompt_types()
    all_rows   = []

    for model in tqdm(INSTRUCT_MODELS):
        for prompt_type in active_pts:
            dataset_correct = {}
            for dataset in DATASETS:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue
                dataset_correct[dataset] = df.set_index("row_id")["sc_correct"]

            if len(dataset_correct) < 2:
                continue

            for ds_a, ds_b in combinations(dataset_correct.keys(), 2):
                common_idx = dataset_correct[ds_a].index.intersection(
                    dataset_correct[ds_b].index
                )
                if len(common_idx) < 10:
                    continue

                corr = np.corrcoef(
                    dataset_correct[ds_a].loc[common_idx].values,
                    dataset_correct[ds_b].loc[common_idx].values
                )[0, 1]

                all_rows.append({
                    "model":       model,
                    "prompt_type": prompt_type,
                    "dataset_a":   ds_a,
                    "dataset_b":   ds_b,
                    "correlation": round(float(corr), 4),
                    "n_common":    len(common_idx),
                })

    df_out = pd.DataFrame(all_rows)
    save_csv(df_out, "cross_task_correlation.csv")
    return df_out


def compute_annotator_simulation():
    print("Computing annotator simulation score (HateXplain)...")
    rows = []
    active_pts = get_active_prompt_types()

    hatexplain_df = pd.read_csv(os.path.join(DATASETS_DIR, "HateXplain.csv"))
    label_map     = {0: "normal", 1: "offensive", 2: "hatespeech"}

    for col in ["annotator_label_1", "annotator_label_2", "annotator_label_3"]:
        if col in hatexplain_df.columns:
            hatexplain_df[f"{col}_text"] = hatexplain_df[col].map(label_map)

    for model in tqdm(INSTRUCT_MODELS):
        for prompt_type in active_pts:
            df = load_predictions(model, "HateXplain", prompt_type)
            if df is None or len(df) == 0:
                continue

            merged = df.merge(
                hatexplain_df[["annotator_label_1_text",
                               "annotator_label_2_text",
                               "annotator_label_3_text"]].reset_index().rename(
                    columns={"index": "row_id"}
                ),
                on="row_id", how="left"
            )

            for ann_col in ["annotator_label_1_text",
                            "annotator_label_2_text",
                            "annotator_label_3_text"]:
                if ann_col not in merged.columns:
                    continue

                valid = merged.dropna(subset=[ann_col])
                if len(valid) == 0:
                    continue

                agreement = (
                    valid["sc_majority_vote"] == valid[ann_col]
                ).mean()

                rows.append({
                    "model":          model,
                    "prompt_type":    prompt_type,
                    "annotator":      ann_col,
                    "agreement_rate": round(float(agreement), 4),
                    "n":              len(valid),
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "annotator_simulation.csv")
    return df_out


def compute_per_class_accuracy():
    print("Computing per-class accuracy...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                labels = [str(l) for l in DATASET_LABELS[dataset]]

                for label in labels:
                    subset = df[df["gold_label_text"] == label]
                    if len(subset) == 0:
                        continue

                    y_pred = subset["sc_majority_vote"].tolist()
                    valid  = [p for p in y_pred if p not in ("invalid", "nan")]
                    if not valid:
                        continue

                    recall          = sum(p == label for p in valid) / len(valid)
                    precision_denom = df[df["sc_majority_vote"] == label]
                    precision       = (
                        sum(precision_denom["gold_label_text"] == label) /
                        len(precision_denom)
                        if len(precision_denom) > 0 else 0.0
                    )
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0
                    )

                    rows.append({
                        "model":       model,
                        "dataset":     dataset,
                        "prompt_type": prompt_type,
                        "label":       label,
                        "precision":   round(float(precision), 4),
                        "recall":      round(float(recall), 4),
                        "f1":          round(float(f1), 4),
                        "n_gold":      len(subset),
                        "n_predicted": len(precision_denom),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "per_class_accuracy.csv")
    return df_out


def compute_format_brittleness():
    print("Computing format brittleness rate (SC)...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                for sc_col in ["sc_sample_1", "sc_sample_2", "sc_sample_3"]:
                    if sc_col not in df.columns:
                        continue

                    invalid_rate = (
                        df[sc_col].astype(str).str.strip().str.lower() == "invalid"
                    ).mean()

                    rows.append({
                        "model":        model,
                        "dataset":      dataset,
                        "prompt_type":  prompt_type,
                        "sc_sample":    sc_col,
                        "invalid_rate": round(float(invalid_rate), 4),
                        "n_invalid":    int((df[sc_col].astype(str).str.strip().str.lower() == "invalid").sum()),
                        "n_total":      len(df),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "format_brittleness.csv")
    return df_out


def compute_label_bias_index():
    print("Computing label bias index...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                labels   = [str(l) for l in DATASET_LABELS[dataset]]
                n_labels = len(labels)
                uniform  = 1.0 / n_labels

                valid = df[df["sc_majority_vote"].isin(labels)]
                pred_counts = valid["sc_majority_vote"].value_counts(normalize=True)

                kl        = 0.0
                pred_dist = {}
                for label in labels:
                    p = float(pred_counts.get(label, 0.0))
                    pred_dist[label] = round(p, 4)
                    if p > 0:
                        kl += p * np.log(p / uniform)

                row = {
                    "model":       model,
                    "dataset":     dataset,
                    "prompt_type": prompt_type,
                    "lbi":         round(float(kl), 4),
                    "n_total":     len(df),
                    "n_valid":     len(valid),
                }
                for label in labels:
                    row[f"pred_rate_{label.replace(' ', '_')}"] = pred_dist.get(label, 0.0)

                rows.append(row)

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "label_bias_index.csv")
    return df_out


def compute_ece():
    print("Computing Expected Calibration Error (ECE)...")
    rows = []
    active_pts = get_active_prompt_types()
    N_BINS = 10

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                df = df.dropna(subset=["logprob_margin"])

                confidences = df["logprob_margin"].values
                correctness = df["sc_correct"].values

                bins = np.linspace(0, 1, N_BINS + 1)
                ece  = 0.0
                n    = len(df)

                for i in range(N_BINS):
                    mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
                    if mask.sum() == 0:
                        continue
                    bin_conf = confidences[mask].mean()
                    bin_acc  = correctness[mask].mean()
                    bin_n    = mask.sum()
                    ece     += (bin_n / n) * abs(bin_acc - bin_conf)

                rows.append({
                    "model":       model,
                    "dataset":     dataset,
                    "prompt_type": prompt_type,
                    "ece":         round(float(ece), 4),
                    "n":           n,
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "ece.csv")
    return df_out


def compute_sc_logprob_consistency():
    print("Computing SC-Logprob consistency...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                if "sc_majority_vote" not in df.columns:
                    continue

                logprob_pred = df["logprob_predicted_label"].astype(str)
                sc_pred      = df["sc_majority_vote"].astype(str)

                valid_mask    = sc_pred != "invalid"
                if valid_mask.sum() == 0:
                    continue

                agreement     = (logprob_pred[valid_mask] == sc_pred[valid_mask]).mean()
                disagree_mask = valid_mask & (logprob_pred != sc_pred)
                agree_mask    = valid_mask & (logprob_pred == sc_pred)

                acc_when_agree    = df.loc[agree_mask,    "sc_correct"].mean() if agree_mask.sum() > 0 else None
                acc_when_disagree = df.loc[disagree_mask, "sc_correct"].mean() if disagree_mask.sum() > 0 else None

                rows.append({
                    "model":             model,
                    "dataset":           dataset,
                    "prompt_type":       prompt_type,
                    "agreement_rate":    round(float(agreement), 4),
                    "n_agree":           int(agree_mask.sum()),
                    "n_disagree":        int(disagree_mask.sum()),
                    "n_sc_invalid":      int((~valid_mask).sum()),
                    "acc_when_agree":    round(float(acc_when_agree), 4) if acc_when_agree is not None else None,
                    "acc_when_disagree": round(float(acc_when_disagree), 4) if acc_when_disagree is not None else None,
                    "acc_gap":           round(float(acc_when_agree - acc_when_disagree), 4)
                                         if acc_when_agree is not None and acc_when_disagree is not None else None,
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "sc_logprob_consistency.csv")
    return df_out


def compute_sc_agreement_accuracy():
    print("Computing SC agreement → accuracy...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                if "sc_agreement" not in df.columns:
                    continue

                for agreement_level in [1, 2, 3]:
                    subset = df[df["sc_agreement"] == agreement_level]
                    if len(subset) == 0:
                        continue

                    acc      = subset["sc_correct"].mean()
                    coverage = len(subset) / len(df)

                    rows.append({
                        "model":           model,
                        "dataset":         dataset,
                        "prompt_type":     prompt_type,
                        "agreement_level": agreement_level,
                        "accuracy":        round(float(acc), 4),
                        "coverage":        round(float(coverage), 4),
                        "n":               len(subset),
                        "n_total":         len(df),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "sc_agreement_accuracy.csv")
    return df_out


def compute_accuracy_coverage():
    print("Computing accuracy-coverage tradeoff...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                if "sc_agreement" not in df.columns:
                    continue

                n_total = len(df)

                for threshold, label in [(1, "all"), (2, "agreement_gte2"), (3, "unanimous")]:
                    subset = df if threshold == 1 else df[df["sc_agreement"] >= threshold]

                    if len(subset) == 0:
                        continue

                    acc      = subset["sc_correct"].mean()
                    coverage = len(subset) / n_total

                    rows.append({
                        "model":       model,
                        "dataset":     dataset,
                        "prompt_type": prompt_type,
                        "threshold":   label,
                        "accuracy":    round(float(acc), 4),
                        "coverage":    round(float(coverage), 4),
                        "n":           len(subset),
                        "n_total":     n_total,
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "accuracy_coverage.csv")
    return df_out


def compute_inference_time():
    print("Computing inference time statistics...")
    rows = []
    active_pts = get_active_prompt_types()

    for model in tqdm(INSTRUCT_MODELS):
        for dataset in DATASETS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                for col, label in [
                    ("logprob_inference_time", "logprob"),
                    ("sc_inference_time",      "sc"),
                ]:
                    if col not in df.columns:
                        continue
                    times = df[col].dropna()
                    if len(times) == 0:
                        continue

                    rows.append({
                        "model":      model,
                        "dataset":    dataset,
                        "prompt_type": prompt_type,
                        "pass_type":  label,
                        "mean_sec":   round(float(times.mean()), 4),
                        "median_sec": round(float(times.median()), 4),
                        "total_sec":  round(float(times.sum()), 2),
                        "total_min":  round(float(times.sum()) / 60, 2),
                        "n":          len(times),
                    })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "inference_time.csv")
    return df_out


def compute_signal_comparison():
    print("Computing logprob vs SC signal comparison...")
    rows = []
    active_pts = get_active_prompt_types()

    for dataset in tqdm(DATASETS):
        for model in INSTRUCT_MODELS:
            for prompt_type in active_pts:
                df = load_predictions(model, dataset, prompt_type)
                if df is None or len(df) == 0:
                    continue

                df = df.dropna(subset=["sc_agreement", "sc_correct",
                                       "logprob_margin", "logprob_correct"])
                if len(df) < 50:
                    continue

                sc_corr      = df["sc_agreement"].corr(df["sc_correct"])
                logprob_corr = df["logprob_margin"].corr(df["logprob_correct"])

                rows.append({
                    "model":        model,
                    "dataset":      dataset,
                    "prompt_type":  prompt_type,
                    "sc_corr":      round(sc_corr, 4),
                    "logprob_corr": round(logprob_corr, 4),
                    "sc_wins":      int(sc_corr > logprob_corr),
                    "n":            len(df),
                })

    df_out = pd.DataFrame(rows)
    save_csv(df_out, "signal_comparison.csv")

    print(f"\n  SC wins:      {df_out['sc_wins'].sum()} / {len(df_out)}")
    print(f"  Mean SC corr:      {df_out['sc_corr'].mean():.4f}")
    print(f"  Mean logprob corr: {df_out['logprob_corr'].mean():.4f}")
    print(f"\n  Per dataset:")
    print(df_out.groupby("dataset")[["sc_corr","logprob_corr"]].mean().round(4).to_string())

    return df_out


def debug_boolq_logprob():
    print("Debugging BoolQ logprob extraction...")
    
    for model in INSTRUCT_MODELS[:3]:
        df = load_predictions(model, "BoolQ", "zero_shot")
        if df is None:
            print(f"  {model}: load_predictions returned None")
            continue

        print(f"\n  {model}:")
        print(f"    Total rows:            {len(df)}")
        print(f"    logprob_correct NaN:   {df['logprob_correct'].isna().sum()}")
        print(f"    logprob_correct zeros: {(df['logprob_correct'] == 0).sum()}")
        print(f"    logprob_correct ones:  {(df['logprob_correct'] == 1).sum()}")
        print(f"    logprob_margin NaN:    {df['logprob_margin'].isna().sum()}")
        print(f"    logprob_margin zeros:  {(df['logprob_margin'] == 0).sum()}")
        print(f"    sc_correct ones:       {(df['sc_correct'] == 1).sum()}")
        print(f"    logprob_positive NaN:  {df['logprob_positive'].isna().sum()}")
        print(f"    logprob_negative NaN:  {df['logprob_negative'].isna().sum()}")
        print(f"\n    Sample logprob_positive values:")
        print(f"    {df['logprob_positive'].dropna().head(5).tolist()}")
        print(f"\n    Sample logprob_margin values:")
        print(f"    {df['logprob_margin'].dropna().head(5).tolist()}")
        print(f"\n    gold_label distribution:")
        print(f"    {df['gold_label'].value_counts().to_dict()}")
        print(f"\n    logprob_predicted_label distribution:")
        print(f"    {df['logprob_predicted_label'].value_counts().to_dict()}")



def main():
    print(f"\nEvaluating prompt mode: {EVAL_PROMPT_MODE}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    debug_boolq_logprob()
    compute_signal_comparison()
    compute_confidence_correctness_sc()
    compute_accuracy_f1()
    compute_bootstrap_ci()
    compute_mcnemar()
    compute_dampening_ratio()
    compute_safety_hallucination()
    compute_entailment_bias()
    compute_entropy_accuracy()
    compute_confidence_correctness()
    compute_cross_task_correlation()
    compute_annotator_simulation()
    compute_per_class_accuracy()
    compute_format_brittleness()
    compute_label_bias_index()
    compute_ece()
    compute_sc_logprob_consistency()
    compute_sc_agreement_accuracy()
    compute_accuracy_coverage()
    compute_inference_time()

    print("\nAll evaluation metrics computed.")


if __name__ == "__main__":
    main()