import os
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


PREDICTIONS_DIR = "predictions_base_instruct"
OUTPUT_DIR      = os.path.join("eval_results", "base_instruct")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PAIRS = [
    {"base": "Llama-3.1-8B",    "instruct": "Llama-3.1-8B-Instruct"},
    {"base": "gemma-2-9b",      "instruct": "gemma-2-9b-it"},
    {"base": "Qwen3.5-9B-Base", "instruct": "Qwen3.5-9B"},
]

DATASETS = ["SST2", "SST5", "CoLA", "SNLI", "MNLI",
            "MRPC", "SWAG", "HateXplain", "BoolQ", "TREC"]

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

N_BOOTSTRAP = 10000
ALPHA       = 0.05



def normalize_labels(df, dataset):
    if dataset == "BoolQ":
        for col in ["gold_label_text", "gold_label"]:
            if col in df.columns:
                df[col] = df[col].map(
                    {True: "true", False: "false",
                     "True": "true", "False": "false"}
                ).fillna(df[col].astype(str).str.lower())
    for col in ["gold_label_text", "gold_label", "logprob_predicted_label"]:
        if col in df.columns:
            df[col] = df[col].fillna("invalid").astype(str).str.strip()
            if dataset != "SWAG":
                df[col] = df[col].str.lower()
    return df


def load_predictions(model, dataset):
    path = os.path.join(PREDICTIONS_DIR, model, f"{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = normalize_labels(df, dataset)
    df["logprob_correct"] = (
        df["logprob_predicted_label"] == df["gold_label_text"]
    ).astype(int)
    return df


def load_pair(pair, dataset):
    base_df     = load_predictions(pair["base"],     dataset)
    instruct_df = load_predictions(pair["instruct"], dataset)

    if base_df is None or instruct_df is None:
        return None, None

    base_df     = base_df.set_index("row_id")
    instruct_df = instruct_df.set_index("row_id")
    common_idx  = base_df.index.intersection(instruct_df.index)

    if len(common_idx) == 0:
        return None, None

    return base_df.loc[common_idx], instruct_df.loc[common_idx]



def bootstrap_proportion(values, n=N_BOOTSTRAP, alpha=ALPHA):
    values = np.array(values)
    scores = [np.mean(np.random.choice(values, size=len(values), replace=True))
              for _ in range(n)]
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)



def mcnemar_test(correct_a, correct_b):
    b = np.sum((np.array(correct_a) == 1) & (np.array(correct_b) == 0))
    c = np.sum((np.array(correct_a) == 0) & (np.array(correct_b) == 1))
    if b + c == 0:
        return 1.0
    chi2  = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - chi2_dist.cdf(chi2, df=1))



def compute_accuracy():
    print("Computing accuracy comparison...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            labels = [str(l) for l in DATASET_LABELS[dataset]]

            for model_type, df in [("base", base_df), ("instruct", instruct_df)]:
                y_true = df["gold_label_text"].astype(str).tolist()
                y_pred = df["logprob_predicted_label"].astype(str).tolist()

                valid  = [(t, p) for t, p in zip(y_true, y_pred)
                          if p not in ("invalid", "nan")]
                if not valid:
                    continue

                y_t, y_p = zip(*valid)
                acc        = accuracy_score(y_t, y_p)
                f1_macro   = f1_score(y_t, y_p, average="macro",
                                      labels=labels, zero_division=0)
                f1_weighted = f1_score(y_t, y_p, average="weighted",
                                       labels=labels, zero_division=0)

                rows.append({
                    "pair":         pair_name,
                    "base_model":   pair["base"],
                    "instruct_model": pair["instruct"],
                    "dataset":      dataset,
                    "model_type":   model_type,
                    "model":        pair[model_type],
                    "accuracy":     round(acc, 4),
                    "f1_macro":     round(f1_macro, 4),
                    "f1_weighted":  round(f1_weighted, 4),
                    "n_valid":      len(valid),
                    "n_total":      len(df),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "accuracy.csv"), index=False)
    print(f"  Saved accuracy.csv — {len(df_out)} rows")
    return df_out



def compute_asi():
    print("Computing Alignment Sensitivity Index (ASI)...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            def get_acc(df):
                y_true = df["gold_label_text"].astype(str).tolist()
                y_pred = df["logprob_predicted_label"].astype(str).tolist()
                valid  = [(t, p) for t, p in zip(y_true, y_pred)
                          if p not in ("invalid", "nan")]
                if not valid:
                    return None
                y_t, y_p = zip(*valid)
                return accuracy_score(y_t, y_p)

            acc_base     = get_acc(base_df)
            acc_instruct = get_acc(instruct_df)

            if acc_base is None or acc_instruct is None or acc_base == 0:
                continue

            asi = (acc_instruct - acc_base) / acc_base

            rows.append({
                "pair":           pair_name,
                "base_model":     pair["base"],
                "instruct_model": pair["instruct"],
                "dataset":        dataset,
                "acc_base":       round(acc_base, 4),
                "acc_instruct":   round(acc_instruct, 4),
                "acc_delta":      round(acc_instruct - acc_base, 4),
                "asi":            round(asi, 4),
                "alignment_effect": "helps" if asi > 0 else "hurts" if asi < 0 else "neutral",
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "asi.csv"), index=False)
    print(f"  Saved asi.csv — {len(df_out)} rows")
    return df_out



def compute_lps():
    print("Computing Label Probability Shift (LPS)...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            labels = DATASET_LABELS[dataset]

            for gold_label in labels:
                base_sub     = base_df[base_df["gold_label_text"] == gold_label]
                instruct_sub = instruct_df[instruct_df["gold_label_text"] == gold_label]

                if len(base_sub) == 0 or len(instruct_sub) == 0:
                    continue

                for label in labels:
                    col = f"logprob_{label.replace(' ', '_')}"
                    if col not in base_df.columns or col not in instruct_df.columns:
                        continue

                    mean_base     = base_sub[col].dropna().mean()
                    mean_instruct = instruct_sub[col].dropna().mean()
                    lps           = mean_instruct - mean_base

                    rows.append({
                        "pair":           pair_name,
                        "base_model":     pair["base"],
                        "instruct_model": pair["instruct"],
                        "dataset":        dataset,
                        "gold_label":     gold_label,
                        "label":          label,
                        "mean_prob_base":     round(float(mean_base), 4),
                        "mean_prob_instruct": round(float(mean_instruct), 4),
                        "lps":            round(float(lps), 4),
                        "direction":      "amplified" if lps > 0 else "suppressed",
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "lps.csv"), index=False)
    print(f"  Saved lps.csv — {len(df_out)} rows")
    return df_out



def compute_failure_attribution():
    print("Computing failure mode attribution...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            base_correct     = base_df["logprob_correct"].values
            instruct_correct = instruct_df["logprob_correct"].values
            n = len(base_correct)

            both_correct          = ((base_correct == 1) & (instruct_correct == 1))
            alignment_induced     = ((base_correct == 1) & (instruct_correct == 0)) 
            alignment_corrected   = ((base_correct == 0) & (instruct_correct == 1)) 
            pretraining_artifact  = ((base_correct == 0) & (instruct_correct == 0))   

            r_both       = both_correct.mean()
            r_induced    = alignment_induced.mean()
            r_corrected  = alignment_corrected.mean()
            r_artifact   = pretraining_artifact.mean()

            induced_lower, induced_upper = bootstrap_proportion(
                alignment_induced.astype(int)
            )
            corrected_lower, corrected_upper = bootstrap_proportion(
                alignment_corrected.astype(int)
            )
            artifact_lower, artifact_upper = bootstrap_proportion(
                pretraining_artifact.astype(int)
            )

            rows.append({
                "pair":                       pair_name,
                "base_model":                 pair["base"],
                "instruct_model":             pair["instruct"],
                "dataset":                    dataset,
                "n":                          n,
                "both_correct":               round(r_both, 4),
                "alignment_induced":          round(r_induced, 4),
                "alignment_induced_ci_lower": round(induced_lower, 4),
                "alignment_induced_ci_upper": round(induced_upper, 4),
                "alignment_corrected":        round(r_corrected, 4),
                "alignment_corrected_ci_lower": round(corrected_lower, 4),
                "alignment_corrected_ci_upper": round(corrected_upper, 4),
                "pretraining_artifact":       round(r_artifact, 4),
                "pretraining_artifact_ci_lower": round(artifact_lower, 4),
                "pretraining_artifact_ci_upper": round(artifact_upper, 4),
                "net_alignment_effect":       round(r_corrected - r_induced, 4),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "failure_attribution.csv"), index=False)
    print(f"  Saved failure_attribution.csv — {len(df_out)} rows")
    return df_out



def compute_mcnemar():
    print("Computing McNemar's tests (base vs instruct)...")
    rows      = []
    p_values  = []
    pair_meta = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            correct_base     = base_df["logprob_correct"].tolist()
            correct_instruct = instruct_df["logprob_correct"].tolist()

            p_val = mcnemar_test(correct_base, correct_instruct)
            p_values.append(p_val)
            pair_meta.append({
                "pair":           pair_name,
                "base_model":     pair["base"],
                "instruct_model": pair["instruct"],
                "dataset":        dataset,
                "p_value":        round(p_val, 6),
                "n":              len(correct_base),
            })

    n_tests           = len(p_values)
    bonferroni_alpha  = ALPHA / n_tests

    for i, meta in enumerate(pair_meta):
        meta["p_value_corrected"] = round(p_values[i] * n_tests, 6)
        meta["significant"]       = p_values[i] < bonferroni_alpha
        rows.append(meta)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "mcnemar.csv"), index=False)
    print(f"  Saved mcnemar.csv — {len(df_out)} rows")
    return df_out



def compute_margin_comparison():
    print("Computing margin comparison (base vs instruct)...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            for model_type, df in [("base", base_df), ("instruct", instruct_df)]:
                if "logprob_margin" not in df.columns:
                    continue

                margins   = df["logprob_margin"].dropna()
                correct   = df.loc[margins.index, "logprob_correct"]

                margin_correct   = margins[correct == 1]
                margin_incorrect = margins[correct == 0]

                rows.append({
                    "pair":                   pair_name,
                    "base_model":             pair["base"],
                    "instruct_model":         pair["instruct"],
                    "dataset":                dataset,
                    "model_type":             model_type,
                    "model":                  pair[model_type],
                    "mean_margin":            round(float(margins.mean()), 4),
                    "mean_margin_correct":    round(float(margin_correct.mean()), 4) if len(margin_correct) > 0 else None,
                    "mean_margin_incorrect":  round(float(margin_incorrect.mean()), 4) if len(margin_incorrect) > 0 else None,
                    "overconfidence_gap":     round(float(margin_incorrect.mean() - margin_correct.mean()), 4)
                                             if len(margin_correct) > 0 and len(margin_incorrect) > 0 else None,
                    "n":                      len(margins),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "margin_comparison.csv"), index=False)
    print(f"  Saved margin_comparison.csv — {len(df_out)} rows")
    return df_out



def compute_kl_divergence():
    print("Computing KL divergence (base vs instruct)...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            labels  = DATASET_LABELS[dataset]
            prob_cols = [f"logprob_{l.replace(' ', '_')}" for l in labels]

            if not all(c in base_df.columns and c in instruct_df.columns
                       for c in prob_cols):
                continue

            base_probs     = base_df[prob_cols].values
            instruct_probs = instruct_df[prob_cols].values

            eps = 1e-10
            kl_vals = np.sum(
                instruct_probs * np.log((instruct_probs + eps) / (base_probs + eps)),
                axis=1
            )

            rows.append({
                "pair":              pair_name,
                "base_model":        pair["base"],
                "instruct_model":    pair["instruct"],
                "dataset":           dataset,
                "mean_kl":           round(float(kl_vals.mean()), 4),
                "median_kl":         round(float(np.median(kl_vals)), 4),
                "max_kl":            round(float(kl_vals.max()), 4),
                "kl_correct":        round(float(kl_vals[base_df["logprob_correct"].values == 1].mean()), 4)
                                     if (base_df["logprob_correct"].values == 1).sum() > 0 else None,
                "kl_incorrect":      round(float(kl_vals[base_df["logprob_correct"].values == 0].mean()), 4)
                                     if (base_df["logprob_correct"].values == 0).sum() > 0 else None,
                "n":                 len(kl_vals),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "kl_divergence.csv"), index=False)
    print(f"  Saved kl_divergence.csv — {len(df_out)} rows")
    return df_out



def compute_per_class_accuracy():
    print("Computing per-class accuracy (base vs instruct)...")
    rows = []

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            labels = [str(l) for l in DATASET_LABELS[dataset]]

            for model_type, df in [("base", base_df), ("instruct", instruct_df)]:
                for label in labels:
                    subset = df[df["gold_label_text"] == label]
                    if len(subset) == 0:
                        continue

                    y_pred = subset["logprob_predicted_label"].tolist()
                    valid  = [p for p in y_pred if p not in ("invalid", "nan")]
                    if not valid:
                        continue

                    recall     = sum(p == label for p in valid) / len(valid)
                    pred_as    = df[df["logprob_predicted_label"] == label]
                    precision  = (
                        sum(pred_as["gold_label_text"] == label) / len(pred_as)
                        if len(pred_as) > 0 else 0.0
                    )
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0
                    )

                    rows.append({
                        "pair":           pair_name,
                        "base_model":     pair["base"],
                        "instruct_model": pair["instruct"],
                        "dataset":        dataset,
                        "model_type":     model_type,
                        "model":          pair[model_type],
                        "label":          label,
                        "precision":      round(float(precision), 4),
                        "recall":         round(float(recall), 4),
                        "f1":             round(float(f1), 4),
                        "n_gold":         len(subset),
                        "n_predicted":    len(pred_as),
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "per_class_accuracy.csv"), index=False)
    print(f"  Saved per_class_accuracy.csv — {len(df_out)} rows")
    return df_out



def compute_ece():
    print("Computing ECE (base vs instruct)...")
    rows  = []
    N_BINS = 10

    for pair in tqdm(MODEL_PAIRS):
        pair_name = f"{pair['base']} vs {pair['instruct']}"

        for dataset in DATASETS:
            base_df, instruct_df = load_pair(pair, dataset)
            if base_df is None:
                continue

            for model_type, df in [("base", base_df), ("instruct", instruct_df)]:
                if "logprob_margin" not in df.columns:
                    continue

                df = df.dropna(subset=["logprob_margin"])
                if len(df) == 0:
                    continue

                confidences = df["logprob_margin"].values
                correctness = df["logprob_correct"].values
                bins        = np.linspace(0, 1, N_BINS + 1)
                ece         = 0.0
                n           = len(df)

                for i in range(N_BINS):
                    mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
                    if mask.sum() == 0:
                        continue
                    bin_conf = confidences[mask].mean()
                    bin_acc  = correctness[mask].mean()
                    ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)

                rows.append({
                    "pair":           pair_name,
                    "base_model":     pair["base"],
                    "instruct_model": pair["instruct"],
                    "dataset":        dataset,
                    "model_type":     model_type,
                    "model":          pair[model_type],
                    "ece":            round(float(ece), 4),
                    "n":              n,
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "ece.csv"), index=False)
    print(f"  Saved ece.csv — {len(df_out)} rows")
    return df_out



def main():
    print(f"\nBase vs Instruct Evaluation")
    print(f"Output dir: {OUTPUT_DIR}\n")

    compute_accuracy()
    compute_asi()
    compute_lps()
    compute_failure_attribution()
    compute_mcnemar()
    compute_margin_comparison()
    compute_kl_divergence()
    compute_per_class_accuracy()
    compute_ece()

    print("\nAll base vs instruct metrics computed.")


if __name__ == "__main__":
    main()