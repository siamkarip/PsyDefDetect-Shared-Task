import pandas as pd
import os

FEWSHOT_DIR = "fewshot_examples"

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


def format_fewshot_block(examples, format_input_fn, label_col):
    blocks = []
    for _, row in examples.iterrows():
        blocks.append(f"{format_input_fn(row)}\nLabel: {row[label_col]}")
    return "\n\n".join(blocks)



def sst2_input(row):
    return f"Text: {row['sentence']}"

def sst2_zero_shot(row):
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Respond with exactly one label from: [positive, negative]. "
        "Do not include any explanation or additional text.\n\n"
        f"Text: {row['sentence']}\n"
        "Label:"
    )

def sst2_few_shot(row):
    examples = load_fewshot("SST2")
    fewshot_block = format_fewshot_block(examples, sst2_input, FEWSHOT_LABEL_COL["SST2"])
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Respond with exactly one label from: [positive, negative]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Text: {row['sentence']}\n"
        "Label:"
    )

def sst2_cot(row):
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Think step-by-step: "
        "Step 1: Identify sentiment-bearing words and phrases. "
        "Step 2: Assess the overall tone. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: [positive, negative] "
        "on a new line prefixed with 'Label:'.\n\n"
        f"Text: {row['sentence']}\n"
    )


def sst5_input(row):
    return f"Text: {row['sentence']}"

def sst5_zero_shot(row):
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Respond with exactly one label from: "
        "[very negative, negative, neutral, positive, very positive]. "
        "Do not include any explanation or additional text.\n\n"
        f"Text: {row['sentence']}\n"
        "Label:"
    )

def sst5_few_shot(row):
    examples = load_fewshot("SST5")
    fewshot_block = format_fewshot_block(examples, sst5_input, FEWSHOT_LABEL_COL["SST5"])
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Respond with exactly one label from: "
        "[very negative, negative, neutral, positive, very positive]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Text: {row['sentence']}\n"
        "Label:"
    )

def sst5_cot(row):
    return (
        "You are an expert sentiment analysis system. "
        "Classify the sentiment of the following text. "
        "Think step-by-step: "
        "Step 1: Identify sentiment-bearing words and phrases. "
        "Step 2: Assess the overall tone. "
        "Step 3: Determine the intensity of the sentiment. "
        "Step 4: State your final label. "
        "Your response must end with exactly one label from: "
        "[very negative, negative, neutral, positive, very positive] "
        "on a new line prefixed with 'Label:'.\n\n"
        f"Text: {row['sentence']}\n"
    )



def cola_input(row):
    return f"Sentence: {row['sentence']}"

def cola_zero_shot(row):
    return (
        "You are an expert linguist specializing in grammatical analysis. "
        "Classify whether the following sentence is grammatically acceptable. "
        "Respond with exactly one label from: [acceptable, unacceptable]. "
        "Do not include any explanation or additional text.\n\n"
        f"Sentence: {row['sentence']}\n"
        "Label:"
    )

def cola_few_shot(row):
    examples = load_fewshot("CoLA")
    fewshot_block = format_fewshot_block(examples, cola_input, FEWSHOT_LABEL_COL["CoLA"])
    return (
        "You are an expert linguist specializing in grammatical analysis. "
        "Classify whether the following sentence is grammatically acceptable. "
        "Respond with exactly one label from: [acceptable, unacceptable]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Sentence: {row['sentence']}\n"
        "Label:"
    )

def cola_cot(row):
    return (
        "You are an expert linguist specializing in grammatical analysis. "
        "Classify whether the following sentence is grammatically acceptable. "
        "Think step-by-step: "
        "Step 1: Identify the grammatical structure of the sentence. "
        "Step 2: Check for syntactic violations or anomalies. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[acceptable, unacceptable] on a new line prefixed with 'Label:'.\n\n"
        f"Sentence: {row['sentence']}\n"
    )



def nli_input(row):
    return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"

def snli_zero_shot(row):
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Respond with exactly one label from: [entailment, neutral, contradiction]. "
        "Do not include any explanation or additional text.\n\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Label:"
    )

def snli_few_shot(row):
    examples = load_fewshot("SNLI")
    fewshot_block = format_fewshot_block(examples, nli_input, FEWSHOT_LABEL_COL["SNLI"])
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Respond with exactly one label from: [entailment, neutral, contradiction]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Label:"
    )

def snli_cot(row):
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Think step-by-step: "
        "Step 1: Identify the key entities and events in both sentences. "
        "Step 2: Determine the logical relationship between premise and hypothesis. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[entailment, neutral, contradiction] on a new line prefixed with 'Label:'.\n\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
    )



def mnli_zero_shot(row):
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Respond with exactly one label from: [entailment, neutral, contradiction]. "
        "Do not include any explanation or additional text.\n\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Label:"
    )

def mnli_few_shot(row):
    examples = load_fewshot("MNLI")
    fewshot_block = format_fewshot_block(examples, nli_input, FEWSHOT_LABEL_COL["MNLI"])
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Respond with exactly one label from: [entailment, neutral, contradiction]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Label:"
    )

def mnli_cot(row):
    return (
        "You are an expert in natural language inference. "
        "Given a premise and a hypothesis, classify their relationship. "
        "Think step-by-step: "
        "Step 1: Identify the key entities and events in both sentences. "
        "Step 2: Determine the logical relationship between premise and hypothesis. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[entailment, neutral, contradiction] on a new line prefixed with 'Label:'.\n\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
    )



def mrpc_input(row):
    return f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"

def mrpc_zero_shot(row):
    return (
        "You are an expert in semantic similarity and paraphrase detection. "
        "Determine whether the two sentences are paraphrases of each other. "
        "Respond with exactly one label from: [paraphrase, not paraphrase]. "
        "Do not include any explanation or additional text.\n\n"
        f"Sentence 1: {row['sentence1']}\n"
        f"Sentence 2: {row['sentence2']}\n"
        "Label:"
    )

def mrpc_few_shot(row):
    examples = load_fewshot("MRPC")
    fewshot_block = format_fewshot_block(examples, mrpc_input, FEWSHOT_LABEL_COL["MRPC"])
    return (
        "You are an expert in semantic similarity and paraphrase detection. "
        "Determine whether the two sentences are paraphrases of each other. "
        "Respond with exactly one label from: [paraphrase, not paraphrase]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Sentence 1: {row['sentence1']}\n"
        f"Sentence 2: {row['sentence2']}\n"
        "Label:"
    )

def mrpc_cot(row):
    return (
        "You are an expert in semantic similarity and paraphrase detection. "
        "Determine whether the two sentences are paraphrases of each other. "
        "Think step-by-step: "
        "Step 1: Identify the core meaning of each sentence. "
        "Step 2: Compare the semantic content for equivalence. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[paraphrase, not paraphrase] on a new line prefixed with 'Label:'.\n\n"
        f"Sentence 1: {row['sentence1']}\n"
        f"Sentence 2: {row['sentence2']}\n"
    )



def swag_input(row):
    return (
        f"Beginning: {row['startphrase']}\n"
        f"A: {row['ending0']}\n"
        f"B: {row['ending1']}\n"
        f"C: {row['ending2']}\n"
        f"D: {row['ending3']}"
    )

def swag_zero_shot(row):
    return (
        "You are an expert in commonsense reasoning. "
        "Given the beginning of a sentence, select the most plausible continuation. "
        "Respond with exactly one label from: [A, B, C, D]. "
        "Do not include any explanation or additional text.\n\n"
        f"Beginning: {row['startphrase']}\n"
        f"A: {row['ending0']}\n"
        f"B: {row['ending1']}\n"
        f"C: {row['ending2']}\n"
        f"D: {row['ending3']}\n"
        "Label:"
    )

def swag_few_shot(row):
    examples = load_fewshot("SWAG")
    fewshot_block = format_fewshot_block(examples, swag_input, FEWSHOT_LABEL_COL["SWAG"])
    return (
        "You are an expert in commonsense reasoning. "
        "Given the beginning of a sentence, select the most plausible continuation. "
        "Respond with exactly one label from: [A, B, C, D]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Beginning: {row['startphrase']}\n"
        f"A: {row['ending0']}\n"
        f"B: {row['ending1']}\n"
        f"C: {row['ending2']}\n"
        f"D: {row['ending3']}\n"
        "Label:"
    )

def swag_cot(row):
    return (
        "You are an expert in commonsense reasoning. "
        "Given the beginning of a sentence, select the most plausible continuation. "
        "Think step-by-step: "
        "Step 1: Understand the beginning of the sentence. "
        "Step 2: Evaluate each option for plausibility. "
        "Step 3: Select the most plausible continuation and state your final label. "
        "Your response must end with exactly one label from: [A, B, C, D] "
        "on a new line prefixed with 'Label:'.\n\n"
        f"Beginning: {row['startphrase']}\n"
        f"A: {row['ending0']}\n"
        f"B: {row['ending1']}\n"
        f"C: {row['ending2']}\n"
        f"D: {row['ending3']}\n"
    )



def hatexplain_input(row):
    return f"Text: {row['text']}"

def hatexplain_zero_shot(row):
    return (
        "You are an expert content moderation system. "
        "Classify the following text for harmful content. "
        "Respond with exactly one label from: [normal, offensive, hatespeech]. "
        "Do not include any explanation or additional text.\n\n"
        f"Text: {row['text']}\n"
        "Label:"
    )

def hatexplain_few_shot(row):
    examples = load_fewshot("HateXplain")
    fewshot_block = format_fewshot_block(examples, hatexplain_input, FEWSHOT_LABEL_COL["HateXplain"])
    return (
        "You are an expert content moderation system. "
        "Classify the following text for harmful content. "
        "Respond with exactly one label from: [normal, offensive, hatespeech]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Text: {row['text']}\n"
        "Label:"
    )

def hatexplain_cot(row):
    return (
        "You are an expert content moderation system. "
        "Classify the following text for harmful content. "
        "Think step-by-step: "
        "Step 1: Identify potentially harmful or offensive language. "
        "Step 2: Assess the severity and intent of the language. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[normal, offensive, hatespeech] on a new line prefixed with 'Label:'.\n\n"
        f"Text: {row['text']}\n"
    )



def boolq_input(row):
    return f"Passage: {row['passage']}\nQuestion: {row['question']}"

def boolq_zero_shot(row):
    return (
        "You are an expert reading comprehension system. "
        "Based on the passage, answer the question. "
        "Respond with exactly one label from: [true, false]. "
        "Do not include any explanation or additional text.\n\n"
        f"Passage: {row['passage']}\n"
        f"Question: {row['question']}\n"
        "Label:"
    )

def boolq_few_shot(row):
    examples = load_fewshot("BoolQ")
    fewshot_block = format_fewshot_block(examples, boolq_input, FEWSHOT_LABEL_COL["BoolQ"])
    return (
        "You are an expert reading comprehension system. "
        "Based on the passage, answer the question. "
        "Respond with exactly one label from: [true, false]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Passage: {row['passage']}\n"
        f"Question: {row['question']}\n"
        "Label:"
    )

def boolq_cot(row):
    return (
        "You are an expert reading comprehension system. "
        "Based on the passage, answer the question. "
        "Think step-by-step: "
        "Step 1: Read the passage and identify information relevant to the question. "
        "Step 2: Evaluate whether the passage supports or contradicts the question. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[true, false] on a new line prefixed with 'Label:'.\n\n"
        f"Passage: {row['passage']}\n"
        f"Question: {row['question']}\n"
    )



def trec_input(row):
    return f"Question: {row['text']}"

def trec_zero_shot(row):
    return (
        "You are an expert question classification system. "
        "Classify the type of answer the following question is seeking. "
        "Respond with exactly one label from: "
        "[abbreviation, entity, description, human, location, numeric]. "
        "Do not include any explanation or additional text.\n\n"
        f"Question: {row['text']}\n"
        "Label:"
    )

def trec_few_shot(row):
    examples = load_fewshot("TREC")
    fewshot_block = format_fewshot_block(examples, trec_input, FEWSHOT_LABEL_COL["TREC"])
    return (
        "You are an expert question classification system. "
        "Classify the type of answer the following question is seeking. "
        "Respond with exactly one label from: "
        "[abbreviation, entity, description, human, location, numeric]. "
        "Do not include any explanation or additional text.\n\n"
        "Here are some examples:\n\n"
        f"{fewshot_block}\n\n"
        "Now classify the following:\n"
        f"Question: {row['text']}\n"
        "Label:"
    )

def trec_cot(row):
    return (
        "You are an expert question classification system. "
        "Classify the type of answer the following question is seeking. "
        "Think step-by-step: "
        "Step 1: Identify what the question is asking for. "
        "Step 2: Match the expected answer type to the closest category. "
        "Step 3: State your final label. "
        "Your response must end with exactly one label from: "
        "[abbreviation, entity, description, human, location, numeric] "
        "on a new line prefixed with 'Label:'.\n\n"
        f"Question: {row['text']}\n"
    )



PROMPT_REGISTRY = {
    "SST2":       {"zero_shot": sst2_zero_shot,       "few_shot": sst2_few_shot,       "cot": sst2_cot},
    "SST5":       {"zero_shot": sst5_zero_shot,       "few_shot": sst5_few_shot,       "cot": sst5_cot},
    "CoLA":       {"zero_shot": cola_zero_shot,       "few_shot": cola_few_shot,       "cot": cola_cot},
    "SNLI":       {"zero_shot": snli_zero_shot,       "few_shot": snli_few_shot,       "cot": snli_cot},
    "MNLI":       {"zero_shot": mnli_zero_shot,       "few_shot": mnli_few_shot,       "cot": mnli_cot},
    "MRPC":       {"zero_shot": mrpc_zero_shot,       "few_shot": mrpc_few_shot,       "cot": mrpc_cot},
    "SWAG":       {"zero_shot": swag_zero_shot,       "few_shot": swag_few_shot,       "cot": swag_cot},
    "HateXplain": {"zero_shot": hatexplain_zero_shot, "few_shot": hatexplain_few_shot, "cot": hatexplain_cot},
    "BoolQ":      {"zero_shot": boolq_zero_shot,      "few_shot": boolq_few_shot,      "cot": boolq_cot},
    "TREC":       {"zero_shot": trec_zero_shot,       "few_shot": trec_few_shot,       "cot": trec_cot},
}


def get_prompt(dataset, prompt_type, row):
    return PROMPT_REGISTRY[dataset][prompt_type](row)



if __name__ == "__main__":
    DATASETS_DIR = "datasets"
    SEP = "=" * 80

    for dataset in PROMPT_REGISTRY:
        df = pd.read_csv(os.path.join(DATASETS_DIR, f"{dataset}.csv"))
        row = df.iloc[0]

        print(f"\n{SEP}")
        print(f"DATASET: {dataset}")
        print(SEP)

        for prompt_type in ["zero_shot", "few_shot", "cot"]:
            print(f"\n--- {prompt_type.upper()} ---\n")
            print(get_prompt(dataset, prompt_type, row))
            print()