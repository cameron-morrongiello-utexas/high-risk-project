import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset as HFDataset

triage_levels = ["Emergency Room", "Urgent Care", "Primary Care", "Self-Care"]
DATA_PATH = Path("data/test.jsonl")
TRIAGE_SNIPPET = f"Possible triage levels: {', '.join(triage_levels)}.\n"
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
FT_MODEL = "model/rft_model"
MAX_NEW_TOKENS = 200
BATCH_SIZE = 4

def load_and_prepare_data(path):
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    prompts = [f"{TRIAGE_SNIPPET}{ex['input']}" for ex in lines]
    labels = [ex["output"].split(":")[1].strip() for ex in lines]
    return HFDataset.from_dict({"prompt": prompts, "label": labels})

def extract_level(output: str) -> str:
    for level in triage_levels:
        if level.lower() in output.lower():
            return level
    return "Unknown"

def benchmark(model_id, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False
    )
    correct = 0
    total = 0
    answered = 0
    prompts = dataset["prompt"]
    golds = dataset["label"]
    all_outputs = pipe(prompts, batch_size=BATCH_SIZE)
    for outputs, gold in zip(all_outputs, golds):
        prediction = extract_level(outputs[0]["generated_text"])
        if prediction != "Unknown":
            answered += 1
            if prediction == gold:
                correct += 1
        total += 1
    accuracy = correct / total if total else 0
    answer_rate = answered / total if total else 0
    return accuracy, answer_rate

if __name__ == "__main__":
    dataset = load_and_prepare_data(DATA_PATH)
    base_acc, base_ans = benchmark(BASE_MODEL, dataset)
    ft_acc, ft_ans = benchmark(FT_MODEL, dataset)
    print("\n--- Benchmark Results ---")
    print(f"Base Model Accuracy:      {base_acc:.2%}")
    print(f"Base Model Answer Rate:   {base_ans:.2%}")
    print(f"Fine-Tuned Model Accuracy:{ft_acc:.2%}")
    print(f"Fine-Tuned Answer Rate:   {ft_ans:.2%}")