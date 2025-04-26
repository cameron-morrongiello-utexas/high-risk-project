import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

triage_levels = ["Emergency Room", "Urgent Care", "Primary Care", "Self-Care"]
DATA_PATH = Path("data/valid.jsonl")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
FT_MODEL = "model/rft_model"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 200

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract_level(output: str) -> str:
    for level in triage_levels:
        if level.lower() in output.lower():
            return level
    return "Unknown"

def benchmark(model_id, data):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    for ex in tqdm(data, desc=f"Benchmarking {model_id}"):
        prompt = ex['input']
        result = pipe(prompt)[0]['generated_text']
        prediction = extract_level(result)
        gold = ex["output"].split(":")[1].strip()
        if prediction == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total else 0
    return accuracy


if __name__ == "__main__":
    data = load_jsonl(DATA_PATH)

    base_acc = benchmark(BASE_MODEL, data)
    ft_acc = benchmark(FT_MODEL, data)

    print("\n--- Benchmark Results ---")
    print(f"Base Model Accuracy:     {base_acc:.2%}")
    print(f"Fine-Tuned Model Accuracy: {ft_acc:.2%}")
