import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset as HFDataset
from peft import PeftModel

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
    labels = [ex["output"].split(":")[1].strip().replace(".", "") for ex in lines]
    return HFDataset.from_dict({"prompt": prompts, "label": labels})

def extract_level(output: str) -> str:
    for level in triage_levels:
        if level.lower() in output.lower():
            return level
    return "Unknown"

def benchmark(model, tokenizer, dataset):
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
        repetition_penalty=2.0
    )

    correct = 0
    total = 0
    answered = 0
    prompts = dataset["prompt"]
    golds = dataset["label"]
    all_outputs = pipe(prompts, batch_size=BATCH_SIZE)
    for output_group, gold in zip(all_outputs, golds):
        print(f"{output_group=}")
        prediction = extract_level(output_group[0]['generated_text'])
        if prediction != "Unknown":
            answered += 1
            if prediction == gold:
                correct += 1
        total += 1
    accuracy = correct / total if total else 0
    answer_rate = answered / total if total else 0
    return accuracy, answer_rate

if __name__ == "__main__":
    test_dataset = load_and_prepare_data(DATA_PATH)

    base_tokenizer = AutoTokenizer.from_pretrained(FT_MODEL)
    base_tokenizer.padding_side = 'left'
    base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(FT_MODEL, device_map="auto")

    ft_model = PeftModel.from_pretrained(base_model, FT_MODEL)


    base_acc, base_ans = benchmark(base_model, base_tokenizer, test_dataset)
    ft_acc, ft_ans = benchmark(ft_model, base_tokenizer, test_dataset)

    print("\n--- Benchmark Results ---")
    print(f"Base Model Accuracy:      {base_acc:.2%}")
    print(f"Base Model Answer Rate:   {base_ans:.2%}")
    print(f"Fine-Tuned Model Accuracy:{ft_acc:.2%}")
    print(f"Fine-Tuned Answer Rate:   {ft_ans:.2%}")