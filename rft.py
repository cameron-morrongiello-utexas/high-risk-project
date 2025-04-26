from .base_llm import BaseLLM
from .data import Dataset


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(input: str, output: str) -> dict[str, str]:
    """
    Construct a question / answer pair
    """
    formatted = {
        "question": input,
        "answer": output
    }
    print(f"{formatted=}")
    return formatted


def train_model(
    output_dir: str,
    **kwargs,
):
    from transformers import TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig, TaskType
    from .base_llm import BaseLLM
    from .data import Dataset
    from .tok import TokenizedDataset
    from .rft import format_example

    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer

    r = 16
    lora_config = LoraConfig(
        r=r,
        lora_alpha=r * 4,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    model.enable_input_require_grads()

    trainset = Dataset("train")
    tokenized_dataset = TokenizedDataset(tokenizer, trainset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=7e-4,
        logging_steps=10,
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    # benchmark_result = benchmark(llm, testset, 100)
    # print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

    llm.batched_generate([testset[0]['input']])


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
