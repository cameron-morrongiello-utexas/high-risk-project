import fire
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig
import torch


class TriageFineTuner:
    def __init__(
        self,
        model_name="gpt2",
        dataset_path="synthetic_triage_cases_with_secondary_decisions.jsonl",
        output_dir="./output",
        epochs=3,
        batch_size=4,
        max_length=256,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 has no pad token

    def load_data(self):
        dataset = load_dataset("json", data_files=self.dataset_path)

        # Split into train and test
        dataset = dataset["train"].train_test_split(test_size=0.1)
        return dataset

    def preprocess(self, dataset):
        def format(example):
            prompt = example["input"].strip()
            completion = example["output"].strip()
            return {"text": f"{prompt}\n{completion}"}

        dataset = dataset.map(format)

        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        return dataset.map(tokenize, batched=True)

    def fine_tune(self):
        dataset = self.load_data()
        tokenized_dataset = self.preprocess(dataset)

        model = GPT2LMHeadModel.from_pretrained(self.model_name)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Model saved to {self.output_dir}")

    def generate_response(self, input_text, max_length=150):
        model = GPT2LMHeadModel.from_pretrained(self.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(self.output_dir)

        model.eval()
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def main(
    model_name="gpt2",
    dataset_path="synthetic_triage_cases_with_secondary_decisions.jsonl",
    output_dir="./output",
    epochs=3,
    batch_size=4,
    input_text=None,
):
    tuner = TriageFineTuner(model_name, dataset_path, output_dir, epochs, batch_size)

    if input_text:
        response = tuner.generate_response(input_text)
        print("\n🤖 Response:\n", response)
    else:
        tuner.fine_tune()


if __name__ == "__main__":
    fire.Fire(main)
