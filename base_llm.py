from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def generate(self, prompt: str) -> str:
        """
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """
        

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.
        """
        from tqdm import tqdm

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]
        num_return_sequences = num_return_sequences or 1
        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        generations = self.model.generate(
            **tokenized_prompts,
            max_new_tokens=200,
            do_sample=temperature > 0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        detokenized = self.tokenizer.batch_decode(generations[:, len(tokenized_prompts["input_ids"][0]) :], skip_special_tokens=True)
        if num_return_sequences == 1:
            return detokenized
        else:
            return [detokenized[i:i + num_return_sequences] for i in range(0, len(detokenized), num_return_sequences)]

def test_model(input: str):
    model = BaseLLM()
    answers = model.generate(input)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
