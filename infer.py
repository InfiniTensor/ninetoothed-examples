import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from rms_norm import RMSNorm
from utils import replace_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a causal language model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or model identifier from Hugging Face.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompts for text generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use for inference (e.g., "cuda", "cpu").',
    )

    args = parser.parse_args()

    model_name_or_path = args.model
    prompts = args.prompts
    max_new_tokens = args.max_new_tokens
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    replace_module(model, RMSNorm)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(strings)
