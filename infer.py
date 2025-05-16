import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fused_rms_norm import RMSNorm, rms_norm_backend
from linear import Linear, bmm_backend
from scaled_dot_product_attention import (
    Attention,
    rope_backend,
    scaled_dot_product_attention_backend,
)
from silu import SiLU, silu_backend
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
    parser.add_argument(
        "--backend",
        type=str,
        default="ninetoothed",
        help='Backend to use for inference (e.g., "ninetoothed", "triton", "torch").',
    )
    parser.add_argument(
        "--num-warmup-iterations",
        type=int,
        default=0,
        help="For profiling. The number of warmup iterations to run before measuring performance.",
    )
    parser.add_argument(
        "--num-profiling-iterations",
        type=int,
        default=1,
        help="For profiling. The number of iterations to run for performance measurement.",
    )

    args = parser.parse_args()

    model_name_or_path = args.model
    prompts = args.prompts
    max_new_tokens = args.max_new_tokens
    device = args.device
    backend = args.backend
    num_warmup_iterations = args.num_warmup_iterations
    num_profiling_iterations = args.num_profiling_iterations

    assert num_profiling_iterations >= 1
    assert num_warmup_iterations >= 0

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if backend != "torch":
        replace_module(model, Attention)
        replace_module(model, Linear)
        replace_module(model, RMSNorm)
        replace_module(model, SiLU)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with (
        bmm_backend(backend),
        rms_norm_backend(backend),
        rope_backend(backend),
        scaled_dot_product_attention_backend(backend),
        silu_backend(backend),
    ):
        for _ in range(num_warmup_iterations):
            model.generate(**inputs, max_new_tokens=max_new_tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(num_profiling_iterations):
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

    strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    average_time = (end_time - start_time) / num_profiling_iterations
    num_input_tokens = inputs["input_ids"].size(-1)
    num_output_tokens = outputs.size(-1) - num_input_tokens
    num_tokens_per_second = num_output_tokens / average_time

    print(
        json.dumps(
            {
                "strings": strings,
                "average_time": average_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "num_tokens_per_second": num_tokens_per_second,
            }
        )
    )
