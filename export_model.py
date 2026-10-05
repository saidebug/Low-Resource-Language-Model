"""Step 3: merge the adapter into the base model, then shrink it to a 4-bit GGUF file.

Merging and conversion need the llama.cpp repo (for convert_hf_to_gguf.py and llama-quantize).
Pass --llama-cpp /path/to/llama.cpp to run the conversion automatically.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=str(OUT_DIR / "adapter"))
    ap.add_argument("--llama-cpp", default=None, help="path to a built llama.cpp checkout")
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()

    merged = OUT_DIR / "merged"
    # Reload the base model in plain bf16 (not 4-bit) so the merge is accurate.
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(merged)
    print(f"merged model saved to {merged}")

    f16 = OUT_DIR / "model-f16.gguf"
    quant_file = OUT_DIR / f"model-{args.quant}.gguf"
    if not args.llama_cpp:
        print("\nTo finish, run these from your llama.cpp folder:")
        print(f"  python convert_hf_to_gguf.py {merged} --outfile {f16} --outtype f16")
        print(f"  ./build/bin/llama-quantize {f16} {quant_file} {args.quant}")
        return

    llama = Path(args.llama_cpp)
    subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)
    subprocess.run([str(llama / "build" / "bin" / "llama-quantize"), str(f16), str(quant_file), args.quant],
                   check=True)
    print(f"ready: {quant_file}")


if __name__ == "__main__":
    main()
