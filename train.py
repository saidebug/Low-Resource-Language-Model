"""Step 2: fine-tune Gemma with QLoRA (4-bit base + small trainable LoRA adapter)."""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import BASE_MODEL, DATA_DIR, OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=True,
                    help="use --no-load-4bit if you have a 16GB+ GPU and no bitsandbytes")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)

    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager")  # eager attention is recommended for Gemma 3 training
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"))["train"]
    val_ds = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"))["train"]
    has_val = len(val_ds) > 0

    cfg = SFTConfig(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_length,
        report_to="none",
        # The dataset is in prompt/completion format, so loss is computed only on the
        # answer tokens (the model learns to write answers, not to repeat prompts).
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        processing_class=tok, peft_config=lora)
    trainer.train()

    adapter_dir = OUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
