# Plain-language explainer (Gemma + LoRA, runs locally or online)

Turns formal government notices and medical instructions into simple Hindi, Tamil or
Assamese. One small Gemma model, fine-tuned with a LoRA adapter, shrunk to ~0.8 GB so it
runs on a laptop or phone-class CPU, and served by the same model online when you want.

## The pipeline

```
seed_examples.jsonl ──prepare_data.py──▶ train.jsonl / val.jsonl
                                              │
                                         train.py  (QLoRA)  ──▶ outputs/adapter  (small LoRA file)
                                              │
                                      export_model.py  (merge + GGUF 4-bit)
                                              │
                                   outputs/model-Q4_K_M.gguf  (~0.8 GB)
                                     │                        │
                      explain.py --mode local      llama-server (online) ◀── explain.py --mode server
                                     └──── explain.py --mode auto: local first, server if checks fail
```

## Run it

```bash
pip install -r requirements.txt

# 0. sanity check (no GPU needed)
python -m unittest discover -s tests

# 1. build training files
python prepare_data.py

# 2. fine-tune (needs a CUDA GPU; accept the Gemma license on Hugging Face and run `huggingface-cli login` first)
python train.py

# 3. merge + quantize (needs a built llama.cpp checkout)
python export_model.py --llama-cpp ~/llama.cpp

# 4. use it locally
python explain.py --language hindi --domain medical --text "इस औषधि को भोजनोपरांत दिन में दो बार ... ग्रहण करें।"

# 5. online mode: serve the same file, then call it
~/llama.cpp/build/bin/llama-server -m outputs/model-Q4_K_M.gguf --port 8080 -c 2048
python explain.py --mode server --language tamil --domain govt --file notice.txt

# 6. evaluate on held-out examples
python evaluate.py
```

## What each file does

| File | Role |
|---|---|
| `config.py` | Languages, domains, base model, and the **one prompt template** shared by every step |
| `checks.py` | Automatic safety checks: numbers kept/not invented, right script, sane length |
| `prepare_data.py` | Cleans text (Unicode NFC), dedupes, rejects bad pairs using `checks.py`, splits train/val |
| `train.py` | QLoRA fine-tuning; loss only on the answer tokens |
| `export_model.py` | Merges adapter into the base model, converts to 4-bit GGUF |
| `explain.py` | Inference with local / server / auto routing and the safety gate |
| `evaluate.py` | Runs the validation set, reports pass rate and chrF, writes a review file |

## Important limits

- `data/seed_examples.jsonl` has only 6 illustrative pairs, to prove the pipeline works.
  They were not reviewed by a native-speaker editor. **Replace them with 1,000-10,000
  reviewed pairs** (real notices and leaflets rewritten by people, or machine-drafted
  rewrites checked by native speakers) before judging quality.
- The safety checks catch changed or invented numbers and wrong scripts. They do **not**
  catch a wrong meaning. For medical and legal text, always show the original beside
  the simple version and have a human expert review a sample.
- Assamese has no seed data. Add Assamese pairs to the same file with `"language": "assamese"`.

## Extending to the other use cases

- **New field:** change `domain`, gather pairs for it, and train a new adapter. The base GGUF stays the same.
- **Agriculture / legal:** add a retrieval step that fetches trusted passages and puts them in the prompt.
- **Tanglish / Hinglish:** add Roman-script and mixed-script pairs to the training file.
