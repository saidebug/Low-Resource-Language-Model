import glob, os, shutil
from huggingface_hub import snapshot_download

p = snapshot_download(
    "google/gemma-3-1b-it",
    allow_patterns=["tokenizer*", "special_tokens_map.json",
                    "added_tokens.json", "chat_template*"],
)
for f in glob.glob(os.path.join(p, "*")):
    if os.path.isfile(f):
        shutil.copy(f, "outputs/merged")
        print("copied", os.path.basename(f))
