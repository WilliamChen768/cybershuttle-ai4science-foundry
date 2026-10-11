import json
import time

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LongSafari/hyenadna-small-32k-seqlen-hf"
SEQ = "ACTGACTGACTGACTG"

# find current commit and load it
revision = HfApi().model_info(MODEL).sha

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL)
print("Model revision (commit SHA):", revision)

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=revision, trust_remote_code=True
).to("cuda").eval()

inputs = {k: v.to("cuda") for k, v in tokenizer(SEQ, return_tensors="pt").items()}

start = time.time()
with torch.no_grad():
    out = model(**inputs)
torch.cuda.synchronize()
elapsed = time.time() - start

print("Input sequence:", SEQ)
print("Token IDs:", inputs["input_ids"].tolist())
print("Logits shape:", tuple(out.logits.shape))
print("Inference time (s):", round(elapsed, 4))

# write output to file
torch.save(
    {"input_ids": inputs["input_ids"].cpu(), "logits": out.logits.cpu()},
    "outputs/smoke_test_logits.pt",
)
with open("outputs/smoke_test_meta.json", "w") as f:
    json.dump(
        {
            "model": MODEL,
            "revision": revision,
            "sequence": SEQ,
            "logits_shape": list(out.logits.shape),
            "inference_seconds": elapsed,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        f,
        indent=2,
    )
print("Saved outputs/smoke_test_logits.pt and outputs/smoke_test_meta.json")