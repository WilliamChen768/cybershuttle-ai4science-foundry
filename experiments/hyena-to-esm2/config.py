import os

# Same pinned model and revision as registry/models/biology/hyenadna-small-32k (PR #34).
MODEL = "LongSafari/hyenadna-small-32k-seqlen-hf"
REVISION = "8fe770c78eb13fe33bf81501612faeddf4d6f331"
OUT_DIR = os.environ.get("OUT_DIR", "outputs")