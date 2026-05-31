import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)
from deepeval.models import GPTModel
import glob


# LM Studio Endpoint
model = GPTModel(
    model="google/gemma-4-e4b",
    base_url="http://localhost:12345/v1",   # /chat/completions is automatically added by deepeval
    api_key="lm-studio"  # ダミーでOK
)

# -----------------------------
# metrics
# -----------------------------
metrics = [
    FaithfulnessMetric(model=model),
    AnswerRelevancyMetric(model=model),
    ContextualRelevancyMetric(model=model)
]

# -----------------------------
# load all logs
# -----------------------------
log_files = glob.glob("logs/*.json")

test_cases = []
for file_path in log_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_case = LLMTestCase(
        input=data["question"],
        retrieval_context=[
            doc["context"] for doc in data["search_results"]
        ],
        context=[data["context"]],
        actual_output=data["answer"],
        expected_output=data.get("expected_answer", "")
    )

    test_cases.append(test_case)

# -----------------------------
# run evaluation
# -----------------------------
evaluate(test_cases, metrics)
