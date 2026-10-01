# this will be the script that runs your classifier against the golden dataset.

import json
import asyncio
import time
import matplotlib.pyplot as plt
import yaml
# import openai
import requests
from pathlib import Path
from datetime import datetime
from src.classifier import classify_email
from src.models import PromptConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.report_generator import generate_html_report
import os

USE_MOCK = os.getenv("MOCK_MODE", "false").lower() == "true"


def send_slack_alert(score, thresholds, run_id):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL") # replace with your Slack webhook
    message = {
        "text": (
            f"📊 Eval Run {run_id}\n"
            f"Accuracy: {score['accuracy']*100:.2f}% "
            f"({score['correct']}/{score['total']})\n"
            f"Avg relevance: {score['avg_relevance']:.2f}\n"
            f"Avg latency: {score['avg_latency']:.2f}s\n"
            f"Avg tokens: {score['avg_tokens']:.2f}\n"
            f"Threshold status: {thresholds['status']} "
            f"(delta={thresholds['delta']*100:.2f}%)"
        )
    }
    try:
        requests.post(webhook_url, json=message)
        print("Slack alert sent.")
    except Exception as e:
        print(f"Slack alert failed: {e}")

async def judge_summary(expected, actual, model="gpt-4o-mini"):
    prompt = f"""
    You are evaluating a customer support summary.
    Expected summary: "{expected}"
    Actual summary: "{actual}"
    Rate relevance on a scale of 1 (poor) to 5 (excellent).
    Only return the number.
    """

    if USE_MOCK:
        # Return a fake score for testing without API calls
        print("MOCK_MODE enabled: returning dummy relevance score")
        return 5  # you can randomize or vary this if you want
    else:
        import openai
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        score = int(response.choices[0].message.content.strip())
        return score


def load_prompt_config(path: str) -> PromptConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return PromptConfig(**data)

def load_dataset(path: str):
    with open(path, "r") as f:
        return json.load(f)["cases"]

def classify_email_mock(input_text, prompt_config):
    return type("MockResponse", (), {
        "model_dump": lambda self: {
            "category": "MOCK_CATEGORY",
            "summary": "MOCK_SUMMARY"
        },
        "usage": {"total_tokens": 0}
    })()

async def run_case(case, prompt_config):
    start = time.time()
    if USE_MOCK:
        response = classify_email_mock(case["input"], prompt_config.model_dump())
    else:
        response = classify_email(case["input"], prompt_config.model_dump())
    latency = time.time() - start

    result = response.model_dump()

    relevance_score = await judge_summary(
        case["expected_output"]["summary"],
        result["summary"]
    )

    tokens_used = getattr(response, "usage", {}).get("total_tokens", None)

    return {
        "id": case["id"],
        "expected": case["expected_output"],
        "actual": result,
        "latency": latency,
        "relevance_score": relevance_score,
        "tokens_used": tokens_used
    }

async def run_all_cases(dataset, prompt_config):
    tasks = [run_case(case, prompt_config) for case in dataset]
    return await asyncio.gather(*tasks)

def compare_runs(old_results, new_results):
    diffs = []
    for old, new in zip(old_results, new_results):
        case_id = old["id"]
        old_cat = old["actual"]["category"]
        new_cat = new["actual"]["category"]
        old_sum = old["actual"]["summary"]
        new_sum = new["actual"]["summary"]

        diffs.append({
            "id": case_id,
            "category_changed": old_cat != new_cat,
            "summary_changed": old_sum != new_sum,
            "old": old["actual"],
            "new": new["actual"]
        })
    return diffs

def calculate_accuracy(results):
    total = len(results)
    correct = 0
    relevance_scores = []
    latencies = []
    tokens = []

    for r in results:
        if r["expected"]["category"] == r["actual"]["category"] and \
           r["expected"]["summary"] == r["actual"]["summary"]:
            correct += 1
        if r.get("relevance_score") is not None:
            relevance_scores.append(r["relevance_score"])
        latencies.append(r["latency"])
        if r.get("tokens_used") is not None:
            tokens.append(r["tokens_used"])

    accuracy = correct / total if total > 0 else 0
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_relevance": avg_relevance,
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens
    }

def check_thresholds(old_acc, new_acc):
    delta = new_acc - old_acc
    if delta <= -0.08:
        status = "CRITICAL regression"
    elif delta <= -0.03:
        status = "WARNING regression"
    else:
        status = "PASS"
    return {"delta": delta, "status": status}


def report_trends():
    runs = sorted(Path("results").glob("run_*.json"))
    history = []
    for run_file in runs:
        with open(run_file, "r") as f:
            data = json.load(f)
            # Only include runs with metadata
            if isinstance(data, dict) and "run_id" in data and "score" in data:
                history.append({
                    "run_id": data["run_id"],
                    "accuracy": data["score"]["accuracy"]
                })
    print("\nAccuracy trend across runs:")
    for h in history:
        print(f"{h['run_id']}: {h['accuracy']*100:.2f}%")

def plot_accuracy_trend():
    runs = sorted(Path("results").glob("run_*.json"))
    history = []
    for run_file in runs:
        with open(run_file, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "run_id" in data and "score" in data:
                history.append((data["run_id"], data["score"]["accuracy"]))

    if not history:
        print("No valid runs to plot.")
        return

    # Extract run IDs and accuracies
    run_ids = [h[0] for h in history]
    accuracies = [h[1] * 100 for h in history]

    # Plot line chart
    plt.figure(figsize=(8, 4))
    plt.plot(run_ids, accuracies, marker="o", linestyle="-", color="blue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Accuracy Trend Across Runs")
    plt.xlabel("Run ID")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    # Save chart
    chart_path = "results/accuracy_trend.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Accuracy trend chart saved to {chart_path}")

def plot_confusion_matrix(results):
    # Extract expected and actual categories
    y_true = [r["expected"]["category"] for r in results]
    y_pred = [r["actual"]["category"] for r in results]

    # Compute confusion matrix
    labels = sorted(set(y_true) | set(y_pred))  # union of all categories
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)

    chart_path = "results/confusion_matrix.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Confusion matrix chart saved to {chart_path}")

def plot_dashboard(results):
    # --- Accuracy trend ---
    runs = sorted(Path("results").glob("run_*.json"))
    history = []
    for run_file in runs:
        with open(run_file, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "run_id" in data and "score" in data:
                history.append((data["run_id"], data["score"]["accuracy"]))

    # --- Confusion matrix ---
    y_true = [r["expected"]["category"] for r in results]
    y_pred = [r["actual"]["category"] for r in results]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Create dashboard figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy trend subplot
    if history:
        run_ids = [h[0] for h in history]
        accuracies = [h[1] * 100 for h in history]
        axes[0].plot(run_ids, accuracies, marker="o", linestyle="-", color="blue")
        axes[0].set_title("Accuracy Trend Across Runs")
        axes[0].set_xlabel("Run ID")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].tick_params(axis="x", rotation=45)

    # Confusion matrix subplot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    chart_path = "results/dashboard.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Dashboard chart saved to {chart_path}")

if __name__ == "__main__":
    prompt_config = load_prompt_config("prompts/v1.0.yaml")
    dataset = load_dataset("dataset/golden_dataset_v1.json")
    results = asyncio.run(run_all_cases(dataset, prompt_config))

    # Accuracy scoring
    score = calculate_accuracy(results)

    # Save results with timestamp + score
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/run_{run_id}.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "score": score,
            "cases": results
        }, f, indent=2)

    print(f"Saved results to {output_path}")

    # Print case results
    for r in results:
        print(f"Case {r['id']}: expected={r['expected']} actual={r['actual']} latency={r['latency']:.2f}s")

    # Compare with last run if available
    diffs = None  # Initialize diffs to None so its always defined
    previous_runs = sorted(Path("results").glob("run_*.json"))
    if len(previous_runs) > 1:
        with open(previous_runs[-2], "r") as f:
            old_data = json.load(f)
        old_results = old_data["cases"]
        diffs = compare_runs(old_results, results)
        print("\nComparison with previous run:")
        for d in diffs:
            if d["category_changed"] or d["summary_changed"]:
                print(f"Case {d['id']} changed: {d['old']} -> {d['new']}")

     # Generate HTML report
    generate_html_report(run_id, score, results, diffs)

if len(previous_runs) > 1:
    old_score = old_data["score"]
    thresholds = check_thresholds(old_score["accuracy"], score["accuracy"])
    print(f"\nThreshold check: {thresholds['status']} (delta={thresholds['delta']*100:.2f}%)")

    send_slack_alert(score, thresholds, run_id)

    # Print summary
    print("\nRun summary:")
    print(f"Total cases: {score['total']}")
    print(f"Correct: {score['correct']}")
    print(f"Accuracy: {score['accuracy']*100:.2f}%")

    # Show accuracy trend across all runs
    report_trends()

    # Generate accuracy trend chart
    plot_accuracy_trend()

    # Generate confusion matrix chart
    plot_confusion_matrix(results)

    # Generate combined dashboard chart
    plot_dashboard(results)



