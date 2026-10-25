from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dspy
from typesafe_sdk import Choice, TypeSafeClient


# Example-local policy for approximately normalized, rounded SDK probabilities.
# Keep accepted values unchanged; this is not an SDK-guaranteed error bound.
PROBABILITY_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class JevProgram:
    instructions: str
    criteria: dict[str, str | None]


@dataclass
class PredictionRecord:
    expected: str
    predicted: str
    probabilities: dict[str, float]
    state: dict[str, Any]


class RefineJevPrompt(dspy.Signature):
    """Improve one reusable TypeSafe Jev Choice program from benchmark failures.

    Preserve the task meaning and label keys. Make the judgment narrow and explicit.
    Put the judgment itself in instructions and label meanings in criteria. Do not
    add facts that are absent from the task specification or examples.
    """

    task_spec: str = dspy.InputField()
    current_program_json: str = dspy.InputField()
    metrics_json: str = dspy.InputField()
    failure_cases_json: str = dspy.InputField()
    jev_instructions: str = dspy.OutputField(desc="Replacement Jev Choice instructions")
    criteria_json: str = dspy.OutputField(desc="JSON object mapping the SAME label keys to improved definitions")
    hypothesis: str = dspy.OutputField(desc="Short explanation of what error pattern this candidate targets")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_dataset(rows: list[dict[str, Any]], seed: int = 42) -> tuple[list, list, list]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = max(1, int(n * 0.60))
    n_val = max(1, int(n * 0.20))
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def _typed_choice(response: Any, question_id: str) -> Any:
    # Current TypeSafe SDKs expose typed views; keep this tolerant of both common views.
    if hasattr(response, "choices"):
        return response.choices[question_id]
    return response.answers[question_id]


def _validated_probabilities(answer: Any, labels: list[str]) -> dict[str, float]:
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError("Choice probabilities must contain exactly the configured label keys")
    if answer.choice not in labels:
        raise ValueError("Choice must be one of the configured labels")
    for value in probabilities.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Choice probabilities must be finite numbers between 0 and 1")
    # The small numerical allowance only avoids rejecting the decimal boundary
    # due to binary floating-point representation; never fill or normalize values.
    if not math.isclose(
        math.fsum(probabilities.values()), 1.0, rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE + 1e-12,
    ):
        raise ValueError("Choice probabilities must sum to approximately 1 (absolute tolerance 0.01)")
    return dict(probabilities)


def evaluate(
    client: TypeSafeClient,
    program: JevProgram,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[PredictionRecord]]:
    records: list[PredictionRecord] = []
    labels = list(program.criteria)
    if not rows or not labels:
        raise ValueError("Evaluation requires examples and configured labels")
    if any(row.get("expected") not in labels for row in rows):
        raise ValueError("Expected labels must be among the configured labels")

    for row in rows:
        response = client.system_one(
            state=row["state"],
            questions={
                "label": Choice(
                    instructions=program.instructions,
                    criteria=program.criteria,
                )
            },
        )
        answer = _typed_choice(response, "label")
        probabilities = _validated_probabilities(answer, labels)
        records.append(
            PredictionRecord(
                expected=row["expected"],
                predicted=answer.choice,
                probabilities=probabilities,
                state=row["state"],
            )
        )

    accuracy = sum(r.expected == r.predicted for r in records) / max(1, len(records))

    # Macro-F1 without sklearn.
    f1s: list[float] = []
    for label in labels:
        tp = sum(r.expected == label and r.predicted == label for r in records)
        fp = sum(r.expected != label and r.predicted == label for r in records)
        fn = sum(r.expected == label and r.predicted != label for r in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    macro_f1 = sum(f1s) / max(1, len(f1s))

    # Multiclass Brier score. Lower is better; max is 2 for one-hot targets.
    brier = 0.0
    for r in records:
        brier += sum(
            (r.probabilities[label] - (1.0 if label == r.expected else 0.0)) ** 2
            for label in labels
        )
    brier /= max(1, len(records))

    # Accuracy is the primary objective; Brier only breaks close ties.
    fitness = accuracy + 0.001 * (1.0 - min(1.0, brier / 2.0))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "brier": brier, "fitness": fitness}, records


def summarize_failures(
    records: list[PredictionRecord],
    max_cases: int = 12,
) -> list[dict[str, Any]]:
    failures = [r for r in records if r.expected != r.predicted][:max_cases]
    return [
        {
            "state": r.state,
            "expected": r.expected,
            "predicted": r.predicted,
            "probabilities": r.probabilities,
        }
        for r in failures
    ]


def validate_candidate(current: JevProgram, candidate: JevProgram) -> None:
    if set(current.criteria) != set(candidate.criteria):
        raise ValueError("DSPy changed label keys; candidate rejected")
    if not candidate.instructions.strip():
        raise ValueError("Empty instructions")


def optimize(
    dataset: Path,
    iterations: int,
    seed: int,
    task_spec: str,
    initial: JevProgram,
    output_dir: Path,
) -> None:
    rows = load_jsonl(dataset)
    train, validation, test = split_dataset(rows, seed)
    if not validation:
        raise ValueError("Need enough examples to create a validation split")

    lm = dspy.LM(os.environ.get("DSPY_LM", "openai/gpt-5.4-mini"))
    dspy.configure(lm=lm)
    refiner = dspy.Predict(RefineJevPrompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    with TypeSafeClient() as client:
        best = initial
        best_val, _ = evaluate(client, best, validation)

        for iteration in range(iterations):
            train_metrics, train_records = evaluate(client, best, train)
            failures = summarize_failures(train_records)

            proposal = refiner(
                task_spec=task_spec,
                current_program_json=json.dumps(asdict(best), indent=2),
                metrics_json=json.dumps(train_metrics, indent=2),
                failure_cases_json=json.dumps(failures, indent=2),
            )

            candidate: JevProgram | None = None
            candidate_val: dict[str, float] | None = None
            try:
                criteria = json.loads(proposal.criteria_json)
                candidate = JevProgram(
                    instructions=proposal.jev_instructions.strip(),
                    criteria=criteria,
                )
                validate_candidate(best, candidate)
                candidate_val, _ = evaluate(client, candidate, validation)
                accepted = (
                    candidate_val["accuracy"] > best_val["accuracy"]
                    or (
                        candidate_val["accuracy"] == best_val["accuracy"]
                        and candidate_val["brier"] < best_val["brier"]
                    )
                )
            except Exception as exc:
                accepted = False
                proposal.hypothesis = f"Rejected candidate: {exc}"

            history.append(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "hypothesis": proposal.hypothesis,
                    "train_metrics": train_metrics,
                    "validation_metrics": candidate_val,
                    "candidate": asdict(candidate) if candidate is not None else None,
                }
            )

            if accepted:
                assert candidate is not None and candidate_val is not None
                best = candidate
                best_val = candidate_val

            print(
                f"iter={iteration:02d} accepted={accepted} "
                f"val_accuracy={best_val['accuracy']:.4f} val_brier={best_val['brier']:.4f}"
            )

        # The test set is touched once, after prompt search is finished.
        test_metrics, _ = evaluate(client, best, test) if test else ({}, [])

    (output_dir / "best_program.json").write_text(
        json.dumps(asdict(best), indent=2),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps({"validation": best_val, "hidden_test": test_metrics}, indent=2),
        encoding="utf-8",
    )

    print("\nBest program:")
    print(json.dumps(asdict(best), indent=2))
    print("\nFinal metrics:")
    print(json.dumps({"validation": best_val, "hidden_test": test_metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="DSPy outer-loop prompt optimization for TypeSafe Jev")
    parser.add_argument("--dataset", type=Path, default=Path("sample_dataset.jsonl"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    args = parser.parse_args()

    initial = JevProgram(
        instructions="Choose the support department that should own this request based on `ticket`.",
        criteria={
            "billing": "Charges, invoices, refunds, payments, or subscription billing.",
            "technical": "Product bugs, errors, integration failures, setup, or troubleshooting.",
            "sales": "Pricing before purchase, plans, procurement, demos, or buying questions.",
            "other": "None of the other departments clearly owns the request.",
        },
    )

    task_spec = (
        "Route each support ticket to exactly one owner: billing, technical, sales, or other. "
        "Optimize held-out classification accuracy. Do not change label keys. Prefer explicit "
        "evidence in the ticket over inferred intent."
    )

    optimize(args.dataset, args.iterations, args.seed, task_spec, initial, args.output_dir)


if __name__ == "__main__":
    main()
