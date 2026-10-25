"""Offline regressions; install this example's dependencies before running."""

from contextlib import ExitStack, redirect_stdout
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# DSPy imports LiteLLM. Use its bundled pricing data and forbid import-time I/O.
with tempfile.TemporaryDirectory() as cache, ExitStack() as imports:
    imports.enter_context(patch.dict(os.environ, {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DSPY_CACHEDIR": cache,
    }))
    import_connect = imports.enter_context(
        patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
    )
    import_resolve = imports.enter_context(
        patch("socket.getaddrinfo", side_effect=AssertionError("Network forbidden"))
    )
    spec = importlib.util.spec_from_file_location(
        "jev_optimizer_example", Path(__file__).resolve().parents[1] / "optimizer.py"
    )
    optimizer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = optimizer
    spec.loader.exec_module(optimizer)
    import httpx2
    from typesafe_sdk import ChoiceAnswer, SystemOneResponse
    from typesafe_sdk import TypeSafeAPIResponseValidationError
    import_connect.assert_not_called()
    import_resolve.assert_not_called()


PROGRAM = optimizer.JevProgram("Route this ticket.", {"billing": None, "technical": None})
ROW = {"state": {"ticket": "Duplicate charge"}, "expected": "billing"}


def response(probabilities, choice="technical"):
    # Real SDK models, not a hand-written approximation of the probability field.
    return SystemOneResponse(
        model="offline-fixture", usage={},
        answers={"label": ChoiceAnswer(choice=choice, confidence=1.0, probabilities=probabilities)},
    )


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.network = ExitStack()
        self.addCleanup(self.network.close)
        self.connect = self.network.enter_context(
            patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        )
        self.resolve = self.network.enter_context(
            patch("socket.getaddrinfo", side_effect=AssertionError("Network forbidden"))
        )

    def tearDown(self):
        self.connect.assert_not_called()
        self.resolve.assert_not_called()

    def evaluate(self, result, rows=None, program=PROGRAM):
        client = SimpleNamespace(system_one=Mock(return_value=result))
        return optimizer.evaluate(client, program, [ROW] if rows is None else rows)

    def test_signature_does_not_shadow_dspy_instructions(self):
        self.assertIn("jev_instructions", optimizer.RefineJevPrompt.output_fields)
        self.assertNotIn("instructions", optimizer.RefineJevPrompt.output_fields)
        self.assertIsInstance(optimizer.RefineJevPrompt.instructions, str)

    def test_malformed_http_responses_cannot_be_scored(self):
        # Exercise the SDK's actual decoder without creating a client or using keys.
        for probabilities in ({}, {"technical": 1.0}, {"billing": -1.0, "technical": 2.0}):
            raw = httpx2.Response(200, json={
                "model": "offline-fixture", "usage": {}, "answers": {"label": {
                    "type": "choice", "choice": "technical", "confidence": 1.0,
                    "probabilities": probabilities,
                }},
            }, request=httpx2.Request("POST", "https://offline.invalid"))
            with self.subTest(probabilities=probabilities):
                with self.assertRaises((ValueError, TypeSafeAPIResponseValidationError)):
                    self.evaluate(SystemOneResponse._decode(raw))

    def test_genuine_zero_probabilities_are_scored(self):
        metrics, records = self.evaluate(response({"billing": 0.0, "technical": 1.0}))
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["brier"], 2.0)
        self.assertEqual(records[0].probabilities, {"billing": 0.0, "technical": 1.0})
        metrics, _ = self.evaluate(response({"billing": 1.0, "technical": 0.0}, "billing"))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["brier"], 0.0)

    def test_empty_and_incomplete_distributions_are_not_scores(self):
        for probabilities in ({}, {"technical": 1.0}, {"billing": 0.0},
                              {"billing": 0.0, "technical": 1.0, "extra": 0.0}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.evaluate(response(probabilities))

    def test_nonfinite_and_out_of_range_probabilities_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(response({"billing": value, "technical": 1.0}))

    def test_probability_mass_is_checked_without_renormalizing(self):
        for probabilities in ({"billing": 0.0, "technical": 0.0},
                              {"billing": 0.2, "technical": 0.3},
                              {"billing": 0.5, "technical": 0.511},
                              {"billing": 0.5, "technical": 0.489}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.evaluate(response(probabilities))
        for technical in (0.49, 0.5, 0.51):
            probabilities = {"billing": 0.5, "technical": technical}
            with self.subTest(technical=technical):
                metrics, records = self.evaluate(response(probabilities))
                self.assertEqual(records[0].probabilities, probabilities)
                self.assertAlmostEqual(metrics["brier"], 0.25 + technical ** 2)

    def test_unknown_selected_label_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(response({"billing": 0.0, "technical": 1.0}, "outside"))

    def test_unknown_ground_truth_is_rejected_before_request(self):
        client = SimpleNamespace(system_one=Mock())
        with self.assertRaises(ValueError):
            optimizer.evaluate(client, PROGRAM, [{**ROW, "expected": "outside"}])
        client.system_one.assert_not_called()

    def test_empty_evaluation_is_not_a_perfect_brier_score(self):
        with self.assertRaises(ValueError):
            self.evaluate(response({"billing": 0.0, "technical": 1.0}), rows=[])

    def test_adapter_keys_and_values_are_not_coerced(self):
        # SDK JSON keys are strings. Also guard the answers-view compatibility path
        # against adapters that would alias distinct keys after str() conversion.
        for probabilities in ({1: 0.0, "1": 0.0, "technical": 1.0},
                              {"1": "0", "technical": 1.0},
                              {"1": False, "technical": 1.0}):
            answer = SimpleNamespace(choice="technical", probabilities=probabilities)
            result = SimpleNamespace(answers={"label": answer})
            program = optimizer.JevProgram("Route.", {"1": None, "technical": None})
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                self.evaluate(result, rows=[{**ROW, "expected": "1"}], program=program)

    def test_answers_view_is_supported(self):
        result = SimpleNamespace(answers=response({"billing": 1.0, "technical": 0.0}, "billing").answers)
        metrics, _ = self.evaluate(result)
        self.assertEqual(metrics["brier"], 0.0)

    def optimize(self, candidate_probabilities, invalid_phase=None):
        initial_instructions = PROGRAM.instructions
        seen = []

        def system_one(*, state, questions):
            instructions = questions["label"].instructions
            seen.append((state["split"], instructions))
            if state["split"] == invalid_phase or instructions != initial_instructions:
                return response(candidate_probabilities)
            return response({"billing": 0.0, "technical": 1.0})

        client = Mock()
        client.__enter__ = Mock(return_value=SimpleNamespace(system_one=system_one))
        client.__exit__ = Mock(return_value=False)
        proposal = SimpleNamespace(jev_instructions="Candidate route.",
                                   criteria_json=json.dumps(PROGRAM.criteria), hypothesis="Try clearer labels.")
        splits = [[{**ROW, "state": {"split": split}}] for split in ("train", "validation", "hidden")]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
            patches.enter_context(patch.object(optimizer, "split_dataset", return_value=splits))
            patches.enter_context(patch.object(optimizer, "TypeSafeClient", return_value=client))
            patches.enter_context(patch.object(optimizer.dspy, "LM"))
            patches.enter_context(patch.object(optimizer.dspy, "configure"))
            patches.enter_context(patch.object(optimizer.dspy, "Predict", return_value=Mock(return_value=proposal)))
            patches.enter_context(redirect_stdout(io.StringIO()))
            output = root / "output"
            optimizer.optimize(dataset, 1, 42, "Route tickets.", PROGRAM, output)
            return ({name: json.loads((output / (name + ".json")).read_text(encoding="utf-8"))
                     for name in ("best_program", "history", "final_metrics")}, seen)

    def test_empty_candidate_cannot_beat_valid_wrong_prediction(self):
        artifacts, seen = self.optimize({})
        self.assertEqual(artifacts["best_program"]["instructions"], PROGRAM.instructions)
        history = artifacts["history"][0]
        self.assertFalse(history["accepted"])
        self.assertIsNone(history["validation_metrics"])
        self.assertEqual(history["candidate"]["instructions"], "Candidate route.")
        self.assertIn("Rejected candidate:", history["hypothesis"])
        self.assertEqual(artifacts["final_metrics"]["validation"]["brier"], 2.0)
        self.assertEqual(seen[-1], ("hidden", PROGRAM.instructions))
        self.assertEqual(sum(split == "hidden" for split, _ in seen), 1)

    def test_valid_brier_improvement_still_wins_equal_accuracy(self):
        artifacts, _ = self.optimize({"billing": 0.4, "technical": 0.6})
        self.assertEqual(artifacts["best_program"]["instructions"], "Candidate route.")
        self.assertTrue(artifacts["history"][0]["accepted"])
        self.assertAlmostEqual(artifacts["final_metrics"]["validation"]["brier"], 0.72)

    def test_invalid_baseline_training_and_hidden_results_abort(self):
        for phase in ("validation", "train", "hidden"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.optimize({}, invalid_phase=phase)


if __name__ == "__main__":
    unittest.main()
