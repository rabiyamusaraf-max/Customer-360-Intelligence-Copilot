"""Acceptance and safety tests for Part D."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.policy_rag import PolicyRAG, evaluate, evaluation_summary
from src import refund_agent as agent


class PolicyRAGTests(unittest.TestCase):
    def test_policy_eval_checkpoints(self):
        results = evaluate(PolicyRAG())
        metrics = evaluation_summary(results)
        self.assertEqual(metrics["questions"], 14)
        self.assertEqual(metrics["source_recall"], 1.0)
        self.assertEqual(metrics["expected_fact_rate"], 1.0)
        self.assertEqual(metrics["unknown_fallback_rate"], 1.0)

    def test_reject_empty_query(self):
        with self.assertRaises(ValueError):
            PolicyRAG().retrieve("   ")


class RefundAgentTests(unittest.TestCase):
    def test_calculation_uses_strictly_more_than_45_minutes(self):
        self.assertEqual(agent.calculate_refund(200, 45)["refund_inr"], 0.0)
        self.assertEqual(agent.calculate_refund(200, 46)["refund_inr"], 200.0)

    def test_unknown_order(self):
        self.assertEqual(agent.get_order("ORD-999999"), {"error": "order not found"})

    def test_auto_refund_ledger_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "ledger.jsonl"
            with patch.object(agent, "LEDGER", ledger):
                first = agent.issue_refund("ORD-100048", 128.25)
                second = agent.issue_refund("ORD-100048", 128.25)
            self.assertFalse(first["already_processed"])
            self.assertTrue(second["already_processed"])
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)

    def test_auto_refund_cannot_bypass_human_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(agent, "LEDGER", Path(folder) / "ledger.jsonl"):
                with self.assertRaises(ValueError):
                    agent.issue_refund("ORD-100145", 1999.0)

    def test_cancelled_order_cannot_be_refunded(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(agent, "LEDGER", Path(folder) / "ledger.jsonl"):
                with self.assertRaises(ValueError):
                    agent.issue_refund("ORD-100025", 45.0)

    def test_scenario_acceptance(self):
        # Use a temporary ledger so the test is independent of previous local runs.
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(agent, "LEDGER", Path(folder) / "ledger.jsonl"):
                results = agent.evaluate_scenarios()
        self.assertTrue(results.passed.all(), results.to_string(index=False))
        self.assertEqual(len(results), 7)


    def test_stop_safely_token_and_timeout(self):
        import mock_llm
        with tempfile.TemporaryDirectory() as folder:
            old_ledger, old_state = agent.LEDGER, agent.STATE
            agent.LEDGER = Path(folder) / "ledger.jsonl"
            agent.STATE = Path(folder) / "state"
            mock_llm.reset()
            token = agent.run_agent(
                "My order ORD-100048 reached me very late. I would like a refund.",
                "S1", token_limit=200, run_id="token-test",
            )
            mock_llm.reset()
            timeout = agent.run_agent(
                "My order ORD-100048 reached me very late. I would like a refund.",
                "S1", time_limit_seconds=0, run_id="timeout-test",
            )
            agent.LEDGER, agent.STATE = old_ledger, old_state
        self.assertEqual(token["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(timeout["status"], "FALLBACK_TIMEOUT")
        self.assertEqual(timeout["llm_calls"], 0)

    def test_crash_resume_does_not_repeat_tools(self):
        import mock_llm
        with tempfile.TemporaryDirectory() as folder:
            old_ledger, old_state = agent.LEDGER, agent.STATE
            agent.LEDGER = Path(folder) / "ledger.jsonl"
            agent.STATE = Path(folder) / "state"
            mock_llm.reset()
            run_id = "resume-test"
            with self.assertRaises(RuntimeError):
                agent.run_agent(
                    "My order ORD-100048 reached me very late. I would like a refund.",
                    "S1", run_id=run_id, crash_after=2,
                )
            result = agent.resume_agent(run_id)
            agent.LEDGER, agent.STATE = old_ledger, old_state
        self.assertEqual(result["status"], "REFUND")
        self.assertEqual([s["tool"] for s in result["steps"]],
                         ["get_order", "search_policy", "calculate_refund", "issue_refund"])


if __name__ == "__main__":
    unittest.main()
