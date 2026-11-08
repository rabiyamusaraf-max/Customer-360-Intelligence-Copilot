"""Reproduce the complete Part D evaluation and safety demonstrations."""
import json
import shutil
from pathlib import Path
import pandas as pd

import mock_llm
import src.policy_rag
from src import refund_agent as agent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
STATE = ROOT / "state"


def reset_artifacts():
    OUT.mkdir(exist_ok=True)
    (OUT / "refund_ledger.jsonl").unlink(missing_ok=True)
    (OUT / "agent_runs.jsonl").unlink(missing_ok=True)
    if STATE.exists():
        shutil.rmtree(STATE)
    STATE.mkdir(exist_ok=True)


def main():
    reset_artifacts()

    # Task 4: overlap and zero-overlap policy evaluation.
    overlap_results = src.policy_rag.evaluate(src.policy_rag.PolicyRAG(chunk_size=40, overlap=10))
    zero_results = src.policy_rag.evaluate(src.policy_rag.PolicyRAG(chunk_size=40, overlap=0))
    overlap_results.to_csv(OUT / "policy_eval_results.csv", index=False)
    zero_results.to_csv(OUT / "policy_eval_zero_overlap.csv", index=False)
    policy_metrics = src.policy_rag.evaluation_summary(overlap_results)
    zero_metrics = src.policy_rag.evaluation_summary(zero_results)

    # Tasks 6 and 7: run all seven supplied scenarios with the six-call cap.
    scenarios = pd.read_csv(ROOT / "data" / "raw" / "agent_scenarios.csv", keep_default_na=False)
    mock_llm.reset()
    scenario_rows = []
    with (OUT / "agent_runs.jsonl").open("w", encoding="utf-8") as log:
        for row in scenarios.itertuples(index=False):
            result = agent.run_agent(
                row.message, scenario_id=row.scenario_id,
                max_iterations=6, token_limit=6000, time_limit_seconds=10,
                run_id=f"scenario_{row.scenario_id}",
            )
            refund_amount = 0.0
            for step in result["steps"]:
                if step["tool"] in ("issue_refund", "escalate_to_human"):
                    refund_amount = float(step["observation"].get("refund_inr", 0.0))
            expected_status = str(row.expected_outcome)
            status_ok = (result["status"].startswith("FALLBACK_")
                         if expected_status == "FALLBACK"
                         else result["status"] == expected_status)
            passed = status_ok and round(refund_amount, 2) == round(float(row.expected_refund_inr), 2)
            record = {
                "scenario_id": row.scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            scenario_rows.append({
                "scenario_id": row.scenario_id,
                "status": result["status"],
                "refund_amount_inr": refund_amount,
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "passed": passed,
            })
    scenario_results = pd.DataFrame(scenario_rows)
    scenario_results.to_csv(OUT / "refund_agent_eval.csv", index=False)

    # Task 7: demonstrate the token and timeout guards on S1.
    s1 = scenarios.loc[scenarios.scenario_id.eq("S1")].iloc[0]
    mock_llm.reset()
    token_demo = agent.run_agent(s1.message, "S1_TOKEN_LIMIT", token_limit=200,
                                 run_id="demo_token_limit")
    mock_llm.reset()
    timeout_demo = agent.run_agent(s1.message, "S1_TIMEOUT", time_limit_seconds=0,
                                   run_id="demo_timeout")

    # Task 8: crash after the second persisted tool observation, then resume.
    mock_llm.reset()
    crash_run_id = "crash_resume_S1"
    try:
        agent.run_agent(s1.message, "S1_CRASH_RESUME", run_id=crash_run_id,
                         max_iterations=6, token_limit=6000, time_limit_seconds=10,
                         crash_after=2)
    except RuntimeError as exc:
        if "Simulated crash" not in str(exc):
            raise
    else:
        raise AssertionError("The crash demonstration did not crash")
    resumed = agent.resume_agent(crash_run_id, max_iterations=6,
                                 token_limit=6000, time_limit_seconds=10)
    checkpoint = json.loads((STATE / f"{crash_run_id}.json").read_text(encoding="utf-8"))
    crash_tools = checkpoint["tools_executed"]
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert crash_tools == expected_tools, crash_tools
    assert resumed["status"] == "REFUND"
    assert round(float(resumed["steps"][-1]["observation"]["refund_inr"]), 2) == 128.25

    summary = {
        "policy_rag_overlap_10": policy_metrics,
        "policy_rag_overlap_0": zero_metrics,
        "refund_agent": {
            "scenario_count": int(len(scenario_results)),
            "passed": int(scenario_results.passed.sum()),
            "pass_rate": round(float(scenario_results.passed.mean()), 4),
        },
        "stop_safely": {
            "token_limit_demo_status": token_demo["status"],
            "timeout_demo_status": timeout_demo["status"],
            "s7_status": scenario_results.loc[scenario_results.scenario_id.eq("S7"), "status"].iloc[0],
        },
        "crash_resume": {
            "status": resumed["status"],
            "tools_executed_in_order": crash_tools,
            "llm_calls": resumed["llm_calls"],
            "tokens_used": resumed["tokens_used"],
        },
    }
    (OUT / "part_d_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("POLICY RAG (10-word overlap):", policy_metrics)
    print("POLICY RAG (zero overlap):", zero_metrics)
    print("\nREFUND SCENARIOS")
    print(scenario_results.to_string(index=False))
    print("\nTOKEN LIMIT DEMO:", token_demo["status"], token_demo["llm_calls"], token_demo["tokens_used"])
    print("TIMEOUT DEMO:", timeout_demo["status"], timeout_demo["llm_calls"], timeout_demo["tokens_used"])
    print("CRASH/RESUME:", resumed["status"], " -> ".join(crash_tools))
    print("\nPART D SUMMARY")
    print(json.dumps(summary, indent=2))
    assert policy_metrics["source_recall"] == 1.0
    assert policy_metrics["expected_fact_rate"] == 1.0
    assert policy_metrics["unknown_fallback_rate"] == 1.0
    assert zero_metrics["expected_fact_rate"] < 1.0
    assert scenario_results.passed.all()
    assert summary["stop_safely"]["s7_status"] == "FALLBACK_MAX_ITERATIONS"
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"


if __name__ == "__main__":
    main()
