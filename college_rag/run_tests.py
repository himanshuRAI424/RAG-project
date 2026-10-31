"""Run all test questions in tests/test_cases.json and write tests/results.md.

Usage:
    python run_tests.py            # full pipeline with the LLM
    python run_tests.py --no-llm   # retrieval-only check (fast, no Ollama needed)
"""
import json
import sys
import os
from rag_engine import KnowledgeBase, answer, DOCS_DIR, INDEX_PATH, NOT_FOUND_MSG

use_llm = "--no-llm" not in sys.argv
kb = KnowledgeBase()
kb.build(DOCS_DIR) if not os.path.exists(INDEX_PATH) else kb.load()

cases = json.load(open("tests/test_cases.json", encoding="utf-8"))
rows, passed = [], 0

for c in cases:
    res = answer(c["question"], kb, use_llm=use_llm)
    sources = {s["source"] for s in res["sources"]}
    if use_llm:
        text = res["answer"].lower()
    else:  # retrieval-only: check the retrieved passages instead of the answer
        text = " ".join(s["text"] for s in res["sources"]).lower()

    if c["expect_found"] is False:
        ok = (not res["found"]) or NOT_FOUND_MSG.lower() in text
    else:
        ok = (res["found"]
              and set(c["expected_sources"]).issubset(sources)
              and all(k.lower() in text for k in c["expected_keywords"]))
    passed += ok
    rows.append((c["id"], c["scenario"], c["question"], "PASS" if ok else "FAIL"))
    print(f"[{'PASS' if ok else 'FAIL'}] #{c['id']} {c['question']}")

print(f"\n{passed}/{len(cases)} passed")
with open("tests/results.md", "w", encoding="utf-8") as f:
    f.write(f"# Test results ({'with LLM' if use_llm else 'retrieval only'})\n\n")
    f.write("| # | Scenario | Question | Result |\n|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
    f.write(f"\n**{passed}/{len(cases)} passed**\n")
