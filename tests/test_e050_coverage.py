import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import e050_coverage


def test_e050_coverage_synthetic():
    # Tiny synthetic dataset built in the test (no data files)
    dataset = [
        {
            "question_id": "q1",
            "answer": "hello world",
            "haystack_sessions": [
                [  # session 1
                    {"role": "user", "content": "hi", "has_answer": False},
                    {"role": "assistant", "content": "short answer hello world", "has_answer": True},  # short evidence turn, answer within cut
                ]
            ],
        },
        {
            "question_id": "q2",
            "answer": "secret42",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "x" * 100 + " secret42 " + "y" * 2000, "has_answer": True},  # long, answer in first 100 chars
                ]
            ],
        },
        {
            "question_id": "q3",
            "answer": "lateanswer",
            "haystack_sessions": [
                [
                    {"role": "assistant", "content": "x" * 2500 + " lateanswer", "has_answer": True},  # long, answer at char 2500, ends past CUT
                ]
            ],
        },
        {
            "question_id": "q4",
            "answer": "notthere",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "x" * 3000, "has_answer": True},  # long, answer not verbatim
                ]
            ],
        },
        {
            "question_id": "q5",
            "answer": "ignoreme",
            "haystack_sessions": [
                [
                    {"role": "assistant", "content": "x" * 5000, "has_answer": False},  # non-evidence turn, must not be counted
                ]
            ],
        },
    ]
    
    result = e050_coverage.coverage(dataset)
    # Assert every count and coverage == 2/4
    assert result["evidence_turns"] == 4
    assert result["within_cut"] == 1  # q1 only (short)
    assert result["answer_before_cut"] == 1  # q2
    assert result["answer_after_cut"] == 1  # q3
    assert result["answer_not_verbatim"] == 1  # q4
    assert result["coverage"] == 2 / 4
    assert "q3" in result["question_ids_with_answer_after_cut"]


def test_main_writes_out(tmp_path):
    dataset = [
        {
            "question_id": "q1",
            "answer": "hello world",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "hi", "has_answer": False},
                    {"role": "assistant", "content": "short answer hello world", "has_answer": True},
                ]
            ],
        },
        {
            "question_id": "q2",
            "answer": "secret42",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "x" * 100 + " secret42 " + "y" * 2000, "has_answer": True},
                ]
            ],
        },
        {
            "question_id": "q3",
            "answer": "lateanswer",
            "haystack_sessions": [
                [
                    {"role": "assistant", "content": "x" * 2500 + " lateanswer", "has_answer": True},
                ]
            ],
        },
        {
            "question_id": "q4",
            "answer": "notthere",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "x" * 3000, "has_answer": True},
                ]
            ],
        },
        {
            "question_id": "q5",
            "answer": "ignoreme",
            "haystack_sessions": [
                [
                    {"role": "assistant", "content": "x" * 5000, "has_answer": False},
                ]
            ],
        },
    ]
    data = tmp_path / "data.json"
    data.write_text(json.dumps(dataset), encoding="utf-8")
    out = tmp_path / "out.json"
    e050_coverage.main(["--data", str(data), "--out", str(out)])
    assert json.load(open(out)) == e050_coverage.coverage(dataset)
