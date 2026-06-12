import pytest

from prime_rl.configs.algorithm import StaticDatasetConfig
from prime_rl.orchestrator.static_sft import static_sft_rollout


def test_static_sft_rollout_splits_messages_on_assistant_turns():
    row = {
        "example_id": "ex-1",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
            {"role": "user", "content": "Three"},
            {"role": "assistant", "content": "Four"},
        ],
    }

    rollout = static_sft_rollout(row, StaticDatasetConfig(name="org/static-sft"))

    assert rollout["example_id"] == "ex-1"
    assert rollout["error"] is None
    assert rollout["reward"] == 1.0
    assert len(rollout["trajectory"]) == 2
    assert rollout["trajectory"][0]["prompt"] == row["messages"][:2]
    assert rollout["trajectory"][0]["completion"] == [row["messages"][2]]
    assert rollout["trajectory"][0]["tokens"] is None
    assert rollout["trajectory"][1]["prompt"] == row["messages"][:4]
    assert rollout["trajectory"][1]["completion"] == [row["messages"][4]]


def test_static_sft_rollout_accepts_prompt_completion_columns():
    row = {"prompt": "Question", "completion": "Answer"}

    rollout = static_sft_rollout(row, StaticDatasetConfig(name="org/static-sft"))

    assert rollout["trajectory"][0]["prompt"] == [{"role": "user", "content": "Question"}]
    assert rollout["trajectory"][0]["completion"] == [{"role": "assistant", "content": "Answer"}]


def test_static_sft_rollout_requires_assistant_target():
    with pytest.raises(ValueError, match="no assistant messages"):
        static_sft_rollout(
            {"messages": [{"role": "user", "content": "Question"}]},
            StaticDatasetConfig(name="org/static-sft"),
        )
