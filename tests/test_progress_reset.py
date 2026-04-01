import json
import pytest
from utils.progress import reset_stages


@pytest.fixture
def mock_progress(tmp_path, monkeypatch):
    """Use a temp progress file."""
    p = tmp_path / "progress.json"
    monkeypatch.setattr("utils.progress.PROGRESS_FILE", p)
    state = {
        "data_preparation": "done",
        "embeddings": {"bert": "done", "finbert": "done"},
        "finetune": {"bert_separate_1d_split": "done", "bert_single_split": "done"},
        "evaluate": "done",
        "market_trend": "done",
        "last_updated": "2026-01-01T00:00:00",
        "last_error": None,
    }
    p.write_text(json.dumps(state))
    return p


def test_reset_single_stage(mock_progress):
    reset_stages(["evaluate"])
    state = json.loads(mock_progress.read_text())
    assert state["evaluate"] == "pending"
    assert state["data_preparation"] == "done"


def test_reset_dict_stage(mock_progress):
    reset_stages(["finetune"])
    state = json.loads(mock_progress.read_text())
    assert state["finetune"] == {}
    assert state["embeddings"]["bert"] == "done"


def test_reset_multiple_stages(mock_progress):
    reset_stages(["finetune", "evaluate", "market_trend"])
    state = json.loads(mock_progress.read_text())
    assert state["finetune"] == {}
    assert state["evaluate"] == "pending"
    assert state["market_trend"] == "pending"
    assert state["data_preparation"] == "done"
