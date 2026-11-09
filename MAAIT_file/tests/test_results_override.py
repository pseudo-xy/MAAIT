"""Starting a new simulation replaces the previous run's results."""
from reschu.web import server


def test_clear_previous_results(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    for name in ("results-2026-11-06_00-17-18-summary.csv",
                 "results-2026-11-06_00-17-18-contacts.csv"):
        (tmp_path / name).write_text("x")
    (tmp_path / "notes.txt").write_text("keep me")
    assert server._clear_previous_results() == 2
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]
