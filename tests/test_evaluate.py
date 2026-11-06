import pytest
from src.evaluate import load_run


def test_load_run_missing(tmp_path, monkeypatch):
    monkeypatch.setattr('src.evaluate.RUNS_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match=str(tmp_path / 'does_not_exist' / 'best.pt')):
        load_run('does_not_exist')


def test_xai_missing_run(tmp_path, monkeypatch):
    from src.xai import main
    monkeypatch.setattr('src.evaluate.RUNS_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='best.pt'):
        main(['--run', 'missing'])
