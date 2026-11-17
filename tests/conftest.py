"""Small audited default CPU profile; archived research tests remain opt-in."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = {
    "test_baseline.py", "test_batching.py", "test_training.py", "test_prepare_refusal.py",
    "test_refusal_common.py", "test_reduction_hooks.py", "test_publish_release.py",
    "test_setup_refusal_artifact.py",
}


def pytest_ignore_collect(collection_path, config):
    # Deselecting after import is too late when a research module imports an
    # optional package or a private runtime. Only this explicit profile filters.
    if config.option.markexpr.strip() == "portable" and collection_path.parent.name == "tests":
        if collection_path.name.startswith("test_") and collection_path.suffix == ".py":
            return collection_path.name not in PORTABLE
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in PORTABLE:
            item.add_marker(pytest.mark.portable)
        else:
            item.add_marker(pytest.mark.research)
        if item.path.name == "test_stage_refusal_7b_release.py" and "raw" in item.fixturenames:
            item.add_marker(pytest.mark.artifact("outputs/refusal-7b-transfer-01/extraction/vectors.pt"))
        for marker in item.iter_markers("artifact"):
            missing = [str(p) for p in marker.args if not (ROOT/p).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(reason="local artifact prerequisite missing: "+", ".join(missing)))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.option.markexpr.strip() == "portable":
        modules = {p.name for p in (ROOT/"tests").glob("test_*.py")}
        terminalreporter.write_line(f"Portable profile: {len(modules & PORTABLE)} modules; "
                                    f"{len(modules - PORTABLE)} research modules excluded before import.")

