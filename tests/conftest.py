import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: performance benchmarks (deselected by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests unless explicitly selected via ``-m benchmark``."""
    if config.getoption("-m") == "benchmark":
        return

    skip = pytest.mark.skip(reason="benchmarks are not selected, use `-m benchmark`")

    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def seed():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(0)
