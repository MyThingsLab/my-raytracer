from __future__ import annotations

import pytest

from myraytracer.core.backend import NUMPY, Backend, torch_backend


def _available_backends() -> list[pytest.param]:
    backends = [pytest.param(NUMPY, id="numpy")]
    try:
        import torch

        backends.append(pytest.param(torch_backend(device=torch.device("cpu")), id="torch"))
    except ImportError:
        pass
    return backends


@pytest.fixture(params=_available_backends())
def backend(request: pytest.FixtureRequest) -> Backend:
    # Every backend-agnostic test runs once per installed backend (numpy
    # always; torch on CPU when the optional dependency is present).
    return request.param
