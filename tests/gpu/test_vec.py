from __future__ import annotations

import pytest
import torch

from myraytracer.gpu.vec import cross, device, dot, length, normalize


def test_dot_single_vector() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    assert dot(a, b).item() == pytest.approx(32.0)


def test_dot_batch() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
    b = torch.tensor([[4.0, 5.0, 6.0], [0.0, 1.0, 0.0]])

    result = dot(a, b)

    assert result.shape == (2,)
    assert result[0].item() == pytest.approx(32.0)
    assert result[1].item() == pytest.approx(0.0)


def test_cross_single_vector() -> None:
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])

    result = cross(a, b)

    assert result.shape == (3,)
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 1.0]))


def test_cross_batch() -> None:
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = cross(a, b)

    assert result.shape == (2, 3)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(result[1], torch.tensor([1.0, 0.0, 0.0]))


def test_length_single_vector() -> None:
    v = torch.tensor([3.0, 4.0, 0.0])

    assert length(v).item() == pytest.approx(5.0)


def test_length_batch() -> None:
    v = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])

    result = length(v)

    assert result.shape == (2,)
    assert result[0].item() == pytest.approx(5.0)
    assert result[1].item() == pytest.approx(2.0)


def test_normalize_single_vector() -> None:
    v = torch.tensor([3.0, 4.0, 0.0])

    result = normalize(v)

    assert torch.allclose(result, torch.tensor([0.6, 0.8, 0.0]))


def test_normalize_batch() -> None:
    v = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])

    result = normalize(v)

    assert result.shape == (2, 3)
    assert torch.allclose(result[0], torch.tensor([0.6, 0.8, 0.0]))
    assert torch.allclose(result[1], torch.tensor([0.0, 0.0, 1.0]))


def test_normalize_raises_on_zero_length_row() -> None:
    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        normalize(v)


def test_normalize_gradcheck() -> None:
    v = torch.tensor([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(normalize, (v,))


def test_dot_gradcheck() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(dot, (a, b))


def test_device_returns_cpu_when_gpu_not_preferred() -> None:
    assert device(prefer_gpu=False) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_device_returns_cuda_when_available_and_preferred() -> None:
    assert device(prefer_gpu=True) == torch.device("cuda")
