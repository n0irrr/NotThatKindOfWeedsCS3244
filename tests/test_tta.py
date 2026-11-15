"""TTA views and the seed / view averaging in src/evaluation/tta.py."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.tta import N_VIEWS, combos, dihedral_view, view_probs  # noqa: E402


def test_views_are_every_flip_and_rotation():
    """The 8 views are distinct, start with the image itself, and are closed under
    the training flips (left-right, up-down) and 90-degree rotation, so they are
    exactly the orientations the flip + rotate augmentation can produce."""
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    views = [dihedral_view(x, i) for i in range(N_VIEWS)]
    keys = {tuple(v.flatten().tolist()) for v in views}
    assert torch.equal(views[0], x)
    assert len(keys) == N_VIEWS
    for v in views:
        for op in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.rot90(1, (-2, -1))):
            assert tuple(op(v).flatten().tolist()) in keys


def test_view_probs_and_combos():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
    data = torch.utils.data.TensorDataset(torch.randn(5, 1, 4, 4), torch.zeros(5, dtype=torch.long))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    p = view_probs(model, loader, torch.device("cpu"))
    assert p.shape == (N_VIEWS, 5, 3)
    np.testing.assert_allclose(p.sum(-1), 1, rtol=1e-5)

    out = combos({1: p, 2: p[::-1]})
    np.testing.assert_allclose(out["s1"], p[0])
    np.testing.assert_allclose(out["s1 + TTA"], p.mean(0), rtol=1e-6)
    np.testing.assert_allclose(out["ensemble"], (p[0] + p[-1]) / 2, rtol=1e-6)
    np.testing.assert_allclose(out["ensemble + TTA"], p.mean(0), rtol=1e-6)
