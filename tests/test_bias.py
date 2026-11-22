"""Per-class decision offsets in src/evaluation/bias.py."""
import numpy as np

from src.evaluation.bias import apply_bias, cv_gain, fit_bias, macro_f1


def skewed_problem(n=600, seed=0):
    """3 classes; the model under-predicts class 0 in favour of class 2 (the 'Negative')."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 2.0
    logits[:, 2] += 1.5                      # systematic pull towards class 2
    log_p = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return log_p, y


def test_fit_bias_improves_and_keeps_fixed_class_at_zero():
    log_p, y = skewed_problem()
    before = macro_f1(y, log_p.argmax(1), 3)
    bias, after = fit_bias(log_p, y, fixed=2)
    assert bias[2] == 0
    assert after > before + 0.05
    assert np.isclose(after, macro_f1(y, apply_bias(log_p, bias), 3))
    assert bias[0] > 0 and bias[1] > 0       # undoes the pull towards class 2


def test_fit_bias_leaves_a_calibrated_model_alone():
    y = np.repeat(np.arange(3), 50)
    log_p = np.full((150, 3), np.log(0.1))
    log_p[np.arange(150), y] = np.log(0.8)   # always right: nothing to gain
    bias, after = fit_bias(log_p, y, fixed=2)
    assert np.all(bias == 0)
    assert after == 1.0


def test_cv_gain_is_out_of_fold():
    log_p, y = skewed_problem(seed=2)
    before, oof = cv_gain(log_p, y, fixed=2)
    assert np.isclose(before, macro_f1(y, log_p.argmax(1), 3))
    assert oof > before                      # the pull towards class 2 is real, so it generalises

    # pure noise: offsets fitted on 4/5 should not reliably help the held-out 1/5
    rng = np.random.default_rng(3)
    noise = np.log(rng.dirichlet(np.ones(3), 600))
    y_noise = rng.integers(0, 3, 600)
    before_n, oof_n = cv_gain(noise, y_noise, fixed=2)
    _, in_sample_n = fit_bias(noise, y_noise, fixed=2)
    assert in_sample_n - before_n > oof_n - before_n   # in-sample overstates the gain
