#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared numerical utilities for strict SCED experiments."""

import numpy as np


def project_box_sum(y, lower, upper, total, tol=1e-8, max_iter=100):
    """Project y onto {x | lower <= x <= upper, sum(x) == total}."""
    y = np.asarray(y, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    total = float(total)

    if y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("y, lower, and upper must have identical shapes")
    if np.any(lower > upper):
        raise ValueError("lower bound exceeds upper bound")
    if total < lower.sum() - tol or total > upper.sum() + tol:
        raise ValueError(
            f"infeasible balance target: total={total:.6f}, "
            f"lower_sum={lower.sum():.6f}, upper_sum={upper.sum():.6f}"
        )

    lo = float(np.min(y - upper))
    hi = float(np.max(y - lower))
    x = np.clip(y, lower, upper)
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        x = np.clip(y - lam, lower, upper)
        if x.sum() > total:
            lo = lam
        else:
            hi = lam

    x = np.clip(y - 0.5 * (lo + hi), lower, upper)
    residual = total - float(x.sum())
    if abs(residual) > tol:
        room = (upper - x) if residual > 0 else (x - lower)
        room_sum = float(room.sum())
        if room_sum > tol:
            x += np.sign(residual) * room * min(abs(residual) / room_sum, 1.0)
            x = np.clip(x, lower, upper)
    return x


def zero_sum_unit(direction, tol=1e-12):
    """Return a unit vector projected to the active-power balance subspace."""
    d = np.asarray(direction, dtype=np.float64).copy()
    d -= d.mean()
    norm = float(np.linalg.norm(d))
    if norm <= tol:
        return None
    return d / norm
