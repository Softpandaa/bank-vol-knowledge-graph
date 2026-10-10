"""Summary statistics and the diagnostics reported alongside the graphs."""
import numpy as np
import pandas as pd

import config
import graph


def yearly_summary(returns):
    """Log returns pooled across banks, one row per calendar year, annualized and in percent."""
    d = config.TRADING_DAYS
    out = returns.groupby(returns.index.year).apply(
        lambda g: pd.Series({"mean": 100 * d * g.values.mean(),
                             "median": 100 * d * np.median(g.values),
                             "sd": 100 * np.sqrt(d) * g.values.std(ddof=1)}))
    out.index.name = "year"
    return out


def latex_summary(returns):
    return "\n".join(f"{year} & {r['mean']:+.1f} & {r['median']:+.1f} & {r['sd']:.1f} \\\\"
                     for year, r in yearly_summary(returns).iterrows())


def dispersion(matrix):
    """Median and sd of |rho_ij| over all pairs i < j."""
    v = np.abs(graph.off_diagonal(matrix))
    return {"median": float(np.median(v)), "sd": float(v.std(ddof=1))}


def crossing(curves, grid):
    """The cut-off at which the crisis curve falls below the non-crisis curve."""
    for c in grid:
        if curves["crisis"][c] < curves["non-crisis"][c]:
            return c
    return None
