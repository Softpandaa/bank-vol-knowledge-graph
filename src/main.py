"""All numbers quoted in the report. Run plots.py, which calls this module."""
import config
import graph
import stats


def results():
    prices = graph.load_prices()
    returns = graph.log_returns(prices)
    vol = graph.volatility(returns)
    out = {"prices": prices, "returns": returns, "vol": vol}

    study = vol[(vol.index >= config.STUDY_START) & (vol.index <= config.STUDY_END)]
    out["study"] = study
    out["embeddings"] = graph.embeddings(graph.standardize(study).T)

    parts = graph.subsamples(vol, config.TOP)
    out["parts"] = parts
    out["panel"] = {}
    for name, part in parts.items():
        scaled = graph.standardize(part)
        precision, alpha = graph.glasso(scaled)
        rho = graph.partial_correlation(precision)
        out["panel"][name] = {
            "n": len(part), "alpha": alpha, "rho": rho,
            "pca": graph.embeddings(scaled.T)["PCA"],
            "dispersion": stats.dispersion(rho),
            "degrees": graph.degrees(rho, config.CUT_LOW),
        }

    out["curve"] = {name: {c: graph.edge_count(d["rho"], c) for c in config.CUT_GRID}
                    for name, d in out["panel"].items()}
    out["crossing"] = stats.crossing(out["curve"], config.CUT_GRID)
    return out


def report(out):
    print(f"prices {out['prices'].shape[0]} days, {out['prices'].shape[1]} banks, "
          f"{out['prices'].index.min().date()} to {out['prices'].index.max().date()}")
    print(f"returns {len(out['returns'])}, volatility {len(out['vol'])}, "
          f"study window {len(out['study'])}")
    for name, d in out["panel"].items():
        deg, order = d["degrees"], d["degrees"].argsort()
        print(f"  {name:10s} n={d['n']:4d} alpha={d['alpha']:.3f} "
              f"partial correlation median {d['dispersion']['median']:.4f} "
              f"sd {d['dispersion']['sd']:.4f}")
        print(f"  {'':10s} at c={config.CUT_LOW}, degree runs {deg.min()} to {deg.max()}, "
              f"least connected "
              + ", ".join(f"{config.TOP[i]} {deg[i]}" for i in order[:3]))
    print(f"  crossing cut-off {out['crossing']}")
    print("  edges out of 300 by common cut-off on the partial correlation")
    for c in config.CUT_GRID:
        print(f"    c={c:.4f}  crisis {out['curve']['crisis'][c]:3d}  "
              f"non-crisis {out['curve']['non-crisis'][c]:3d}")
    print("\nsummary statistics, LaTeX rows")
    print(stats.latex_summary(out["returns"]))
