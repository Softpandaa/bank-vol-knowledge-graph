"""The knowledge graph pipeline of Chen and Zhang (2024), applied to bank volatility."""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import GraphicalLassoCV
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing import StandardScaler

import config


def load_prices():
    return pd.read_csv(config.DATA, index_col=0, parse_dates=True)[config.TICKERS]


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def volatility(returns, window=config.VOL_WINDOW):
    """Annualized standard deviation over a trailing window of window trading days."""
    return returns.rolling(window).std().dropna() * np.sqrt(config.TRADING_DAYS)


def standardize(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)


def subsamples(vol, columns):
    """The study window split into the pandemic crisis and everything else within it."""
    idx = vol.index
    crisis = (idx >= config.CRISIS_START) & (idx < config.CRISIS_END)
    study = (idx >= config.STUDY_START) & (idx <= config.STUDY_END)
    return {"crisis": vol[study & crisis][columns],
            "non-crisis": vol[study & ~crisis][columns]}


def embeddings(scaled):
    """Each bank is one point in the space of its standardized volatility path."""
    out = {}
    out["LLE"] = LocallyLinearEmbedding(
        n_neighbors=config.N_NEIGHBORS, n_components=config.N_COMPONENTS,
        random_state=config.RANDOM_SEED).fit_transform(scaled)
    out["Spectral"] = SpectralEmbedding(
        n_components=config.N_COMPONENTS,
        random_state=config.RANDOM_SEED).fit_transform(scaled)
    out["PCA"] = PCA(n_components=config.N_COMPONENTS,
                     random_state=config.RANDOM_SEED).fit_transform(scaled)
    return out


def clusters(points, n_clusters=config.N_CLUSTERS):
    return KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED).fit_predict(points)


def glasso(scaled):
    """Precision matrix with the penalty chosen by cross validation.

    The panel is close to rank one, so the fold likelihood is not finite at the zero
    penalty end of the search grid. That is a property of this data, reported in the
    text, and it reaches the selected penalty only through scikit-learn's own record
    of the grid scores, which is what the filter below silences."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        model = GraphicalLassoCV(alphas=config.ALPHAS, cv=config.CV,
                                 max_iter=config.MAX_ITER).fit(scaled)
    return model.precision_, float(model.alpha_)


def partial_correlation(precision):
    """Scale free form of the precision matrix, so two fits can be compared cut-off by cut-off."""
    d = np.sqrt(np.diag(precision))
    return -precision / np.outer(d, d)


def off_diagonal(matrix):
    return matrix[np.triu_indices(len(matrix), 1)]


def edges(matrix, cut):
    """Index pairs and weights above the cut-off, for the spring graph."""
    n = len(matrix)
    out = [((i, j), abs(matrix[i, j]))
           for i in range(n) for j in range(i + 1, n) if abs(matrix[i, j]) > cut]
    return [e for e, _ in out], [w for _, w in out]


def edge_count(matrix, cut):
    return int((np.abs(off_diagonal(matrix)) > cut).sum())


def degrees(matrix, cut):
    """Number of edges at each vertex above the cut-off."""
    adjacency = np.abs(matrix) > cut
    np.fill_diagonal(adjacency, False)
    return adjacency.sum(axis=1)
