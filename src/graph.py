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


def volatility(returns):
    """Annualized standard deviation over a trailing window of VOL_WINDOW trading days."""
    return returns.rolling(config.VOL_WINDOW).std().dropna() * np.sqrt(config.TRADING_DAYS)


def standardize(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)


def study_window(vol):
    return vol[(vol.index >= config.STUDY_START) & (vol.index <= config.STUDY_END)]


def subsamples(vol, columns):
    """The study window split into the pandemic crisis and everything else within it."""
    study = study_window(vol)[columns]
    crisis = (study.index >= config.CRISIS_START) & (study.index < config.CRISIS_END)
    return {"crisis": study[crisis], "non-crisis": study[~crisis]}


def pca(scaled):
    return PCA(n_components=config.N_COMPONENTS,
               random_state=config.RANDOM_SEED).fit_transform(scaled)


def embeddings(scaled):
    """The three embeddings of Section 3; each row of scaled is one bank."""
    return {
        "LLE": LocallyLinearEmbedding(n_neighbors=config.N_NEIGHBORS,
                                      n_components=config.N_COMPONENTS,
                                      random_state=config.RANDOM_SEED).fit_transform(scaled),
        "Spectral": SpectralEmbedding(n_components=config.N_COMPONENTS,
                                      random_state=config.RANDOM_SEED).fit_transform(scaled),
        "PCA": pca(scaled),
    }


def clusters(points):
    return KMeans(n_clusters=config.N_CLUSTERS,
                  random_state=config.RANDOM_SEED).fit_predict(points)


def glasso(scaled):
    """Precision matrix of equation (1) and the cross-validated penalty."""
    # Fold scores are -inf at most penalties below the selected one; numpy warns on their sd.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        model = GraphicalLassoCV(alphas=config.ALPHAS, cv=config.CV,
                                 max_iter=config.MAX_ITER).fit(scaled)
    return model.precision_, float(model.alpha_)


def partial_correlation(precision):
    """rho_ij = -Theta_ij / sqrt(Theta_ii Theta_jj) of equation (2)."""
    d = np.sqrt(np.diag(precision))
    return -precision / np.outer(d, d)


def off_diagonal(matrix):
    return matrix[np.triu_indices(len(matrix), 1)]


def edges(matrix, cut):
    """Edges (i, j, |rho_ij|), i < j, above the cut-off."""
    n = len(matrix)
    return [(i, j, abs(matrix[i, j]))
            for i in range(n) for j in range(i + 1, n) if abs(matrix[i, j]) > cut]


def edge_count(matrix, cut):
    return len(edges(matrix, cut))


def degrees(matrix, cut):
    """Number of edges at each vertex above the cut-off."""
    adjacency = np.abs(matrix) > cut
    np.fill_diagonal(adjacency, False)
    return adjacency.sum(axis=1)
