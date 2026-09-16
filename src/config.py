"""Parameters for the bank knowledge graph study. Every constant is declared here once."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank_prices.csv"
FIG_DIR = ROOT / "latex" / "figures"

TICKERS = [
    "JPM", "BAC", "HSBC", "WFC", "RY", "C", "MUFG", "SAN", "TD", "HDB",
    "SMFG", "BBVA", "UBS", "BMO", "IBN", "MFG", "CM", "ITUB", "BNS", "PNC",
    "USB", "ING", "LYG", "BCS", "NWG", "DB", "TFC", "FITB",
    "BBD", "KB", "HBAN", "MTB", "SHG", "BAP", "CFG", "BSBR", "FCNCA",
    "RF", "KEY", "BCH", "CIB", "WF", "BSAC", "EWBC", "PNFP", "FHN",
    "WTFC", "SSB", "BPOP", "UMBF", "CFR", "ZION", "COLB", "BOKF", "GGAL", "WAL",
    "CBSH", "PB", "VLY", "GBCI", "FNB", "UBSI", "HOMB", "ABCB", "FLG", "HWC",
    "BMA", "AUB", "ASB", "AVAL",
]
TOP = TICKERS[:25]                  # the cross section carried into the spring graphs

STUDY_START, STUDY_END = "2018-06-01", "2021-12-01"
CRISIS_START, CRISIS_END = "2019-12-01", "2020-06-01"

VOL_WINDOW = 252                    # trading days in the rolling volatility estimate
TRADING_DAYS = 252
N_COMPONENTS = 2                    # dimension every embedding maps into
N_NEIGHBORS = 5                     # neighbourhood of the locally linear embedding
N_CLUSTERS = 5                      # k of the k means colouring

ALPHAS = 10                         # size of the penalty grid searched by cross validation
CV = 3
MAX_ITER = 2000                     # the crisis fit does not converge within 500

CUT_LOW = 0.035                     # common low cut-off on the partial correlation
CUT_HIGH = 0.050                    # common high cut-off, past the crossing point
CUT_GRID = [round(0.02 + 0.0025 * i, 4) for i in range(23)]   # 0.020 .. 0.075

RANDOM_SEED = 42
