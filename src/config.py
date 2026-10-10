"""Parameters for the bank knowledge graph study, each declared here once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank_prices.csv"
FIG_DIR = ROOT / "latex" / "figures"

# Sample ----------------------------------------------------------------------
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
TOP = TICKERS[:25]                  # largest by market cap, for the graphs
N_PAIRS = len(TOP) * (len(TOP) - 1) // 2
STUDY_START, STUDY_END = "2018-06-01", "2021-12-01"
CRISIS_START, CRISIS_END = "2019-12-01", "2020-06-01"
VOL_WINDOW = 252                    # trading days
TRADING_DAYS = 252                  # per year

# Embedding -------------------------------------------------------------------
N_COMPONENTS = 2
N_NEIGHBORS = 5                     # LLE
N_CLUSTERS = 5                      # k-means
RANDOM_SEED = 42

# Graphical lasso -------------------------------------------------------------
ALPHAS = 10                         # penalty grid size
CV = 3                              # folds
MAX_ITER = 2000                     # the crisis fit does not converge within 1000

# Graph cut-offs --------------------------------------------------------------
CUT_LOW = 0.035
CUT_HIGH = 0.050
CUT_GRID = [round(0.02 + 0.0025 * i, 4) for i in range(23)]   # 0.020 .. 0.075
