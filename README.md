# Knowledge Graphs of Bank Volatility in the COVID-19 Crisis

This study applies the knowledge graph framework of Chen and Zhang (2024), *From Liquidity Risk to Systemic Risk: A Use of Knowledge Graph*, to 70 banks listed in the United States over the COVID-19 episode. At a cut-off $c=0.035$, we found that none of the 25 largest banks are isolated during the crisis, and the knowledge graph holds a dense core. This aligns with the expectation that firms tend to become more interconnected during crisis periods when common macroeconomic shocks overshadow idiosyncratic risks. Nevertheless, this is a claim about weak links only. A large share of strong conditional dependence in this sample occurs in the non-crisis period. At $c = 0.050$, there are 8 edges inside the crisis against 84 outside it, and the surviving structure matches the geographic split of the non-crisis graph. This matches the volatility result of Chen and Zhang (2024), which they describe as counterintuitive relative to their liquidity result.

## Layout

```
data/     committed input prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

## Data

`data/bank_prices.csv` is the 70 bank stocks daily close from Yahoo Finance, from May 2017 to December 2021. The universe is the ranking by market capitalization published at stockanalysis.com.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints every number quoted in the report and rewrites the three figures it uses to `latex/figures/`, which is created on the first run and is not tracked.

