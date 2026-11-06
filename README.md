# Knowledge Graphs of Bank Volatility in the COVID-19 Crisis

This study examines whether the volatilities of 70 banks listed in the United States were more closely connected during the COVID-19 crisis than outside it. Following the knowledge graph framework of Chen and Zhang (2024), *From Liquidity Risk to Systemic Risk: A Use of Knowledge Graph*, edges are drawn where the graphical lasso partial correlation exceeds a common cut-off.

## Findings

The crisis graph is denser only in small partial correlations. At a cut-off of 0.035, the 25 largest banks share 244 of the 300 possible edges inside the crisis against 175 outside it.

From a cut-off of 0.040 the ordering reverses. At 0.050 the crisis graph keeps 8 edges and the non-crisis graph 84, which split into a North American and a European group, as in the volatility result of Chen and Zhang (2024).

## Layout

```
data/     committed input prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/bank_prices.csv` holds the daily adjusted closes of the 70 bank stocks from Yahoo Finance for May 2017 to December 2021. The banks are those in the market-capitalization ranking published at stockanalysis.com.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints every number quoted in the report and draws its three figures.

