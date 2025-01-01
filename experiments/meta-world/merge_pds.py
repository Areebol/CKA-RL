import pandas as pd

a = pd.read_csv('data/agg_results.csv')
b = pd.read_csv('tmp1.csv')
c = pd.concat([a, b])
c.to_csv('agg_results.csv')