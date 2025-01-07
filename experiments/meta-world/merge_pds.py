import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="Debug", type=str,
    help="experiment tag name")
args = parser.parse_args()

a = pd.read_csv('./data/agg_results.csv')
b = pd.read_csv(f'./data/{args.tag}/fusenet.csv')
c = pd.concat([a, b])
c.to_csv(f'./data/{args.tag}/agg_results.csv')