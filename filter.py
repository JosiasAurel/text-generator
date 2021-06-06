import pandas as pd
import sys

file = sys.argv[1]

df = pd.read_json(file, lines=True)

df.drop(df.columns.difference(["content"], 1, inplace=True))

df.to_json(f"l{file}")
