import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimpy as skim

df = pd.read_csv('train.csv', index_col=0)
df.sample(5)
df.info()

df.columns

skim.skim(df)

use_cols = df.columns.to_list()

df.isna().sum()

df[df.fuel_type.isna() & df.engine.str.contains('Electric')]

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
        print(df[col].nunique())

df.drop_duplicates(inplace=True)

df.drop()











 