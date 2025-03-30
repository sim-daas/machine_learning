import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn
import warnings


pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


df = pd.read_csv('train.csv', index_col=0)
df.sample(5)

df.info()
df.describe()
corr = df.corr(numeric_only=True)
df.dtypes


df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['hour'] = df['date'].dt.hour
df['min'] = df['date'].dt.minute
df.drop(columns=['date', 'rv1', 'rv2', 'Visibility', 'Tdewpoint'])


plt.figure(figsize=(22, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

df.loc[:, ['Appliances', 'lights','T2', 'T3','RH_5','RH_out','hour']].describe()


sns.displot(data=df, x='lights', kde=True )
plt.title('light distplot')
plt.show()

from sklearn.preprocessing import MinMaxScaler





















