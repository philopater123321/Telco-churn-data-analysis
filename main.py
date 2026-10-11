import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("data.csv")

print(df.shape)
print(df.columns)
print(df.info())
print(df["Churn"].value_counts(normalize=True))

s = pd.Series(df["TotalCharges"])
pd.to_numeric(s)
df["TotalCharges"] = s
df["Churn"].value_counts().plot(kind="bar")

plt.title("Churn")
plt.xlabel("Number of customers")
plt.ylabel("Yes or NO")
plt.show()