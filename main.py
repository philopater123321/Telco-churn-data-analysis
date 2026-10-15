import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("data.csv")

print(df.shape)
print(df.columns)
print(df.info())
print(df["Churn"].value_counts(normalize=True))


df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df['TotalCharges'].dtype)
print(df["TotalCharges"].isnull().sum())

df["Churn"].value_counts().plot(kind="bar")

plt.title("Churn")
plt.xlabel("Yes or No")
plt.ylabel("Number of Churns")
plt.show()