
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


df=pd.read_csv("data.csv")
df.head()

print(df["Churn"].value_counts(normalize=True))



df["TotalCharges"] =pd.to_numeric(df["TotalCharges"], errors= "coerce")
print(df["TotalCharges"].dtype)
print(df["TotalCharges"].isnull().sum())


df=df.dropna(subset=["TotalCharges"]).copy()
print(df["TotalCharges"].isnull().sum())

x = df.drop(columns=["Churn", "customerID"])
y = df['Churn']
print("Features shape", x.shape)
print("Target shape", y.shape)
print(y.value_counts())


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.4,
    random_state=42,
    stratify=y

)

print(x_train.shape)
print(x_test.shape)

numeric_features = x_train.select_dtypes(
    include=["int64", "float64"]
).columns

category_features = x_train.select_dtypes(
    include=["object", "str"]
).columns

print("Numerical features: ", numeric_features)
print("Category features:", category_features)

numeric_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

category_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num", numeric_transform, numeric_features),
        ("category", category_transformer, category_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter = 1000))
    ])

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction))

print("Prediction results:")
print(confusion_matrix(y_test, y_prediction))

y_prob = model.predict_proba(x_test)[:, 1]

print("ROC-AUC score: ", roc_auc_score(
    (y_test == "Yes").astype(int),
    y_prob
))


print(df.groupby("Contract")["Churn"].value_counts(normalize=True))


result = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
result["Yes"].plot(kind="bar")

plt.title("Churn by Contract")
plt.xlabel("Contract")
plt.ylabel("Churn Rate")
plt.show()


result = df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean()

print(result)


