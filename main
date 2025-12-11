import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data = {
    "age": [25, 32, 47, 51, 62, 23, 38, 44],
    "salary": [2500, 3200, 4700, 5100, 6200, 2300, 3800, 4400],
    "purchased": [0, 1, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data) 

X = df[["age", "salary"]]
y = df["purchased"]


model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())
