# model_training.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset (study hours vs test score)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Test_Score": [10, 20, 30, 40, 49, 60, 68, 80, 90, 98]
}
df = pd.DataFrame(data)

# Features and target
X = df[["Hours_Studied"]]
y = df["Test_Score"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
