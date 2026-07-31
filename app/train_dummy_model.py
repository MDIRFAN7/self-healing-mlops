from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

X = np.random.rand(200, 4)
y = np.random.randint(0, 2, 200)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("model.pkl created successfully")