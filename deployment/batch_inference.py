import mlflow
import pandas as pd

model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")
data = pd.DataFrame({
    'sepal length (cm)': [5.1, 6.2, 5.9],
    'sepal width (cm)': [3.5, 3.4, 3.0],
    'petal length (cm)': [1.4, 5.4, 5.1],
    'petal width (cm)': [0.2, 2.3, 1.8]
})
predictions = model.predict(data)
print("Predictions:", predictions)
