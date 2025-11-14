import mlflow
import pandas as pd

# Load production model
model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")

# Test data
data = pd.DataFrame({
    'sepal length (cm)': [5.1, 6.2, 5.9],
    'sepal width (cm)': [3.5, 3.4, 3.0],
    'petal length (cm)': [1.4, 5.4, 5.1],
    'petal width (cm)': [0.2, 2.3, 1.8]
})

# Make predictions
predictions = model.predict(data)
probabilities = model.predict_proba(data)

# Display results
target_names = ['setosa', 'versicolor', 'virginica']
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    pred_name = target_names[pred]
    confidence = probs[pred] * 100
    print(f"Sample {i+1}: {pred_name} (confidence: {confidence:.1f}%)")
