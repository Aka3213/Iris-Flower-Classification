import joblib
import pandas as pd

model = joblib.load('model/iris_model.pkl')
sample_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
prediction = model.predict(sample_data)
print(f"Predicted species: {prediction[0]}")