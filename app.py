from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['sepal_length']),
        float(request.form['sepal_width']),
        float(request.form['petal_length']),
        float(request.form['petal_width'])
    ]

    sample = pd.DataFrame([data], columns=[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ])

    prediction = model.predict(sample)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
