# Iris Flower Classification

This project uses machine learning to classify iris flowers into three species: Setosa, Versicolor, and Virginica, based on features like sepal and petal length and width.

## Project Structure
- `data/` - contains the Iris dataset (`iris.csv`).
- `src/` - Python scripts for model training (`train_model.py`) and predictions (`predict.py`).
- `notebooks/` - Jupyter notebooks for exploration and model training.
- `README.md` - Project overview and instructions.

## Requirements
Install dependencies using pip:
```
pip install -r requirements.txt
```

## Usage
1. To train the model:
```
python src/train_model.py
```
2. To make predictions:
```
python src/predict.py
```

## Evaluation
The model is evaluated using the `classification_report` from `sklearn`, which gives precision, recall, and F1-score for each class.
