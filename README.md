# AdaBoost – Classification & Regression

## Overview

This repository contains Jupyter Notebooks demonstrating **AdaBoost (Adaptive Boosting)** for both **classification** and **regression** tasks. The notebooks focus on understanding how boosting works by combining multiple weak learners to form a strong predictive model.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. AdaBoost Classification  
4. AdaBoost Regression  
5. Model Evaluation  

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `AdaboostClassification.ipynb` – AdaBoost applied to a classification problem  
- `AdaboostRegression.ipynb` – AdaBoost applied to a regression problem  

---

## AdaBoost Classification

### `AdaboostClassification.ipynb`

This notebook demonstrates **AdaBoost Classifier**, where multiple weak learners (typically decision stumps) are combined to improve classification performance.

Key points:
- Misclassified samples are given higher weight
- Each new model focuses on correcting previous errors
- Improves performance over a single weak learner

Basic commands used:
```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
```

The notebook also includes dataset exploration and evaluation of classification performance.

---

## AdaBoost Regression

### `AdaboostRegression.ipynb`

This notebook applies **AdaBoost Regressor** to predict continuous target values.

Key points:
- Boosting is applied to regression trees
- Focuses on reducing large prediction errors
- Useful for capturing complex patterns in data

Basic commands used:
```python
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_reg.fit(X_train, y_train)
y_pred = ada_reg.predict(X_test)
```

The notebook focuses on data preparation and evaluating regression performance.

---

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

