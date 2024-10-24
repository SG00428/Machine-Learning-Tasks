# Iris Species Prediction with Machine Learning and Flask

## Overview

This project focuses on predicting the species of the iris flower using machine learning models (Logistic Regression and Decision Tree) trained on the popular Iris dataset. It also includes a Flask web application to allow users to input flower measurements and get predictions.

## Files and Code Structure

1. **Task_2.ipynb : Training and Evaluating ML Models & Predicting on New Data**
    - Implements two machine learning models:
        - **Logistic Regression**
        - **Decision Tree Classifier**
    - Accuracy, Confusion matrices, Feature importance are visialuzed
    - Predict the species of new iris flower measurements

3. **task2.py : Flask Web Application for Predictions**
    - Implements a simple Flask web application.
    - The web app allows users to enter the following features of the iris flower:
        - Sepal Length
        - Sepal Width
        - Petal Length
        - Petal Width
    - The user can choose between Logistic Regression and Decision Tree models to make predictions.
    - The prediction is displayed on the web page after submission.
    
## How to Run the Project

### Requirements
- Python 3.x
- Required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    **Required Libraries**:
    - `Flask`
    - `scikit-learn`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `numpy`

### Running the ML models (Task_2.ipynb)

1. **Run the model training and evaluation code**:
    - Execute the script to train the models and evaluate them using accuracy, confusion matrices, and feature importance in Jupyter Notebook or Colab

2. **Predict on new data**:
    - Test the model on random new iris data to see the prediction results.

### Running the Flask App (task2.py)

1. Run the Flask web app:
    ```bash
    python task2.py
    ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`.

3. Input the flower measurements and select a model to get a prediction for the iris species.

## Conclusion

This project demonstrates how to use machine learning models to classify iris species and provides a simple Flask-based web application to interactively make predictions based on user inputs. The project highlights the differences in accuracy between Logistic Regression and Decision Tree models.
