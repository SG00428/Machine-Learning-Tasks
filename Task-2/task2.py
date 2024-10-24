from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load iris dataset and create DataFrame
iris = load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['species'])

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_scaled, y.values.ravel())

# Train decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_scaled, y)

# Function to predict species
def predict_species(model, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return iris['target_names'][prediction][0]

# HTML template for the form
form_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Species Predictor</title>
</head>
<body>
    <h1>Iris Species Predictor</h1>
    <form method="POST" action="/predict">
        <label>Sepal Length:</label>
        <input type="number" step="any" name="sepal_length" required><br><br>
        <label>Sepal Width:</label>
        <input type="number" step="any" name="sepal_width" required><br><br>
        <label>Petal Length:</label>
        <input type="number" step="any" name="petal_length" required><br><br>
        <label>Petal Width:</label>
        <input type="number" step="any" name="petal_width" required><br><br>
        <label>Select Model:</label>
        <select name="model">
            <option value="logistic">Logistic Regression</option>
            <option value="tree">Decision Tree</option>
        </select><br><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
    <h2>Predicted Species: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

# Home route
@app.route('/')
def home():
    return render_template_string(form_template)

# Prediction route that handles form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    model_type = request.form['model']

    features = [sepal_length, sepal_width, petal_length, petal_width]

    # Predict using the selected model
    if model_type == 'logistic':
        species = predict_species(log_reg, features)
    elif model_type == 'tree':
        species = predict_species(tree_clf, features)
    else:
        return "Model type not supported."

    # Render the form again with the prediction
    return render_template_string(form_template, prediction=species)

if __name__ == '__main__':
    app.run(debug=True)
