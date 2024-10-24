# Machine Learning Models for Predicting Student's Academic Performance

## Overview

This task applies several machine learning regression models to predict students' **math scores** based on various features from the `StudentsPerformance.csv` dataset. 

## Project Steps

### 2. Model Selection and Training

Various regression models are trained on the dataset to predict math scores:
- **Linear Regression**
- **Lasso Regression**
- **K-Neighbors Regressor**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**
- **AdaBoost Regressor**

The performance of each model is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

These metrics are calculated for both the training and testing sets, giving insight into the model's performance and overfitting behavior.

### 3. Hyperparameter Tuning

**GridSearchCV** is applied to each model to tune the hyperparameters. The key hyperparameters tuned include:
- **Lasso**: `alpha`
- **K-Neighbors Regressor**: `n_neighbors`
- **Decision Tree**: `max_depth`, `criterion`
- **Random Forest Regressor**: `n_estimators`, `max_depth`
- **Gradient Boosting**: `learning_rate`, `subsample`, `n_estimators`
- **XGBoost**: `depth`, `learning_rate`, `iterations`
- **CatBoost**: `iterations`, `depth`
- **AdaBoost**: `learning_rate`, `n_estimators`

### 4. Model Performance Evaluation

After training, the models are evaluated based on the best hyperparameters found using GridSearchCV. Performance metrics (RMSE, MSE, MAE, R²) for both the training and test sets are printed for each model. Additionally, the best hyperparameters for each model are displayed.

A scatter plot and a regression plot are generated to visualize the relationship between the **actual** and **predicted** math scores.
