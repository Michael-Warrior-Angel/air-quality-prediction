# Air Quality Prediction Using Machine Learning

Overview

This repository contains a machine learning project focused on predicting air quality, specifically PM2.5 levels, using various regression models. The project involves data preprocessing, feature selection, and hyperparameter tuning using GridSearchCV to optimize model performance.

Dataset

The dataset used in this project contains air quality measurements with various meteorological and pollutant features. The target variable is PM2.5 concentration (pm2_5).

Workflow

The project follows these main steps:

Data Preprocessing

Handle missing values.

Remove duplicate rows.

Convert categorical features to numerical representations.

Apply transformations to improve data quality.

Feature Selection

Compute Pearson correlation coefficients to remove highly correlated features.

Use Variance Inflation Factor (VIF) to detect multicollinearity and drop redundant features.

Data Transformation

Apply the Interquartile Range (IQR) method to remove outliers in pm2_5.

Use PowerTransformer to normalize the target variable for better model performance.

Model Training and Hyperparameter Tuning

Train multiple regression models including:

Linear Regression

Random Forest Regressor

Support Vector Regressor (SVR)

Gradient Boosting Regressor

Use GridSearchCV with 5-fold cross-validation to tune hyperparameters and find the best performing model.

Model Evaluation

Compare models using performance metrics such as:

Root Mean Squared Error (RMSE)


Dependencies

The project requires the following Python libraries:

pip install numpy pandas scikit-learn seaborn matplotlib

How to Run the Code

Clone the repository:

git clone https://github.com/yourusername/air-quality-prediction.git
cd air-quality-prediction

Install dependencies:

pip install -r requirements.txt

Run the script:

python air_quality_model.py

Results

The best performing model based on evaluation metrics is identified, and hyperparameter tuning improves its accuracy. The results and final model can be used for air quality forecasting and analysis.

Future Improvements

Incorporate deep learning models such as LSTMs for time-series forecasting.

Explore additional feature engineering techniques.

Deploy the model as a web service for real-time air quality prediction.

Author
Michael K. Tessema
https://github.com/Michael-Warrior-Angel

License

This project is licensed under the MIT License - see the LICENSE file for details.
