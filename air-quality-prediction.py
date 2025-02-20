#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

pd.set_option('display.max_columns', None)

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Load the training and test data
train_data = pd.read_csv('/kaggle/input/air-quality-datasets/Train.csv')
test_data = pd.read_csv('/kaggle/input/air-quality-datasets/Test.csv')


# In[4]:


train_data.head(2)


# In[5]:


train_data.isnull().sum()


# In[6]:


test_data.head(2)


# In[7]:


# target distribution
plt.figure(figsize = (11, 5))
sns.histplot(train_data.pm2_5)
plt.title('Target Distribution')
plt.show()


# In[8]:


# Check for outliers in the target variable
plt.figure(figsize = (11, 5))
sns.boxplot(train_data.pm2_5)
plt.title('Boxplot showing outliers - target variable')
plt.show()


# In[9]:


train_relevant_features = []
for col in train_data.columns:
    if train_data[col].dtype != 'object' and col != 'id':
        train_relevant_features.append(col)

# Print the list of relevant features
print(train_relevant_features)


# In[10]:


test_relevant_features = []
for col in test_data.columns:
    if test_data[col].dtype != 'object' and col != 'id':
        test_relevant_features.append(col)

# Print the list of relevant features
print(test_relevant_features)


# In[11]:


# Assuming train_data is your DataFrame containing the actual data
train_relevant_features_df = train_data[train_relevant_features]

# Check the newly created DataFrame
train_relevant_features_df.head()


# In[12]:


# Assuming train_data is your DataFrame containing the actual data
test_relevant_features_df = test_data[test_relevant_features]

# Check the newly created DataFrame
test_relevant_features_df.head()


# In[13]:


# Check if 'pm2_5' is missing in the test data and add a placeholder column if needed
if 'pm2_5' not in test_relevant_features_df.columns:
    test_relevant_features_df.loc[:, 'pm2_5'] = 0  # Add a placeholder column 'pm2_5' with value 0

# Initialize the IterativeImputer
imputer = IterativeImputer(random_state=42)

# Fit the imputer on the train and test data, and transform it
train_relevant_features_df_imputed = pd.DataFrame(imputer.fit_transform(train_relevant_features_df), columns=train_relevant_features_df.columns)
test_relevant_features_df_imputed = pd.DataFrame(imputer.transform(test_relevant_features_df), columns=test_relevant_features_df.columns)


# In[14]:


# Check the imputed train DataFrame
train_relevant_features_df_imputed.head()


# In[15]:


# Check the imputed test DataFrame
test_relevant_features_df_imputed.head()


# In[16]:


# Using the train_relevant_features_df_imputed DataFrame
# And also using the test_relevant_features_df_imputed DataFrame

# Calculate correlation matrix for training data
correlation_matrix_train = train_relevant_features_df_imputed.corr()

# Plotting the correlation matrix using heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_train, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Training Data)')
plt.show()


# In[17]:


# Extract the correlation values of all features with the target variable 'pm2_5'
correlation_with_target = correlation_matrix_train['pm2_5']

# Compute the absolute values of these correlations
absolute_correlation_with_target = correlation_with_target.abs()

# Sort the absolute correlation values in descending order
sorted_absolute_correlation_with_target = absolute_correlation_with_target.sort_values(ascending=False)

# Display the sorted absolute correlation values
print(sorted_absolute_correlation_with_target)


# In[18]:


# Plotting the sorted absolute correlation values

plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_absolute_correlation_with_target.index, y=sorted_absolute_correlation_with_target.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Absolute Correlation of Features with pm2_5')
plt.ylabel('Absolute Correlation')
plt.xlabel('features + target variable')
plt.show()


# In[19]:


# Calculate VIF for training data
def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(len(dataframe.columns))]
    return vif_data

train_vif_data = calculate_vif(train_relevant_features_df_imputed)
print(train_vif_data)

# Calculate VIF for test data (if needed)
test_vif_data = calculate_vif(test_relevant_features_df_imputed)
print(test_vif_data)


# In[20]:


# Filter features based on correlation with the target variable 'pm2_5'

target_correlation_threshold = 0.0008  # Set a threshold for correlation with the target variable
target_correlation = correlation_matrix_train['pm2_5'].abs()  # Get absolute correlation values
relevant_features_target = target_correlation[target_correlation > target_correlation_threshold].index.tolist()

relevant_features_target


# In[21]:


# Filter features based on VIF values to remove multicollinear features

vif_threshold = 1000000000  # Set a threshold for VIF
relevant_features_vif = train_vif_data[train_vif_data['VIF'] < vif_threshold]['Feature'].tolist()

relevant_features_vif


# In[22]:


# Take the intersection of relevant features based on correlation and VIF
final_features = list(set(relevant_features_target).intersection(relevant_features_vif))

final_features


# In[23]:


X_train_data = train_relevant_features_df_imputed[final_features]
X_train_data.head()


# In[24]:


X_test_data = test_relevant_features_df_imputed[final_features]
X_test_data.head()


# In[25]:


# Reshape y_train before outlier handling and transformation
y_train = train_data['pm2_5']
y_train_reshaped = y_train.values.reshape(-1, 1)


# In[26]:


# Handle outliers using the IQR method
Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
y_train_clipped = np.where(y_train < lower_bound, lower_bound, y_train)
y_train_clipped = np.where(y_train > upper_bound, upper_bound, y_train)


# In[27]:


# Reshape y_train_clipped before applying PowerTransformer
y_train_clipped_reshaped = y_train_clipped.reshape(-1, 1)

# Apply PowerTransformer to the target variable
transformer = PowerTransformer(method='yeo-johnson', standardize=True)
y_train_transformed = transformer.fit_transform(y_train_clipped_reshaped).flatten()


# In[28]:


column = ['pm2_5']

y_train_transformed_df = pd.DataFrame(y_train_transformed, columns=column)
y_train_transformed_df


# In[29]:


# target distribution
plt.figure(figsize = (11, 5))
sns.histplot(y_train_transformed_df.pm2_5)
plt.title('Target Distribution')
plt.show()


# In[30]:


# Check for outliers in the target variable
plt.figure(figsize = (11, 5))
sns.boxplot(y_train_transformed_df.pm2_5)
plt.title('Boxplot showing outliers - target variable')
plt.show()


# In[31]:


# Define the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler())
])

# Transform the data using the preprocessing pipeline
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train_data)
X_test_preprocessed = preprocessing_pipeline.transform(X_test_data)


# In[32]:


X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=final_features)
X_train_preprocessed_df.head(2)


# In[33]:


# Drop the existing 'pm2_5' column from X_train_preprocessed_df
X_train_preprocessed_df = X_train_preprocessed_df.drop('pm2_5', axis=1)

# Add the new 'pm2_5' column from y_train_transformed_df
X_train_preprocessed_df['pm2_5'] = y_train_transformed_df['pm2_5']


# In[34]:


X_train_preprocessed_df.head()


# In[35]:


# target distribution
plt.figure(figsize = (11, 5))
sns.histplot(X_train_preprocessed_df.pm2_5)
plt.title('Target Distribution')
plt.show()


# In[36]:


# Check for outliers in the target variable
plt.figure(figsize = (11, 5))
sns.boxplot(X_train_preprocessed_df.pm2_5)
plt.title('Boxplot showing outliers - target variable')
plt.show()


# In[37]:


X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=final_features)
X_test_preprocessed_df.head(2)


# In[38]:


# Drop the fictitious 'pm2_5' column from X_train_preprocessed_df
X_test_preprocessed_df = X_test_preprocessed_df.drop('pm2_5', axis=1)


# In[39]:


X_test_preprocessed_df.columns


# In[40]:


# Define the list of models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'SVR': SVR()
}


# In[41]:


# Define the parameter grids

param_grids = {
    'LinearRegression': {
        'fit_intercept': [True, False]
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': [1e-4, 'scale'],
        'kernel': ['linear', 'rbf']
    }
}


# In[42]:


# Define the target variable
target_variable = 'pm2_5'

# Split the preprocessed data into X_train_final and y_train_final
X_train_final = X_train_preprocessed_df.drop(target_variable, axis=1)
y_train_final = X_train_preprocessed_df[target_variable]


# In[43]:


# Define a function to perform the grid search and tune the models
def tune_model(model_name, model, param_grid, X_train_fold, y_train_fold):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_fold, y_train_fold)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_


# In[44]:


# Evaluate each model using KFold cross-validation
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    rmses = []
    for train_index, val_index in kf.split(X_train_final, y_train_final):
        X_train_final_fold, X_val_fold = X_train_final.iloc[train_index], X_train_final.iloc[val_index]
        y_train_final_fold, y_val_fold = y_train_final.iloc[train_index], y_train_final.iloc[val_index]

        # Tune the model using grid search
        tuned_model = tune_model(name, model, param_grids[name], X_train_final_fold, y_train_final_fold)
        
        tuned_model.fit(X_train_final_fold, y_train_final_fold)
        y_val_pred = tuned_model.predict(X_val_fold)
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        rmses.append(rmse)
    
    avg_rmse = np.mean(rmses)
    results[name] = avg_rmse
    print(f'{name} Mean RMSE: {avg_rmse}')


# In[45]:


# Find the best model based on RMSE
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f'Best model: {best_model_name} with Mean RMSE: {results[best_model_name]}')


# In[46]:


# Prepare test data
X_test = X_test_preprocessed_df


# In[47]:


# Train the best model on the entire training set and predict on the test set
best_model = tune_model(best_model_name, best_model, param_grids[best_model_name], X_train_final, y_train_final)
best_model.fit(X_train_final, y_train_final)
y_test_pred = best_model.predict(X_test)


# In[48]:


# Inverse transform the predictions to the original scale
y_test_predicted = transformer.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()


# In[49]:


# Create a DataFrame with 'id' and predicted 'pm2_5' values
result = pd.DataFrame({'id': test_data['id'], 'pm2_5': y_test_predicted})

# Save the predictions to a CSV file with 'id' and 'pm2_5' columns
result.to_csv('air_quality.csv', index=False)


# In[ ]:




