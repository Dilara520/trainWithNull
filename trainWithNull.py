# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames
import matplotlib.pyplot as plt # for data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# Reading the data
dataset = pd.read_csv('trainmaternal _withNulls.csv')
#Print the first 5 rows
dataset.head()

#Change the RiskLevel values to integer.
dataset['RiskLevel'] = dataset['RiskLevel'].replace('low risk', 0).replace('mid risk', 1).replace('high risk', 2)

#Print the first 5 row and see the difference.
dataset.head()
#here are some NULL values (NaN) in the dataset. We need to handle them later.

# Use info() function
dataset.info()

# Use isnull() function
dataset.isnull()

#Filling missing (null) values

#method 1
# Create a dataset with filling missing (null) values with zero. Do not change the original dataset.
dataset1 = dataset.fillna(0)

#method2
# Create a dataset with filling missing values with next ones. Do not change the original dataset.  
dataset2 = dataset.fillna(method ='bfill')

#method 3
# Create a new dataset with dropping null values. Do not change the original dataset. 
dataset3 = dataset.dropna()
#dataset got smaller, this is not desirable when you have little data

#method 4
# Create a dataset with filling the null values with the mean of the values from that column for each class. Do not change the original dataset. 
dataset4 = (dataset.fillna(dataset.groupby('RiskLevel').transform('mean')).astype(int))

#continuing with method 4
#random forest classification
y = dataset4['RiskLevel']
X = dataset4.drop(['RiskLevel'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

#Scaling
n_scaler = MinMaxScaler()
X_train_scaled = n_scaler.fit_transform(X_train.astype(np.float)) #fit_transform
X_test_scaled = n_scaler.transform(X_test.astype(np.float)) #transform

rfc = RandomForestClassifier(n_estimators=3, max_depth=2)

# Fit RandomForestClassifier
rfc.fit(X_train_scaled, y_train)

# Predict the test set labels
y_pred = rfc.predict(X_test_scaled)

#print the classification report
print(classification_report(y_test,y_pred))

# Define the grid
param_grid = {
'n_estimators': [50, 100, 200],
'min_samples_leaf': [1, 5, 7],
'max_depth': [4, 6, 8],
'max_features': ['auto', 'sqrt'],
'bootstrap': [True, False]}

# Create the GridSearchCV model
model_gridsearch = GridSearchCV(
estimator=rfc,
param_grid=param_grid,
cv=3
)

# Fit the GridSearchCV model
model_gridsearch.fit(X_train_scaled, y_train)

print(model_gridsearch.best_params_)

rfc_last = RandomForestClassifier(bootstrap=True, max_depth= 8, max_features= 'sqrt', min_samples_leaf= 1, n_estimators= 50)

# Fit RandomForestClassifier
rfc_last.fit(X_train_scaled, y_train)

# Predict on the test set 
y_pred_last = rfc_last.predict(X_test_scaled)

print(classification_report(y_test,y_pred_last))