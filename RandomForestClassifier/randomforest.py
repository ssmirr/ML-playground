# Based on: https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
###########################################

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np

import sys

# Create a dataframe from csv file
df = pd.read_csv(sys.argv[1])

# View the top 5 rows
print('================================')
print('Top five rows of the dataset:\n')
print(df.head())
print('================================')

# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# View the top 5 rows after adding is_train
# print(df.head())

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print('================================')

# Create a list of the feature column's names
# choosing Weighted Density and Density columns
features = df.columns[-3:-1]

# View features
print('features: ', features)
print('================================')

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['Kind'])[0]

# # View target
# print('target: ', y)
# print('================================')

# # Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

# print('================================')
preds = clf.predict(test[features])
# # View the PREDICTED kinds for the first five observations
# print('preds:\n', preds[0:5])
# # View the ACTUAL kinds for the first five observations
# print('actual\n', test['Kind'].head())
# print('================================')

# Create confusion matrix
print('Confusion matrix: novice(0)  expert(1)')
print(pd.crosstab(test['Kind'], preds, rownames=['Actual Kinds'], colnames=['Predicted Kinds']))
print('================================')

# View a list of the features and their importance scores
print('Feature importance: ')
print(list(zip(train[features], clf.feature_importances_)))
print('================================')
