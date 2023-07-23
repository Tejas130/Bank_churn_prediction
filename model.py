import pandas as pd
import numpy as np
import pickle

# load dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')
# customer id is unique for every person so this not useful in our analysis
df.drop('customer_id', axis=1, inplace=True)
# split our dataframe into x-feature variable and y-target variable.
X = df.drop('churn', axis=1)  # Features (all columns except 'churn')
Y = df['churn']  # Target variable ('churn' column)
from sklearn.model_selection import train_test_split

random_state = 10

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=random_state,
                                                    stratify=df.churn)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

# Need to be numeric not string to specify columns name
preprocess = make_column_transformer(
    (MinMaxScaler(), [0, 3, 4, 5, 6, 7, 8, 9]),
    (OneHotEncoder(sparse=False), [1, 2])
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Define the pipeline
RF_model = make_pipeline(preprocess, RandomForestClassifier(random_state=random_state))

# Define the parameter grid
rf_param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 5, 10],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}

# Create the GridSearchCV model
rf_grid = GridSearchCV(RF_model, rf_param_grid, verbose=3, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
print(rf_grid.fit(X_train, y_train))
# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(rf_grid, file)
