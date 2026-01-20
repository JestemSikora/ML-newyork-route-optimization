import pandas as pd
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


df = pd.read_parquet(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\dataset-marged.parquet')


# Train & Test variables
Y = df['time_diffrence [h]']
X = df.drop(columns=['time_diffrence [h]'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)


# Converting dataset into DMatrix structure
xgb_train = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, Y_test, enable_categorical=True)

# Specifying params for our trees
params = {'objective': 'reg:squarederror',
          'learning_rate': 0.01,
          'max_depth': 3,
          'tree_method': 'hist',
          'min_child_weight': 8}

# Watching performence for validation
watchlist = [(xgb_test, "test"), (xgb_train, "train")]
model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=500, evals=watchlist)

y_pred = model.predict(xgb_test)