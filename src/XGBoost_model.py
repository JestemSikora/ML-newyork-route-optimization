import pandas as pd
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


df = pd.read_parquet(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\dataset-marged.parquet')


# Train & Test variables
Y = df['time_diffrence h']
X = df.drop(columns=['time_diffrence h'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)


# Converting dataset into DMatrix structure
xgb_train = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, Y_test, enable_categorical=True)

# Specifying params for our trees
params = {'objective': 'reg:squarederror',
          'learning_rate': 0.01,
          'max_depth': 3,
          'tree_method': 'hist',
          'min_child_weight': 8
          }

# Training model
nb = 500
evals_result = {}
watchlist = [(xgb_test, "test"), (xgb_train, "train")]
model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=nb, evals=watchlist,
                  verbose_eval=100, early_stopping_rounds=50, evals_result=evals_result)

# Watching performance for evaluating

#metric_name = list(evals_result['test'].keys())[0]  
metric_name = 'rmse'

train_score = evals_result['train'][metric_name][-1]
test_score = evals_result['test'][metric_name][-1]

# Best last result
log_data = {
    "best_iteration": model.best_iteration,
    "best_score": model.best_score,
    "train_score": round(train_score, 4),
    "test_score": round(test_score, 4),
    "features": ", ".join(X.columns.tolist()),
    "params": str(params)
}

# Saving to *.txt
with open('model_summary.txt', 'a', encoding='utf-8') as txt_file:
    for key, value in log_data.items():
        txt_file.write(f"{key}: {value}\n")

print(f"Średnia z czasu przejazdu {Y_train.mean()}")
print(f"Odcyhelnie standardowe: {Y_train.std()}")

y_pred = model.predict(xgb_test)


# Error Analysis - why mean is 0.25, rmse 0.26-0.28 and std 0.48
error =abs(Y_test - y_pred)
