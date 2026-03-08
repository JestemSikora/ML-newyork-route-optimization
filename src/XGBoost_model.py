import pandas as pd
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import r2_score
import json
import os


def XGBoost_model(xgb_train, xgb_test, X, Y_test):
    '''
    Training and testing model on defined parameters.
    '''

    # Specify params for xgboost trees
    params = {'objective': 'reg:squarederror',
            'learning_rate': 0.02,
            'max_depth': 4,
            'tree_method': 'hist',
            'min_child_weight': 8,
            'eval_metric': ['rmse', 'mae']
            }

    # Train model
    nb = 700
    evals_result = {}
    watchlist = [(xgb_test, "test"), (xgb_train, "train")]
    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=nb, evals=watchlist,
                    verbose_eval=100, early_stopping_rounds=50, evals_result=evals_result)

    y_pred_test = model.predict(xgb_test)

    model.save_model('road_model.ubjson')

    ### Watch performance for evaluating  ###

    # Variables for RMSE
    train_rmse = evals_result['train']['rmse'][-1]
    test_rmse = evals_result['test']['rmse'][-1]

    train_rmse_list = evals_result['train']['rmse'][::50]
    test_rmse_list = evals_result['test']['rmse'][::50]

    # Variables for MAE
    train_mae = evals_result['train']['mae'][-1]
    test_mae = evals_result['test']['mae'][-1]

    train_mae_list = evals_result['train']['mae'][::50]
    test_mae_list = evals_result['test']['mae'][::50]

    # Variables for R^2
    y_test_true = xgb_test.get_label()
    test_r2 = r2_score(y_test_true, y_pred_test)

    # Largest residuals
    residuals_series = pd.Series(
        np.round(Y_test.values - y_pred_test, 4), 
        index=Y_test.index, 
        name='residuals in h'
    )
    
    # To DataFrame
    residuals_test = residuals_series.to_frame()
    residuals_test['true_value'] = Y_test.values
    residuals_test['y_predictions'] = y_pred_test
    
    # Top 800 largest residuals
    largest_residuals = residuals_test.nlargest(800, 'residuals in h')

    # Save to csv for later analysis
    largest_residuals.to_csv('top_800_errors.csv')

    # How many total rounds
    total_rounds = len(evals_result['train']['rmse'])

    # Step every 50 rounds
    steps = list(range(0, total_rounds, 50))

    if (total_rounds - 1) not in steps:
        steps.append(total_rounds - 1)

    # Log data for later analysis
    log_data = {
        "best_iteration": model.best_iteration,
        "features": ", ".join(X.columns.tolist()),
        "params": str(params),
        "best_score": model.best_score,
        "train_rmse": round(train_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "train_rmse_history": {step: round(evals_result['train']['rmse'][step], 6) for step in steps},
        "test_rmse_history": {step: round(evals_result['test']['rmse'][step], 6) for step in steps},
        "train_mae": round(train_mae, 4),
        "test_mae": round(test_mae, 4),
        "train_mae_history": {step: round(evals_result['train']['mae'][step], 6) for step in steps},
        "test_mae_history": {step: round(evals_result['test']['mae'][step], 6) for step in steps},
        "test_r2": round(test_r2, 4)
    }

    # Save to *.txt
    with open('model_summary.txt', 'a', encoding='utf-8') as txt_file:
        for key, value in log_data.items():
            txt_file.write(f"{key}: {value}\n")
        txt_file.write('-'*30)


    # Save to *.json
    json_file = 'model_summary.json'

    # Checks if file already exisists - if it is true, adds new data
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = [] # protection against an empty file
    else:
        all_results = [] # if there's no file make a new list

    all_results.append(log_data)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    
    return model, y_pred_test

def data_prepare(df):
    '''
    Prepares data for XGBoost model.
    '''
     # Changing datatype to category to be sure it's not 'datetime64[us]'
    for i in df:
        if df[i].dtype == 'object':
            df[i] = df[i].astype('category')

    # Train & Test variables
    Y = df['time_diffrence h']
    X = df.drop(columns=['time_diffrence h'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)

    # Converting dataset into DMatrix structure
    xgb_train = xgb.DMatrix(X_train, Y_train, enable_categorical=True)
    xgb_test = xgb.DMatrix(X_test, Y_test, enable_categorical=True)

    
    return xgb_train, xgb_test, X, Y_test

def model_pipeline(df):
    '''
    Main orchestrator of data prepare, trainning and test process.
    '''
    xgb_train, xgb_test, X , Y_test = data_prepare(df)
    model, predictions = XGBoost_model(xgb_train, xgb_test, X, Y_test)

    return model, predictions


def main_process():
    '''
    Executes whole model training script.
    '''

    dataset_path = Path(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\test-preprocessed-dataset')

    if not dataset_path.exists():
        print('Error! Path does not exist.')
        return

    df = pd.read_parquet(dataset_path)
    model, y_pred = model_pipeline(df)
    print('Training ended successfully.')


if __name__ == "__main__":
    main_process()