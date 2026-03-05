import pandas as pd
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path


def XGBoost_model(xgb_train, xgb_test, X):
    '''
    Training and testing model on defined parameters.
    '''

    # Specifying params for our trees
    params = {'objective': 'reg:squarederror',
            'learning_rate': 0.02,
            'max_depth': 4,
            'tree_method': 'hist',
            'min_child_weight': 8
            }

    # Training model
    nb = 700
    evals_result = {}
    watchlist = [(xgb_test, "test"), (xgb_train, "train")]
    model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=nb, evals=watchlist,
                    verbose_eval=100, early_stopping_rounds=50, evals_result=evals_result)

    model.save_model('road_model.ubjson')
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

    y_pred_test = model.predict(xgb_test)

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

    
    return xgb_train, xgb_test, X

def model_pipeline(df):
    '''
    Main orchestrator of data prepare, trainning and test process.
    '''
    xgb_train, xgb_test, X = data_prepare(df)
    model, predictions = XGBoost_model(xgb_train, xgb_test, X)

    return model, predictions


def main_process():
    '''
    Executes whole model training script.
    '''

    dataset_path = Path(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\preprocessed-dataset')

    if not dataset_path.exists():
        print('Error! Path does not exist.')
        return

    df = pd.read_parquet(dataset_path)
    model, y_pred = model_pipeline(df)
    print('Training ended successfully.')


if __name__ == "__main__":
    main_process()