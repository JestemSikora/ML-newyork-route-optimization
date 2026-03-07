import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb

# Class for all variables from .json
class ModelResults:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def plot_metrics_graphs(self):
        """
        Generates MAE and RMSE graphs.
        """
        plt.figure(figsize=(12, 5))

        # --- Graph RMSE ---
        plt.subplot(1, 2, 1) # 2 row, 2 columns, first graph
        
        # Data prepare
        train_iters = sorted([int(k) for k in self.train_rmse_history.keys()])
        train_values = [self.train_rmse_history[str(i)] for i in train_iters]
        test_values = [self.test_rmse_history[str(i)] for i in train_iters]

        plt.plot(train_iters, train_values, label="Train RMSE", marker='o', linestyle='--')
        plt.plot(train_iters, test_values, label="Test RMSE", marker='s')
        
        plt.title(f"RMSE Over Iterations")
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- Graph MAE ---
        plt.subplot(2, 2, 1) # Second graph
        
        mae_train_values = [self.train_mae_history[str(i)] for i in train_iters]
        mae_test_values = [self.test_mae_history[str(i)] for i in train_iters]

        plt.plot(train_iters, mae_train_values, label="Train MAE", color='orange', marker='o', linestyle='--')
        plt.plot(train_iters, mae_test_values, label="Test MAE", color='red', marker='s')

        plt.title("MAE Over Iterations")
        plt.xlabel('Iteration')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    def plot_feature_importance(self, model):
        plt.tight_layout()
        xgb.plot_importance(model,importance_type='gain', max_num_features=27, title='Top 10 Cech (Gain)' )
        plt.show()


def evaluation_function():
    '''
    '''
    # Open saved model_summary.json
    with open('model_summary.json', 'r') as f:
        data = json.load(f)

    model = xgb.Booster()
    model.load_model('road_model.ubjson')

    # Class definition
    results = ModelResults(data[-1])

    # Read preprocessed dataset .paraquet
    df = pd.read_parquet(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\test-preprocessed-dataset')

    #results.plot_metrics_graphs(model)
    results.plot_feature_importance(model)

if __name__ == "__main__":
    evaluation_function()