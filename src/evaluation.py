import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages

# Class for all variables from .json
class ModelResults:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def plot_metrics_graphs(self):
            """
            Generates MAE and RMSE graphs on one page (two subplots).
            """
            fig = plt.figure(figsize=(12, 5))

            # --- Graph RMSE ---
            plt.subplot(1, 2, 1) 
            
            train_iters = sorted([int(k) for k in self.train_rmse_history.keys()])
            train_values = [self.train_rmse_history[str(i)] for i in train_iters]
            test_values = [self.test_rmse_history[str(i)] for i in train_iters]

            plt.plot(train_iters, train_values, label="Train RMSE", marker='o', linestyle='--')
            plt.plot(train_iters, test_values, label="Test RMSE", marker='s')
            
            plt.title("RMSE Over Iterations")
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # --- Graph MAE ---
            plt.subplot(1, 2, 2)
            
            mae_train_values = [self.train_mae_history[str(i)] for i in train_iters]
            mae_test_values = [self.test_mae_history[str(i)] for i in train_iters]

            plt.plot(train_iters, mae_train_values, label="Train MAE", color='orange', marker='o', linestyle='--')
            plt.plot(train_iters, mae_test_values, label="Test MAE", color='red', marker='s')

            plt.title("MAE Over Iterations")
            plt.xlabel('Iteration')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig 

        
    def plot_feature_importance(self, model):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

            plt.subplot(1, 3, 1)
            xgb.plot_importance(model, importance_type='gain', max_num_features=27, 
                            title='Top Features (Gain)', ax=ax1)
            
            plt.subplot(1, 3, 2)
            xgb.plot_importance(model, importance_type='cover', max_num_features=27, 
                            title='Top Features (Cover)', ax=ax2)
            
            plt.subplot(1, 3, 3)
            xgb.plot_importance(model, importance_type='weight', max_num_features=27, 
                            title='Top Features (Weight)', ax=ax3)
            
            ax2.set_ylabel('')
            ax3.set_ylabel('')
            
            plt.tight_layout()
            return fig


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

    results.plot_metrics_graphs()
    results.plot_feature_importance(model)

    with PdfPages('evaluation_graphs.pdf') as pdf:
        
        # 1. Graphs of metrics
        fig1 = results.plot_metrics_graphs() 
        pdf.savefig(fig1) 
        plt.close(fig1)    

        # 2. Feature Importance
        fig2 = results.plot_feature_importance(model)
        pdf.savefig(fig2) 
        plt.close(fig2)


if __name__ == "__main__":
    evaluation_function()