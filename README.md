# 🚕🗽 NYC Taxi Route Optimization System 🚕🗽
*Project focused on urban logistics optimization through Machine Learning.*
This project is an advanced machine learning pipeline designed to analyze and optimize New York City "Yellow Taxi" routes. The system predicts trip duration (`time_difference`) by integrating raw trip data with weather conditions and geographical metadata.

## 🚀 Key Features

* **Automated Pipeline**: The `main.py` script monitors directories for new data files, automatically triggering data preparation, model training, and evaluation.
* **Data Integration**: Merges large-scale `.parquet` trip datasets with hourly weather data (`weather-data.csv`) and taxi zone lookups (`id_lookup.csv`).
* **Advanced Preprocessing**:
    * **Data Cleaning**: Removes system errors, such as negative fare amounts, congestion surcharge errors, and unrealistic travel times.
    * **Feature Engineering**: Generates predictive features including "minutes after midnight," trip distance in kilometers, and average speed.
    * **Intelligent Pruning**: Implements a downsampling strategy for the most common routes (e.g., the 236-237 Manhattan route) to prevent model bias and optimize memory usage.
* **XGBoost Modeling**: Utilizes a Gradient Boosting Regressor optimized with specific hyperparameters (`max_depth: 4`, `learning_rate: 0.02`) to minimize RMSE and MAE.
* **Error Analysis**: Automatically identifies and exports the top 800 largest prediction residuals (`top_800_errors.csv`) for further traffic anomaly investigation.

## 📁 Project Structure

* `data/`: Contains CSV and Parquet files (weather data, zone lookups, and trip datasets).
* `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and visualization.
    * `0_eda.ipynb`: In-depth analysis of trip patterns, errors, and route popularity.
* `src/`: Source code for the system:
    * `main.py`: The central orchestrator that manages the execution flow.
    * `data_prep.py`: Handles data cleaning, merging, and feature engineering.
    * `XGBoost_model.py`: Manages data splitting, model training, and performance logging.
    * `evaluation.py`: Generates metric graphs and evaluation reports.
* `road_model.ubjson`: The serialized XGBoost model ready for deployment.

## 🛠️ Installation & Setup

1.  **Requirements**: Python 3.8+
2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy xgboost scikit-learn matplotlib seaborn polars pyarrow
    ```
3.  **Path Configuration**: Update the `folder_path` variables in `src/main.py` and `src/data_prep.py` to point to your local dataset directory.

## 📈 Workflow Pipeline

The system follows a three-step automated process:

1.  **Preprocessing (`data_prep.py`)**: Transforms raw data into a structured format, merges external weather features, and balances the dataset.
2.  **Training (`XGBoost_model.py`)**: Executes an 80/20 train-test split and trains the model using early stopping to prevent overfitting.
3.  **Evaluation (`evaluation.py`)**: Produces performance summaries, logs metrics to `model_summary.json`, and generates visual reports.

## 📊 Results & Metrics

Model performance is tracked using several key metrics:
* **R² Score**: Measures how well the model explains the variance in trip durations.
* **RMSE (Root Mean Squared Error)**: Standard deviation of prediction errors.
* **MAE (Mean Absolute Error)**: Average magnitude of prediction errors.

Detailed training logs and history are maintained in `model_summary.txt` and `model_summary.json` for long-term analysis.

## 🔍 EDA Insights

Preliminary analysis conducted in the notebooks revealed:
* The route between zones **236 and 237** is among the most frequent, requiring specific sampling techniques to avoid overfitting.
* Temporal features, such as the exact minute of pickup, are critical indicators for predicting urban traffic flow.
* Data cleaning is essential, as raw datasets often contain negative values for `fare_amount` and `congestion_surcharge`.

---