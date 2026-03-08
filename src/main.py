import subprocess
import sys
from pathlib import Path



def run_pipeline():
    '''
    Pipeline of the data preperation, ML training and then evaluation by generating important metric graphs.
    '''
    # Data preparation (check if there's new file)
    # If there is - start data preparation script with this new file
    folder_path = Path(r"C:\Users\wikto\OneDrive\Dokumenty\all-datasets\taxi_routes_datasets_23-25")
    history_file = Path("processed_files.txt")

    if not folder_path.exists():
        print('Error! No folder path.')
        return
    
    known_files = set()
    if history_file.exists():
        known_files = set(history_file.read_text().splitlines())

    # Checks what's new
    current_files = {f.name for f in folder_path.iterdir() if f.is_file()}
    new_files = current_files - known_files

    if new_files:
            print(f"Founded {len(new_files)} new files. Starting data_prep.py...")
            try:
                subprocess.run([sys.executable, "data_prep.py"], check=True)
            except subprocess.CalledProcessError:
                print("Error accured in data_prep.py... Stopping pipeline")
                return

    # Train
    try:
        subprocess.run([sys.executable, "XGBoost_model.py"], check=True)
    except subprocess.CalledProcessError:
         print('Error accured in XGBoost_model.py... Stopping pipeline')

    # Generate PDF
    try:
        subprocess.run([sys.executable, "evaluation.py"], check=True)
    except subprocess.CalledProcessError:
         print('Error accured in evaluation.py... Stopping pipeline')

if __name__ == "__main__":
    run_pipeline()