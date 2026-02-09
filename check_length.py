import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings if needed
warnings.filterwarnings("ignore")

def check_smd_lengths():
    print("Checking SMD dataset lengths...")
    
    base_dir = "datasets/SMD"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found.")
        return

    # List all .txt files in train directory
    files = sorted([f for f in os.listdir(train_dir) if f.endswith('.txt')])
    
    if not files:
        print(f"No .txt files found in {train_dir}")
        return

    print(f"Found {len(files)} SMD datasets.")

    max_len_global = 0
    min_len_global = float('inf')
    files_shorter_than_512 = []

    for fname in files:
        # Check Train
        train_path = os.path.join(train_dir, fname)
        if os.path.exists(train_path):
            try:
                # Read using pandas (text file, assuming csv format)
                df = pd.read_csv(train_path)
                length = len(df)
                
                max_len_global = max(max_len_global, length)
                min_len_global = min(min_len_global, length)
                
                if length < 512:
                    files_shorter_than_512.append(f"TRAIN {fname}: {length}")
            except Exception as e:
                print(f"Error reading {train_path}: {e}")
        else:
            print(f"Warning: {train_path} does not exist.")

        # Check Test
        test_path = os.path.join(test_dir, fname)
        if os.path.exists(test_path):
            try:
                # Read using pandas
                df = pd.read_csv(test_path)
                length = len(df)
                
                max_len_global = max(max_len_global, length)
                min_len_global = min(min_len_global, length)
                
                if length < 512:
                    files_shorter_than_512.append(f"TEST  {fname}: {length}")
            except Exception as e:
                print(f"Error reading {test_path}: {e}")
        else:
            print(f"Warning: {test_path} does not exist.")

    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Global Maximum Length: {max_len_global}")
    print(f"Global Minimum Length: {min_len_global}")
    
    if files_shorter_than_512:
        print(f"\nFiles shorter than length 512 ({len(files_shorter_than_512)} files):")
        for f in files_shorter_than_512:
            print(f)
    else:
        print("\nNo files are shorter than length 512.")

def check_swat_lengths():
    print("\nChecking SWAT dataset lengths...")
    
    # Check in local datasets/SWAT folder first
    base_dir = "datasets/SWAT"
    # Fallback to Kaggle input if needed (though we expect them to be copied/available)
    kaggle_input_path = "/kaggle/input/swat-dataset-secure-water-treatment-system"
    
    files_to_check = ["normal.csv", "attack.csv"]
    files_shorter_than_512 = []
    
    for fname in files_to_check:
        # Try local first
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            # Try kaggle input
             path = os.path.join(kaggle_input_path, fname)
        
        if os.path.exists(path):
            try:
                # Read using pandas
                # Skipping first row header is default in pandas read_csv
                # We should trim spaces in column names just in case, but usually length check is fine
                df = pd.read_csv(path)
                length = len(df)
                
                print(f"File {fname}: length {length}")
                
                if length < 512:
                    files_shorter_than_512.append(f"SWAT {fname}: {length}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            print(f"Warning: {fname} does not exist in {base_dir} or {kaggle_input_path}")

    if files_shorter_than_512:
        print(f"\nFiles shorter than length 512 in SWAT ({len(files_shorter_than_512)} files):")
        for f in files_shorter_than_512:
            print(f)
    else:
        print("\nAll checked SWAT files are longer than 512.")

if __name__ == "__main__":
    check_smd_lengths()
    check_swat_lengths()