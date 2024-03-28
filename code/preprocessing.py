import argparse
import pandas as pd
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--data_dir', type=str, help='Input data directory', default='/data')

args = parser.parse_args()

# Use the specified data directory or default to a specific path
INPUT_DATA_DIR = args.data_dir if args.data_dir else 'data'
books_merged = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'books_merged.csv'))
