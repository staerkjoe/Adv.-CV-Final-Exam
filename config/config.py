import os

# Dataset configuration
DATASET_NAME = "mmoreaux/environmental-sound-classification-50"
CSV_FILE = "esc50.csv"  # Adjust based on actual CSV filename in the dataset
AUDIO_DIR = "audio"  # Adjust based on actual audio directory name in the dataset

# Column names in CSV
FILENAME_COLUMN = "filename"  # Adjust based on actual column name
CAPTION_COLUMN = "category"  # Adjust based on actual column name (might be 'category' or 'target')

# DataLoader configuration
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 4

# Paths will be set after downloading
DATASET_PATH = None
