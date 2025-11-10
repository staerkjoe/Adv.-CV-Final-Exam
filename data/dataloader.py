import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class AudioCaptionDataset(Dataset):
    def __init__(self, subset_size=None):
        """
        Load audio-caption dataset using configuration.
        
        Args:
            subset_size: If provided, only use the first N samples (for local testing)
        """
        # Download dataset if not already downloaded
        dataset_path = kagglehub.dataset_download(config.DATASET_NAME)
        
        # Load CSV with captions
        csv_path = os.path.join(dataset_path, config.CSV_FILE)
        self.df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            config.DATASET_NAME,
            config.CSV_FILE
        )
        
        # Apply subset if specified
        if subset_size is not None:
            self.df = self.df.head(subset_size)
            print(f"Using subset of {len(self.df)} samples")
        
        self.audio_dir = os.path.join(dataset_path, config.AUDIO_DIR)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row[config.FILENAME_COLUMN])
        caption = row[config.CAPTION_COLUMN]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        return waveform, sample_rate, caption


def get_dataloader(batch_size=None, shuffle=None, num_workers=None, subset_size=None):
    """
    Create a DataLoader for the audio-caption dataset.
    
    Args:
        batch_size: Batch size (uses config default if None)
        shuffle: Whether to shuffle data (uses config default if None)
        num_workers: Number of workers (uses config default if None)
        subset_size: If provided, only use first N samples (for local testing)
    
    Returns:
        DataLoader instance
    """
    batch_size = batch_size or config.BATCH_SIZE
    shuffle = shuffle if shuffle is not None else config.SHUFFLE
    num_workers = num_workers or config.NUM_WORKERS
    
    dataset = AudioCaptionDataset(subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
