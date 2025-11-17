import os
import cv2
import yaml
from dotenv import load_dotenv

class DataLoader:
    def __init__(self, config):
        load_dotenv()

        from roboflow import Roboflow
        rf = Roboflow(api_key=os.getenv('api_key'))
        project = rf.workspace(config['robowflow']['workspace']).project(config['robowflow']['project'])
        self.version = project.version(config['robowflow']['version'])
        self.dataset = None
        self.config = config
        self.dataset_location = None
        self.download_path = config['data']['download_path']
        
    def download_dataset(self, format='yolov8'):
        """Download the dataset from Roboflow"""
        self.dataset = self.version.download(format)
        self.dataset_location = self.dataset.location
        return self.dataset
    
    def load_images(self):
        """
        Load images from all splits
        
        Returns:
            tuple: (train_data, valid_data, test_data)
                   Each is a list of dicts: [{image_path, label_path, image, filename}, ...]
        """
        if self.dataset is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")
        
        train_data = self._load_split('train')
        valid_data = self._load_split('valid')
        test_data = self._load_split('test')
        
        return train_data, valid_data, test_data
    
    def _load_split(self, split):
        """
        Helper function to load a single split
        
        Args:
            split: 'train', 'valid', or 'test'
            
        Returns:
            list of dicts: [{image_path, label_path, image, filename}, ...]
        """
        images_path = os.path.join(self.dataset_location, split, "images")
        labels_path = os.path.join(self.dataset_location, split, "labels")
        
        if not os.path.exists(images_path):
            print(f"Warning: Split '{split}' not found in dataset, returning empty list")
            return []
        
        image_files = os.listdir(images_path)
        loaded_data = []
        
        for img_file in image_files:
            img_path = os.path.join(images_path, img_file)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(labels_path, label_file)
            
            # Load image
            img = cv2.imread(img_path)
            
            loaded_data.append({
                'image_path': img_path,
                'label_path': label_path if os.path.exists(label_path) else None,
                'image': img,
                'filename': img_file
            })
        
        return loaded_data
    
    def get_class_names(self):
        """Get class names from data.yaml"""
        if self.dataset_location is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")
        
        yaml_path = os.path.join(self.dataset_location, "data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data.get('names', [])
    
    def get_dataset_info(self):
        """Get dataset information"""
        if self.dataset_location is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")
        
        yaml_path = os.path.join(self.dataset_location, "data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return {
            'location': self.dataset_location,
            'classes': data.get('names', []),
            'num_classes': data.get('nc', 0),
            'train_path': data.get('train', ''),
            'val_path': data.get('val', ''),
            'test_path': data.get('test', '')
        }