import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class COCODataset(Dataset):
    """COCO Dataset using FiftyOne - returns images, labels, and bounding boxes."""
    
    def __init__(self, fo_dataset, transform=None, class_mapping=None, target_classes=None):
        self.fo_dataset = fo_dataset
        self.transform = transform
        self.samples = list(fo_dataset)
        self.class_mapping = class_mapping or {}
        self.target_classes = target_classes or []
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.filepath).convert('RGB')
        
        # Get detections
        detections = sample.ground_truth.detections if sample.ground_truth else []
        
        # Process labels and bounding boxes
        labels = []
        boxes = []
        
        for det in detections:
            label_name = det.label
            
            # Map to class index or 'other' category
            if label_name in self.target_classes:
                label_idx = self.class_mapping[label_name]
            else:
                label_idx = self.class_mapping['other']
            
            labels.append(label_idx)
            
            # Convert relative coordinates [x, y, width, height] to absolute [x1, y1, x2, y2]
            bbox = det.bounding_box
            img_width, img_height = image.size
            x1 = bbox[0] * img_width
            y1 = bbox[1] * img_height
            x2 = (bbox[0] + bbox[2]) * img_width
            y2 = (bbox[1] + bbox[3]) * img_height
            boxes.append([x1, y1, x2, y2])
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensors
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'boxes': boxes,
            'image_id': idx,
            'num_detections': len(labels)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length detections."""
    images = torch.stack([item['image'] for item in batch])
    
    # Keep labels and boxes as lists since they have variable lengths
    labels = [item['labels'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch])
    num_detections = torch.tensor([item['num_detections'] for item in batch])
    
    return {
        'images': images,
        'labels': labels,
        'boxes': boxes,
        'image_ids': image_ids,
        'num_detections': num_detections
    }


class COCODataLoader:
    """DataLoader manager for COCO dataset."""
    
    def __init__(self, config):
        self.config = config
        self.dataset_config = config['dataset']
        self.training_config = config['training']
        self.validation_config = config['validation']
        self.class_mapping = self._create_class_mapping()
        
    def _create_class_mapping(self):
        """Create mapping from class names to indices."""
        target_classes = self.dataset_config.get('classes', [])
        class_mapping = {cls: idx for idx, cls in enumerate(target_classes)}
        # Add 'other' category
        class_mapping['other'] = len(target_classes)
        return class_mapping
    
    def get_class_mapping(self):
        """Return the class mapping dictionary."""
        return self.class_mapping
    
    def get_num_classes(self):
        """Return total number of classes including 'other'."""
        return len(self.class_mapping)
    
    def _get_transforms(self):
        """Create simple transforms."""
        img_size = self.dataset_config['image_size']
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_fiftyone_dataset(self, split='train'):
        """Load COCO dataset using FiftyOne."""
        dataset_name = f"{self.dataset_config['name']}-{split}"
        
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
        else:
            dataset = foz.load_zoo_dataset(
                self.dataset_config['name'],
                split=split,
                dataset_name=dataset_name,
                max_samples=self.dataset_config['max_samples']
            )
        
        # Filter by classes if specified in config
        target_classes = self.dataset_config.get('classes', None)
        if target_classes:
            # Get samples with target classes
            class_view = dataset.filter_labels(
                "ground_truth",
                F("label").is_in(target_classes)
            ).match(F("ground_truth.detections").length() > 0)
            
            # Get samples without target classes (for "other" category)
            other_view = dataset.filter_labels(
                "ground_truth",
                ~F("label").is_in(target_classes)
            ).match(F("ground_truth.detections").length() > 0)
            
            # Take limited samples from each
            other_ratio = self.dataset_config['other_ratio']
            
            class_samples = int(self.dataset_config['max_samples'] * (1 - other_ratio))
            other_samples = int(self.dataset_config['max_samples'] * other_ratio)
            
            dataset = class_view.take(class_samples).concat(other_view.take(other_samples))
        
        return dataset
    
    def get_train_loader(self):
        """Get training data loader."""
        fo_dataset = self._load_fiftyone_dataset(split='train')
        target_classes = self.dataset_config.get('classes', [])
        dataset = COCODataset(
            fo_dataset, 
            transform=self._get_transforms(),
            class_mapping=self.class_mapping,
            target_classes=target_classes
        )
        
        return DataLoader(
            dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=self.training_config['shuffle'],
            num_workers=self.training_config['num_workers'],
            pin_memory=self.training_config['pin_memory'] and torch.cuda.is_available(),
            drop_last=self.training_config['drop_last'],
            collate_fn=collate_fn
        )
    
    def get_val_loader(self):
        """Get validation data loader."""
        fo_dataset = self._load_fiftyone_dataset(split='validation')
        target_classes = self.dataset_config.get('classes', [])
        dataset = COCODataset(
            fo_dataset, 
            transform=self._get_transforms(),
            class_mapping=self.class_mapping,
            target_classes=target_classes
        )
        
        return DataLoader(
            dataset,
            batch_size=self.validation_config['batch_size'],
            shuffle=self.validation_config['shuffle'],
            num_workers=self.validation_config['num_workers'],
            pin_memory=self.validation_config['pin_memory'] and torch.cuda.is_available(),
            collate_fn=collate_fn
        )
