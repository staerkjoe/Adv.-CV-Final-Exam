from ultralytics import YOLO
import torch


class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.weights = config['model']['weights']
        self.variant = config['model']['variant']
        self.conf_threshold = config['model']['conf_threshold']
        self.iou_threshold = config['model']['iou_threshold']
        self.num_classes = config['model']['num_classes']
        
        # Auto-detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load YOLOv8 model based on configuration."""
        if self.weights:
            model_path = self.weights
        else:
            variant = self.variant
            model_path = f"yolov8{variant}.pt"
        
        model = YOLO(model_path)
        model.to(self.device)
        
        return model
    
