from ultralytics import YOLO
import torch


class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        
        # Auto-detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load YOLOv8 model based on configuration."""
        if self.model_config['weights']:
            model_path = self.model_config['weights']
        else:
            variant = self.model_config['variant']
            model_path = f"yolov8{variant}.pt"
        
        model = YOLO(model_path)
        model.to(self.device)
        
        return model
    
    def get_inference_params(self):
        """Get inference parameters from config."""
        return {
            'conf': self.model_config['conf_threshold'],
            'iou': self.model_config['iou_threshold'],
            'device': self.device
        }
