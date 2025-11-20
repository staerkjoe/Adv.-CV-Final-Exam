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
    
    def get_inference_params(self):
        """Get inference parameters from config."""
        return {
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'device': self.device
        }
    
    def get_raw_model(self, yolo_wrapper):
        """Return the internal torch model from the YOLO wrapper."""
        return getattr(yolo_wrapper, 'model', yolo_wrapper)