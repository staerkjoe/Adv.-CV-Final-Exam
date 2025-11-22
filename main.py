from xml.parsers.expat import model
import yaml
from pathlib import Path
import argparse
import torch
from data.data_loader import DataLoader
from ultralytics import YOLO
import wandb

def load_config(config_path='config/config.yaml'):
    """
    Load YAML configuration file
    
    This is the ONLY place we load config in the entire project.
    Config is then passed to all other modules.
    """
    # Always resolve path relative to this file's location (main.py)
    main_dir = Path(__file__).parent
    config_path = main_dir / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")
    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train and compare LLM models on classification task'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    # Training overrides (CLI)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--imgsz', type=int, help='Image size (square)')
    parser.add_argument('--lr', type=float, help='Initial learning rate (lr0)')
    parser.add_argument('--project', type=str, help='YOLO project folder to save runs')
    parser.add_argument('--name', type=str, help='YOLO run name')
    parser.add_argument('--sample', action='store_true', help='Use a small sample of training data')
    parser.add_argument('--device', type=str, help='Device to use for training (e.g., "cpu", "cuda:0")')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size when --sample is used')
    args = parser.parse_args()
    config = load_config(args.config)

    # Set device (default)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device or config.get("training", {}).get("device", default_device)
    print(f"Using device: {device}")

    # initialize data loader
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    data_loader = DataLoader(config)
    try:
        train_df, valid_df, test_df = data_loader.load_images()
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return

    # Use sample if requested
    import random
    if args.sample:
        print(f"\nUsing sample of {args.sample_size} rows for testing")
        k = min(args.sample_size, len(train_df))
        train_df = random.sample(train_df, k)

    # Prepare model and training parameters
    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)

    # allow data YAML path in config (fallback to hardcoded path if absent)
    data_config_path ='C:\\Users\\Besitzer\\OneDrive\\Dokumente\\CBS_Copenhagen\\Semester\\WS2025\\AdvCV\\Final Exam\\Adv.-CV-Final-Exam\\scripts\\PlayingCardDetection-1\\data.yaml'


    # model checkpoint (can be in config)
    model = YOLO("yolov8s.pt")

    # Build train params by merging config defaults and CLI overrides
    train_defaults = config.get("training", {})
    train_params = {
        "data": data_config_path,
        "epochs": args.epochs if args.epochs is not None else train_defaults.get("epochs", 50),
        "batch": args.batch if args.batch is not None else train_defaults.get("batch", None),
        "imgsz": args.imgsz if args.imgsz is not None else train_defaults.get("imgsz", None),
        "lr0": args.lr if args.lr is not None else train_defaults.get("lr0", None),
        "device": device,
        "project": args.project or train_defaults.get("project", "runs/train"),
        "name": args.name or train_defaults.get("name", "exp"),
        'freeze': train_defaults.get("freeze", 10),
        # add other keys you want to expose here; ultralytics accepts many kwargs
    }

    # Remove None values so YOLO.train uses its defaults for unspecified params
    train_params = {k: v for k, v in train_params.items() if v is not None}

    print("Training with parameters:", train_params)

    # Start training
    try:
        model.train(**train_params)
    except Exception as e:
        print("Training failed:", e)

if __name__ == "__main__":
    main()
