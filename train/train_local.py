from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np
import os
import sys
import yaml
from pprint import pprint
import torch
import torch.nn as nn

# Add paths to Python path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core import data
from core import models
from core.metrics.eval_detr_metrics import eval_detr_metrics
from core.trainer import TrainerDetr
from core.local_experiment import ExperimentWrapper

def get_values_from_args():
    """Command line arguments to control the run"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./configs/train.yaml')
    parser.add_argument('--test-only', '-t', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config, args

if __name__ == '__main__':
    from pprint import pprint 
    import gc
    
    # Clear CUDA cache before processing
    if torch.cuda.is_available():
        pprint(f"Clearing CUDA cache...")
        torch.cuda.empty_cache()
        gc.collect()
    
    np.set_printoptions(precision=4, suppress=True)
    config, args = get_values_from_args()
    
    # Create experiment
    experiment = ExperimentWrapper(config)
    
    # Initialize experiment
    experiment.init_run(config)
    
    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(config['dataset']['train_dir'], config['dataset']["sim_root"], config['dataset'], 
                         gt_caching=True, feature_caching=False)

    # Create trainer
    trainer = TrainerDetr(
        config['trainer'], experiment, dataset, config['data_split'], 
        with_norm=True, 
        # with_visualization=config['trainer']['with_visualization']
    )
    
    trainer.init_randomizer()

    # --- Model ---
    model, criterion = models.build_model(config)
    model_without_ddp = model
    
    # Set up device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    criterion.to(device)
    
    # Wrap model with DataParallel if using GPU
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0])
    
    # Load pre-trained model if specified
    if config["NN"]["step-trained"] is not None and os.path.exists(config["NN"]["step-trained"]):
        model.load_state_dict(torch.load(config["NN"]["step-trained"], map_location=device)["model_state_dict"])
        print("Train::Info::Load Pre-step-trained model: {}".format(config["NN"]["step-trained"]))
    
    # Print model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Train::Info::Number of params: {n_parameters}')

    # Training or testing
    if not args.test_only:    
        trainer.fit(model, model_without_ddp, criterion, device, config)
    else:
        config["loss"]["lepoch"] = -1
        if config["NN"]["pre-trained"] is None or not os.path.exists(config["NN"]["pre-trained"]):
            print("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    model.load_state_dict(experiment.get_best_model()['model_state_dict'])
    datawrapper = trainer.datawraper
    
    # Evaluate on validation set
    final_metrics = eval_detr_metrics(model, criterion, datawrapper, device, 'validation')
    experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
    pprint(final_metrics)
    
    # Evaluate on test set
    final_metrics = eval_detr_metrics(model, criterion, datawrapper, device, 'test')
    experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
    pprint(final_metrics)
    
    # Stop experiment
    experiment.stop()
    
    print("Training completed successfully!")
