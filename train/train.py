from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
from pprint import pprint
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import data
import models
from core.metrics.eval_detr_metrics import eval_detr_metrics
from core.trainer import TrainerDetr
from core.local_experiment import ExperimentWrapper


def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./configs/train.yaml')
    parser.add_argument('--test-only', '-t',  action='store_true', default=False)
    parser.add_argument('--local_rank', default=0)
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

    # DDP
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    print(f"INFO::{__file__}::Start running basic DDP example on rank {rank}.")
    config['trainer']['multiprocess'] = True

    experiment = ExperimentWrapper(
        config,
        no_sync=False) 
    
    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(config['dataset']['train_dir'], config['dataset']['sim_root'], config['dataset'], gt_caching=True, feature_caching=False)

    trainer = TrainerDetr(
            config['trainer'], experiment, dataset, config['data_split'], 
            with_norm=True, with_visualization=config['trainer']['with_visualization'])  # only turn on visuals on custom garment data
    trainer.init_randomizer()

    # --- Model ---
    model, criterion = models.build_model(config)
    model_without_ddp = model
    # DDP
    torch.cuda.set_device(rank)
    model.cuda(rank)
    criterion.cuda(rank)

    # Wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if config["NN"]["step-trained"] is not None and os.path.exists(config["NN"]["step-trained"]):
        model.load_state_dict(torch.load(config["NN"]["step-trained"], map_location="cuda:{}".format(rank))["model_state_dict"])
        print("Train::Info::Load Pre-step-trained model: {}".format(config["NN"]["step-trained"]))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Train::Info::Number of params: {n_parameters}')

    if not args.test_only:    
        trainer.fit(model, model_without_ddp, criterion, rank, config)
    else:
        config["loss"]["lepoch"] = -1
        if config["NN"]["pre-trained"] is None or not os.path.exists(config["NN"]["pre-trained"]):
            print("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    if rank == 0:
        model.load_state_dict(experiment.get_best_model()['model_state_dict'])
        datawrapper = trainer.datawraper
        final_metrics = eval_detr_metrics(model, criterion, datawrapper, rank, 'validation')
        experiment.add_statistic('valid_on_best', final_metrics, log='Validation metrics')
        pprint(final_metrics)
        final_metrics = eval_detr_metrics(model, criterion, datawrapper, rank, 'test')
        experiment.add_statistic('test_on_best', final_metrics, log='Test metrics')
        pprint(final_metrics)
        
