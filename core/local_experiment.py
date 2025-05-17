import os
from pathlib import Path
import json
import yaml
import time
import shutil

import torch
from torch import nn

from . import data
from . import models


class ExperimentWrapper(object):
    """Class provides 
        * a convenient way to store & load experiment info locally
        * functions for saving and loading model checkpoints
        * tracking of experiment metrics
    """
    def __init__(self, config, no_sync=False):
        """Initialize experiment tracking with local storage"""
        self.checkpoint_filetag = 'checkpoint'
        self.final_filetag = 'fin_model_state'
        
        if 'training' in config:
            # New config format
            self.project = config['training'].get('project_name', 'PanelFormer')
            self.run_name = config['training'].get('run_name', 'Default')
            self.run_id = config['training'].get('run_id', self._generate_run_id())
            self.run_local_path = config['training'].get('checkpoint_dir', 'outputs/checkpoints')
            is_training = config['training'].get('is_training', False)
        else:
            # Default values
            self.project = "PanelFormer"
            self.run_name = "Default"
            self.run_id = self._generate_run_id()
            self.run_local_path = "outputs/checkpoints"
            is_training = False
            
        if is_training and self.run_local_path is not None:
            os.makedirs(self.run_local_path, exist_ok=True)

        self.in_config = config
        self.summary = {}
        self.config = {}
        self.initialized = False
        self.checkpoint_counter = 0
        self.best_valid_loss = float('inf')
        
        # Create output directories
        self._create_directories()
        
        # Save initial config
        self._save_config()
    
    def _generate_run_id(self):
        """Generate a unique run ID based on timestamp"""
        return f"run_{int(time.time())}"
    
    def _create_directories(self):
        """Create necessary directories for experiment"""
        # Main experiment directory
        os.makedirs(self.run_local_path, exist_ok=True)
        
        # Checkpoints directory
        self.checkpoint_dir = os.path.join(self.run_local_path, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logs directory
        self.logs_dir = os.path.join(self.run_local_path, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Artifacts directory
        # self.artifacts_dir = os.path.join(self.run_local_path, 'artifacts')
        # os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def _save_config(self):
        """Save configuration to file"""
        config_path = os.path.join(self.run_local_path, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.in_config, f)
    
    # ----- start&stop ------
    def init_run(self, config={}):
        """Start experiment logging"""
        self.config.update(config)
        self.initialized = True
        self.checkpoint_counter = 0
        
        # Save updated config
        self._save_config()
        
        print(f'Initialized Training: {self.run_name} (ID: {self.run_id})')
    
    def stop(self):
        """Stop experiment logging"""
        if self.initialized:
            # Save final summary
            self._save_summary()
            print(f'{self.__class__.__name__}::Stopped experiment: {self.run_name} (ID: {self.run_id})')
        self.initialized = False

    # -------- run info ------
    def full_name(self):
        """Get the full name of the experiment"""
        name = self.project if self.project else ''
        name += ('-' + self.run_name) if self.run_name else ''
        if self.run_id:
            name += ('-' + self.run_id)
        else:
            name += self.in_config['NN']['pre-trained'] if 'pre-trained' in self.in_config['NN'] else ''
        
        return name
    
    def last_epoch(self):
        """Get the last epoch processed"""
        return self.summary.get('epoch', -1)
    
    def data_info(self):
        """Get data configuration information"""
        config = self._run_config()
        
        # Handle both old and new config formats
        if 'dataset' in config and 'data_split' in config:
            # Old config format
            split_config = config['data_split']
            data_config = config['dataset']
            batch_size = config['trainer']['batch_size'] if 'trainer' in config else config.get('batch_size', 32)
        elif 'data' in config:
            # New config format
            data_config = config['data']
            split_config = data_config.get('split', {})
            batch_size = data_config.get('batch_size', 32)
        else:
            # Fallback to defaults
            print(f'{self.__class__.__name__}::Warning::No data configuration found in config')
            split_config = {}
            data_config = {}
            batch_size = 32
        
        # This part might be missing from the configs
        if 'unseen_data_folders' not in data_config:
            if 'dataset' in self.in_config and 'unseen_data_folders' in self.in_config['dataset']:
                data_config['unseen_data_folders'] = self.in_config['dataset']['unseen_data_folders']
            elif 'data' in self.in_config:
                data_config['unseen_data_folders'] = self.in_config['data'].get('unseen_data_folders', [])
        
        return split_config, batch_size, data_config
    
    def last_best_validation_loss(self):
        """Get the best validation loss"""
        return self.best_valid_loss if self.best_valid_loss != float('inf') else None

    def NN_config(self):
        """Get Neural Network model configuration"""
        config = self._run_config()
        
        # Handle both old and new config formats
        if 'NN' in config:
            # Old config format
            return config['NN']
        elif 'model' in config:
            # New config format - convert to old format for compatibility
            model_config = config['model']
            nn_config = {
                'model': model_config.get('name', 'PanelFormer'),
                'pre-trained': config.get('inference', {}).get('checkpoint_path'),
                'step-trained': config.get('inference', {}).get('step_trained'),
                'backbone': model_config.get('backbone', 'resnet50'),
                'in_channel': model_config.get('in_channel', 3),
                'dilation': model_config.get('dilation', True),
                'position_embedding': model_config.get('position_embedding', 'sine'),
                'enc_layers': model_config.get('num_encoder_layers', 6),
                'dec_layers': model_config.get('num_decoder_layers', 6),
                'dim_feedforward': model_config.get('dim_feedforward', 2048),
                'hidden_dim': model_config.get('d_model', 256),
                'dropout': model_config.get('dropout', 0.1),
                'nheads': model_config.get('nhead', 8),
                'num_queries': model_config.get('num_queries', 25),
                'pre_norm': model_config.get('pre_norm', True),
                'aux_loss': model_config.get('aux_loss', True),
                'frozen_weights': model_config.get('frozen_weights'),
                'masks': model_config.get('masks'),
                'max_num_edges': model_config.get('num_edges', 56)
            }
            
            # Add loss configuration if available
            if 'loss' in config:
                loss_config = config['loss']
                nn_config['loss'] = {
                    'loss_components': loss_config.get('components', ['shape', 'loop', 'rotation', 'translation']),
                    'quality_components': loss_config.get('quality_components', ['shape', 'discrete', 'rotation', 'translation']),
                    'loss_weight_dict': {
                        'loop_loss_weight': loss_config.get('loop_weight', 1.0),
                        'edge_loss_weight': loss_config.get('edge_weight', 1.0),
                        'rotation_loss_weight': loss_config.get('rotation_weight', 1.0),
                        'translation_loss_weight': loss_config.get('translation_weight', 1.0)
                    },
                    'stitches': loss_config.get('stitches', 'ce'),
                    'lepoch': loss_config.get('lepoch', 0),
                    'eos_coef': loss_config.get('eos_coef', 0.1),
                    'aux_loss': loss_config.get('aux_loss', True),
                    'panel_origin_invariant_loss': loss_config.get('panel_origin_invariant_loss', False),
                    'panel_order_inariant_loss': loss_config.get('panel_order_inariant_loss', False),
                    'epoch_with_order_matching': loss_config.get('epoch_with_order_matching', 0),
                    'order_by': loss_config.get('order_by', 'shape_translation')
                }
            
            return nn_config
        else:
            # Return empty config if neither format is found
            print(f'Warning::No NN or model configuration found in config')
            return {}
    
    def add_statistic(self, tag, info, log=''):
        """Add info to the run summary"""
        # Log
        if log:
            print(f'Saving statistic <{log}>:')
            print(json.dumps(info, sort_keys=True, indent=2) if isinstance(info, dict) else info)

        # Update summary
        self.summary[tag] = info
        
        # Save summary to file
        self._save_summary()
    
    def _save_summary(self):
        """Save summary to file"""
        summary_path = os.path.join(self.run_local_path, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)
    
    def add_config(self, tag, info):
        """Add new value to run config"""
        self.config[tag] = info
        
        # Save updated config
        config_path = os.path.join(self.run_local_path, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def add_artifact(self, path, name, type):
        """Copy artifact to experiment directory"""
        path = Path(path)
        artifact_dir = os.path.join(self.artifacts_dir, name)
        
        # Create artifact directory
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Copy files
        if path.is_file():
            shutil.copy2(str(path), artifact_dir)
        else:
            # Copy directory contents
            for item in path.glob('*'):
                if item.is_file():
                    shutil.copy2(str(item), artifact_dir)
                else:
                    shutil.copytree(str(item), os.path.join(artifact_dir, item.name), dirs_exist_ok=True)
        
        print(f'{self.__class__.__name__}::Saved artifact: {name} (type: {type})')
    
    def is_finished(self):
        """Check if experiment is finished"""
        return not self.initialized
    
    # ----- dataset and model loading ----
    def load_dataset(self, data_root, eval_config={}, unseen=False, batch_size=5, load_all=False):
        """Load dataset for experiment"""
        split, _, data_config = self.data_info()
        if unseen:
            load_all = True  # load data as a whole without splitting
            data_config.update(data_folders=data_config['unseen_data_folders'])  # use the unseen folders list
        split = split if not load_all else None

        # Extra evaluation configuration
        data_config.update(eval_config)
        
        # Dataset
        data_class = getattr(data, data_config['class'])
        dataset = data_class(data_root, data_config, 
                             gt_caching=data_config.get('gt_caching', True), 
                             feature_caching=data_config.get('feature_caching', False))
        
        if 'wrapper' in data_config and data_config["wrapper"] is not None:
            datawrapper_class = getattr(data, data_config["wrapper"])
            datawrapper = datawrapper_class(dataset, known_split=split, batch_size=batch_size)
        else:
            datawrapper = data.GarmentDatasetWrapper(dataset, known_split=split, batch_size=batch_size)

        return dataset, datawrapper
    
    def load_detr_model(self, data_config, others=False):
        """Load model for inference"""
        model, criterion = models.build_model(self.in_config)
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        model = nn.DataParallel(model, device_ids=[0] if torch.cuda.is_available() else None)
        criterion.to(device)

        state_dict = self.get_best_model(device=device)['model_state_dict']
        model.load_state_dict(state_dict)
        return model, criterion, device
    
    def prediction(self, save_to, model, datawrapper, criterion=None, nick='test', sections=['test'], art_name='multi-data', use_gt_stitches=False):
        """Perform inference and save predictions"""
        prediction_path = datawrapper.predict(model, save_to=save_to, sections=sections, orig_folder_names=True, use_gt_stitches=use_gt_stitches)

        if nick:
            self.add_statistic(nick + '_folder', os.path.basename(prediction_path), log='Prediction save path')

        if art_name:
            art_name = art_name if len(datawrapper.dataset.data_folders) > 1 else datawrapper.dataset.data_folders[0]
            self.add_artifact(prediction_path, art_name, 'result')

        return prediction_path
    
    def prediction_single(self, save_to, model, datawrapper, image, data_name="outside_single"):
        """Perform inference on a single image"""
        panel_order, panel_idx, prediction_img = datawrapper.predict_single(model, image, data_name, save_to)
        return panel_order, panel_idx, prediction_img
    
    def run_single_img(self, image, model, datawrapper):
        """Run inference on a single image"""
        return datawrapper.run_single_img(image, model, datawrapper)
    
    # ---- file info -----
    def checkpoint_filename(self, check_id=None):
        """Generate checkpoint filename"""
        check_id_str = '_{}'.format(check_id) if check_id is not None else ''
        return '{}{}.pth'.format(self.checkpoint_filetag, check_id_str)

    def final_filename(self):
        """Generate final model filename"""
        return self.final_filetag + '.pth'
    
    # ----- working with files -------
    def get_best_model(self, to_path=None, device=None):
        """Load best model from checkpoint"""
        # Check for pre-trained model path in config
        if 'pre-trained' in self.in_config.get('experiment', {}) and self.in_config['experiment']['pre-trained'] is not None and os.path.exists(self.in_config['experiment']['pre-trained']):
            # Local model available
            print(f'{self.__class__.__name__}::Info::Loading locally saved model from experiment config')
            return self._load_model_from_file(self.in_config['experiment']['pre-trained'], device)
        elif 'pre-trained' in self.in_config.get('NN', {}) and self.in_config['NN']['pre-trained'] is not None and os.path.exists(self.in_config['NN']['pre-trained']):
            # Local model available
            print(f'{self.__class__.__name__}::Info::Loading locally saved model from NN config')
            return self._load_model_from_file(self.in_config['NN']['pre-trained'], device)
        else:
            # Look for best checkpoint in experiment directory
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_checkpoint_path):
                print(f'{self.__class__.__name__}::Info::Loading best checkpoint from experiment directory')
                return self._load_model_from_file(best_checkpoint_path, device)
            else:
                # Look for latest checkpoint
                checkpoints = list(Path(self.checkpoint_dir).glob('checkpoint_*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                    print(f'{self.__class__.__name__}::Info::Loading latest checkpoint: {latest_checkpoint.name}')
                    return self._load_model_from_file(latest_checkpoint, device)
                else:
                    raise RuntimeError(f'{self.__class__.__name__}::Error::No model checkpoint found')
    
    def save_checkpoint(self, state, is_best=False, wait_for_upload=False):
        """Save model checkpoint"""
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename(self.checkpoint_counter))
        torch.save(state, checkpoint_path)
        
        # Update counter
        self.checkpoint_counter += 1
        
        # If this is the best model, save a copy
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            shutil.copy2(checkpoint_path, best_model_path)
            print(f'New best model found !!')
        
        print(f'Saved checkpoint: {checkpoint_path}')
        
        # Update validation loss if provided
        if 'valid_loss' in state:
            current_loss = state['valid_loss']
            if current_loss < self.best_valid_loss:
                self.best_valid_loss = current_loss
                self.add_statistic('best_valid_loss', current_loss)
    
    def _run_config(self):
        """Get run configuration"""
        return self.in_config
    
    def _load_model_from_file(self, file, device=None):
        """Load model from file"""
        print(f'Loading model from: {file}')
        if device is not None:
            return torch.load(file, map_location=device)
        else: 
            return torch.load(file)  # to the same device it was saved from
