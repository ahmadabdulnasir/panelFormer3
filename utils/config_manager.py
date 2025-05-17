import os
import yaml
import json
from pathlib import Path

class ConfigManager:
    """
    Central configuration manager for PanelFormer.
    Loads configuration from default.yaml and provides access to configuration values.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.root_path, 'configs/default.yaml')
        self.config = self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """Load configuration from default.yaml"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def get(self, section, key=None, default=None):
        """
        Get configuration value.
        
        Args:
            section: Configuration section (e.g., 'dataset', 'model')
            key: Configuration key within section (optional)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key, default)
    
    def get_standardization(self):
        """Get standardization data from configuration"""
        # Try to get standardization from dataset section (new format)
        if 'dataset' in self.config and 'standardize' in self.config['dataset']:
            return self.config['dataset']['standardize']
        
        # Try to get standardization from data section (newer format)
        if 'data' in self.config and 'standardize' in self.config['data']:
            return self.config['data']['standardize']
            
        # Return default standardization if not found
        print("Standardization data not found in config, using default values")
        return {
            'gt_shift': {
                'outlines': [0., 0., 0, 0],
                'rotations': [-0.38268343, -0.9238795, -1.,  0.],
                'stitch_tags': [-59.99474, -78.23346, -52.926674],   
                'translations': [-55.25636, -20.001333, -17.086796]
            },
            'gt_scale': {
                'outlines': [26.674109, 29.560705, 1, 1],
                'rotations': [1.3826834, 1.9238795, 1.2877939, 1.],
                'stitch_tags': [119.964195, 109.62911, 105.657364],
                'translations': [109.58753, 51.449017, 37.846794]
            }
        }
    
    def get_panel_classification_path(self):
        """Get panel classification path"""
        # Try to get from dataset section (new format)
        if 'dataset' in self.config and 'panel_classification' in self.config['dataset']:
            path = self.config['dataset']['panel_classification']
        # Try to get from data section (newer format)
        elif 'data' in self.config and 'panel_classification' in self.config['data']:
            path = self.config['data']['panel_classification']
        else:
            path = './configs/data_configs/panel_classes_condenced.json'
        
        # Convert relative path to absolute
        if path.startswith('./'):
            path = os.path.join(self.root_path, path[2:])
        
        return path
        
    def get_max_pattern_len(self):
        """Get max_pattern_len with fallback to default"""
        # Try to get from dataset section (new format)
        if 'dataset' in self.config and 'max_pattern_len' in self.config['dataset']:
            return self.config['dataset']['max_pattern_len']
        # Try to get from data section (newer format)
        elif 'data' in self.config and 'max_pattern_len' in self.config['data']:
            return self.config['data']['max_pattern_len']
        else:
            return 23  # Default value
            
    def get_max_panel_len(self):
        """Get max_panel_len with fallback to default"""
        # Try to get from dataset section (new format)
        if 'dataset' in self.config and 'max_panel_len' in self.config['dataset']:
            return self.config['dataset']['max_panel_len']
        # Try to get from data section (newer format)
        elif 'data' in self.config and 'max_panel_len' in self.config['data']:
            return self.config['data']['max_panel_len']
        else:
            return 14  # Default value
    
    def get_filter_params_path(self):
        """Get filter parameters path"""
        # Try to get from dataset section (new format)
        if 'dataset' in self.config and 'filter_by_params' in self.config['dataset']:
            path = self.config['dataset']['filter_by_params']
        # Try to get from data section (newer format)
        elif 'data' in self.config and 'filter_by_params' in self.config['data']:
            path = self.config['data']['filter_by_params']
        else:
            path = './configs/data_configs/param_filter.json'
        
        # Convert relative path to absolute
        if path.startswith('./'):
            path = os.path.join(self.root_path, path[2:])
        
        return path
    
    def update_config(self, section, key, value):
        """Update configuration value"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save_config(self):
        """Save configuration to default.yaml"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

# Create a singleton instance
config_manager = ConfigManager()
