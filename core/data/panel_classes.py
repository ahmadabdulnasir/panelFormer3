""" Panel classification Interface """

from collections import OrderedDict
import json
import numpy as np


class PanelClasses():
    """ Interface to access panel classification by role """
    def __init__(self, classes_file):

        self.filename = classes_file

        # Handle relative paths by checking if the path is relative to the script directory
        import os
        if classes_file.startswith('./') or not os.path.isabs(classes_file):
            # Try to find the file relative to the script directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.join(script_dir, classes_file.lstrip('./'))
            
            if os.path.exists(abs_path):
                classes_file = abs_path
            else:
                # Try another approach - use the current working directory
                cwd_path = os.path.join(os.getcwd(), classes_file.lstrip('./'))
                if os.path.exists(cwd_path):
                    classes_file = cwd_path
                else:
                    # Try with the former directory as base
                    former_path = os.path.join(os.getcwd(), 'former', classes_file.lstrip('./'))
                    if os.path.exists(former_path):
                        classes_file = former_path
            
        # print(f"Loading panel classes from: {classes_file}")
        
        try:
            with open(classes_file, 'r') as f:
                # preserve the order of classes names
                self.classes = json.load(f, object_pairs_hook=OrderedDict)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find panel classes file at {classes_file}. Original error: {e}")

        self.names = list(self.classes.keys())
        
        self.panel_to_idx = {}
        for idx, class_name in enumerate(self.classes):
            panels_list = self.classes[class_name]
            for panel in panels_list:
                self.panel_to_idx[tuple(panel)] = idx
        
    def __len__(self):
        return len(self.classes)

    def class_idx(self, template, panel):
        """
            Return idx of class for given panel (name) from given template(name)
        """
        # TODO process cases when given pair does not exist in the classes

        return self.panel_to_idx[(template, panel)]

    def class_name(self, idx):
        return self.names[idx]

    def map(self, template_name, panel_list):
        """Map the list of panels for given template to the given list"""
        
        out_list = np.empty(len(panel_list))
        for idx, panel in enumerate(panel_list):
            if panel == 'stitch':
                out_list[idx] = -1
                print(f'{self.__class__.__name__}::Warning::Mapping stitch label')
            else:
                out_list[idx] = self.panel_to_idx[(template_name, panel)]

        return out_list
