import yaml
from argparse import Namespace

def load_additional_config(yaml_path):
    """
    Load ablation configuration from YAML file.
    Converts nested YAML structure into a flat namespace with dot notation.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten the nested YAML structure into a namespace
    namespace_dict = {}
    if config:
        for section, params in config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    # Create flattened key: section_key
                    flat_key = f"{section}_{key}"
                    namespace_dict[flat_key] = value
            else:
                namespace_dict[section] = params
    
    return Namespace(**namespace_dict)

def merge_args(cmd_args, yaml_args):
    """
    Merge command line arguments with YAML configuration.
    Command line arguments take precedence over YAML values.
    """
    # Convert yaml_args to dict and update cmd_args
    yaml_dict = vars(yaml_args) if yaml_args else {}
    cmd_dict = vars(cmd_args)
    
    # Only add YAML values that are not already in cmd_args
    for key, value in yaml_dict.items():
        if key not in cmd_dict or cmd_dict[key] is None:
            cmd_dict[key] = value
    
    return cmd_args