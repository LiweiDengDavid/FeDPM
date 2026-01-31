def build_checkpoint_dirname(args):
    """
    Build the checkpoint directory name based on the configured hyperparameters.
    
    Args:
        args: Arguments object containing checkpoint_params and hyperparameter values
        
    Returns:
        str: Directory name in format like "Federated_Tin96_Tout96_Seed2025"
    """
    # Default parameters if not specified
    checkpoint_params = getattr(args, 'checkpoint_params', ['Tin', 'Tout', 'Seed'])
    
    # Build the directory name from specified parameters
    param_parts = []
    for param in checkpoint_params:
        attr_name = param
        if hasattr(args, attr_name):
            value = getattr(args, attr_name)
            param_parts.append(f"{param}{value}")
    
    if param_parts:
        dirname = "FeDPM_" + "_".join(param_parts)
    else:
        # Fallback if no valid parameters specified
        dirname = f"FeDPM_Tin{args.Tin}_Tout{args.Tout}_Seed{args.seed}"
    
    return dirname


def build_log_filename(args):
    """
    Build the log filename based on the configured hyperparameters.
    
    Args:
        args: Arguments object containing log_params and hyperparameter values
        
    Returns:
        str: Log filename in format like "FL_Tin96_Tout96_Seed2025.log"
    """
    # Default parameters if not specified
    log_params = getattr(args, 'log_params', ['num_rounds', 'local_epochs', 'seed'])
    
    # Build the log filename from specified parameters
    param_parts = []
    for param in log_params:
        if hasattr(args, param):
            value = getattr(args, param)
            param_parts.append(f"{param}{value}")
    
    if param_parts:
        log_filename = "FeDPM_" + "_".join(param_parts) + ".log"
    else:
        # Fallback if no valid parameters specified
        log_filename = f"FeDPM_rounds{args.num_rounds}_epochs{args.local_epochs}_seed{args.seed}.log"
    
    return log_filename
