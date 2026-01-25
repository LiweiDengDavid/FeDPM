import json
import csv
import os

def human_count_params(all_params):
    '''
    Counts the total number of parameters in a model and returns a human-readable string.
    
    Args:
        all_params (iterable): An iterable of model parameters (e.g., model.parameters()).

    Returns:
        str: A string representing the total number of parameters in a human-readable format.
        int: The exact total number of parameters.
    '''
    total_params = sum(p.numel() for p in all_params)

    if total_params >= 1e9:
        param_str = f"{total_params/1e9:.2f}B"
    elif total_params >= 1e6:
        param_str = f"{total_params/1e6:.2f}M"
    elif total_params >= 1e3:
        param_str = f"{total_params/1e3:.2f}K"
    else:
        param_str = str(total_params)
    return param_str, total_params

def log_results_to_jsonl(filepath, args, metrics):
    """
    Logs experiment parameters and final metrics as a new line in a JSON Lines file.

    Args:
        filepath (str): Path to the output .jsonl file.
        args (argparse.Namespace): The command-line arguments from the experiment.
        metrics (dict): A dictionary containing the final performance metrics.
    """
    # 1. Combine arguments and metrics into a single dictionary
    results_dict = vars(args).copy()
    results_dict.update(metrics)
    
    # 2. Convert the dictionary to a JSON string
    # Ensure client_data_types list is stored as a simple string
    if 'client_data_types' in results_dict and isinstance(results_dict['client_data_types'], list):
        results_dict['client_data_types'] = '-'.join(results_dict['client_data_types'])

    json_string = json.dumps(results_dict, sort_keys=True)
    
    # 3. Open the file in append mode and write the JSON string as a new line
    with open(filepath, 'a') as f:
        f.write(json_string + '\n')

def convert_jsonl_to_csv(jsonl_path, csv_path, fields_to_include):
    """
    Reads a JSON Lines file and converts selected fields into a CSV file.

    Args:
        jsonl_path (str): Path to the input .jsonl file.
        csv_path (str): Path to the output .csv file.
        fields_to_include (list): A list of strings representing the keys to include in the CSV.
    """
    
    # 1. Read all experiment data from the .jsonl file
    all_experiments = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                all_experiments.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at '{jsonl_path}'")
        return

    if not all_experiments:
        print("No data found in the input file.")
        return

    # 2. Write the selected data to a CSV file
    with open(csv_path, 'w', newline='') as f:
        # Use only the fields that actually exist in the first record to avoid errors
        valid_fields = [field for field in fields_to_include if field in all_experiments[0]]
        
        writer = csv.DictWriter(f, fieldnames=valid_fields)
        writer.writeheader()
        
        for experiment_data in all_experiments:
            # Create a new dictionary containing only the desired fields
            filtered_data = {key: experiment_data.get(key, '') for key in valid_fields}
            writer.writerow(filtered_data)

    print(f"Successfully converted {len(all_experiments)} records to '{csv_path}'")
    print(f"Included fields: {valid_fields}")
