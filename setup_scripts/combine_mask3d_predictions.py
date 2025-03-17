import os

from utils.recursive_config import Config

config = Config()

# File paths
base_path_output = config.get_subpath("masks")
ending_output = config["pre_scanned_graphs"]["high_res"]
predictions_output_dir = os.path.join(base_path_output, ending_output)

base_path_input = config.get_subpath("prescans")
ending_input = config["pre_scanned_graphs"]["high_res"]
predictions_input_dir = os.path.join(base_path_input, ending_input)
os.makedirs(predictions_input_dir, exist_ok=True)

files_to_combine = [
    os.path.join(predictions_input_dir, "predictions.txt"),
    os.path.join(predictions_input_dir, "predictions_light_switches.txt"),
    os.path.join(predictions_input_dir, "predictions_drawers.txt")
]

output_file = os.path.join(predictions_output_dir, "predictions_combined.txt")

# Combine files into the output file
with open(output_file, 'w') as outfile:
    for file_path in files_to_combine:
        if os.path.exists(file_path):
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())