#Importing required libraries
import os
import subprocess
from ruamel.yaml import YAML
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# importing user defined libraries
from src.utils.patch_utils import get_patchSize_list
from src.utils.eval_utils import pd_read_csv

def update_yaml(file_path, new_values):
    """Update the YAML file with new values, preserving comments."""
    yaml = YAML()

    # Load the existing YAML file
    with open(file_path, "r") as file:
        data = yaml.load(file)

    # Update the data with new values
    def update_dict(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                original[key] = update_dict(original.get(key, {}), value)
            else:
                original[key] = value
        return original

    updated_data = update_dict(data, new_values)

    yaml.default_flow_style = False
    # Write the updated YAML file, preserving comments
    with open(file_path, "w") as file:
        yaml.dump(updated_data, file)


def run_script(script_name, config_path, config):
    update_yaml(config_path, config)
    subprocess.run(
        [
            "python3",
            script_name,
            "--config_path",
            config_path,
        ]
    )


def run_experiments():
    config = {
        "default_params": {
            "base_path": "/home/ubuntu/personal"
        },
        "tl_params": {
            "base_model_path": "VAE_Analysis/data/video_dataset/8K_basketball/trained_models/E2_model_dis.pth",
            "copy_model": True,
            "epoch": 15,
            "model_copy_path": "VAE_Analysis/data/video_dataset/8K_sunny/TL_model_results/E2_model/E2_base_model.pth",
            "is_model_pth": True,
            "enc_unfreeze_perc": 0.75,
            "dec_unfreeze_perc": 0.75,
            "save_csv_path": "VAE_Analysis/data/video_dataset/8K_sunny/TL_model_results/E2_model/logs/E2_enc_75_dec_75.csv",
            "overwrite_prev_csv": True,
            "save_model_path": "VAE_Analysis/data/video_dataset/8K_sunny/TL_model_results/E2_model/models",
            "save_model_name": "E2_based_75",  # Don't add .pt or .pth
            "save_model_as_pth": True,
        }

    }

    diff_train_paths = [
        "VAE_Analysis/data/video_dataset/8K_basketball/",
        "VAE_Analysis/data/video_dataset/8K_sunny/",
        "VAE_Analysis/data/video_dataset/8K_football/",
        "VAE_Analysis/data/video_dataset/8K_grass/",
        "VAE_Analysis/data/video_dataset/8K_park/",
    ]
    config_paths = [
        "configs/E2_8K_basketball.yaml",
        "configs/E3_8K_sunny.yaml",
        "configs/E4_8K_football.yaml",
        "configs/E5_8K_grass.yaml",
        "configs/E6_8K_park.yaml",
    ]
    for train_path, config_path in zip(diff_train_paths, config_paths):
        print("\nRunning experiment for base path: ", train_path)
        print("\nUpdating config file: ", config_path)

        datasets_to_train = [ exp for exp in diff_train_paths if exp != train_path]

        for dataset in datasets_to_train:
            print("\nRunning experiment for dataset: ", dataset)
            config["tl_params"]["base_model_path"] = os.path.join(dataset, "trained_models/mE2_tE2_ds_40.pth")
            config["tl_params"]["copy_model"] = True
            config["tl_params"]["model_copy_path"] = os.path.join(dataset, "TL_model_results/"+config_path.split("/")[1][:2]+"_model/"+config_path.split("/")[1][:2]+"_base_model.pth")
            config["tl_params"]["is_model_pth"] = True
            config["tl_params"]["enc_unfreeze_perc"] = 1.0
            config["tl_params"]["dec_unfreeze_perc"] = 1.0
            config["tl_params"]["save_csv_path"] = os.path.join(dataset, "TL_model_results/"+config_path.split("/")[1][:2]+"_model/logs/"+config_path.split("/")[1][:2]+"_enc_100_dec_100.csv")
            config["tl_params"]["overwrite_prev_csv"] = True
            config["tl_params"]["save_model_path"] = os.path.join(dataset, "TL_model_results/"+config_path.split("/")[1][:2]+"_model/models")
            config["tl_params"]["save_model_name"] = config_path.split("/")[1][:2]+"_based_100"

            df = pd_read_csv(os.path.join(config["default_params"]["base_path"], dataset, "logs/mE2_tE2_ds_40.csv"))
            
            config["tl_params"]["prev_best_loss"] = round(float(df['perceptual_loss'].min()),5)
            print("\nPrevious best loss: ", config["tl_params"]["prev_best_loss"])

            run_script("scripts/transfer_learning_11.py", config_path, config)
            
        
            #print(config)
        print("\n ------------------------------------------------------ \n")
        


                
        


if __name__ == "__main__":
    run_experiments()
