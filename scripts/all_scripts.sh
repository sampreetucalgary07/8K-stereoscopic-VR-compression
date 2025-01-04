# List of configuration files
# configs/E2_8K_basketball.yaml" "configs/E3_8K_sunny.yaml" "configs/E4_8K_football.yaml" "configs/E5_8K_grass.yaml" "configs/E6_8K_park.yaml"
# echo "Running Dynamic script file"
# python3 "scripts/dynamic_scripts.py"

CONFIG_FILES=("configs/E2_8K_basketball.yaml" )
#"configs/E4_8K_football.yaml" "configs/E5_8K_grass.yaml" "configs/E6_8K_park.yaml")
#print config files selected
echo "Selected config files: ${CONFIG_FILES[@]}"
# Function to print messages
print_message() {
    local message=$1
    echo "==== $message ===="
}

# Function to run a Python script
run_script() {
    local script=$1
    local config=$2

    print_message "Running ${script} with config ${config}"
    if python3 "${script}" --config_path "${config}"; then
        print_message "${script} completed successfully."
    else
        print_message "Error occurred while running ${script}."
        exit 1
    fi
}


# Iterate through each configuration file
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    # Check if the configuration file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_message "Configuration file ${CONFIG_FILE} not found!"
        exit 1
    fi

    if [[ "$CONFIG_FILE" == "configs/E2_8K_basketball.yaml" ]]; then
        # run_script "scripts/video_processing_1.py" "$CONFIG_FILE" && \
        # run_script "scripts/dataset_preprocess_2.py" "$CONFIG_FILE" && \
        # run_script "scripts/patches_3.py" "$CONFIG_FILE" && \
        # run_script "scripts/patches_filter_4.py" "$CONFIG_FILE" && \
        # run_script "scripts/train_vae_model_5.py" "$CONFIG_FILE" && \
        run_script "scripts/save_latents_6.py" "$CONFIG_FILE" && \
        run_script "scripts/save_frames_7.py" "$CONFIG_FILE"
        #run_script "scripts/evaluation_8.py" "$CONFIG_FILE" 
        # run_script "scripts/frame_merging_8.py" "$CONFIG_FILE" && \ 
        #run_script "scripts/save_video_9.py" "$CONFIG_FILE"
    else

        # Execute the scripts for the current configuration file

        # run_script "scripts/video_processing_1.py" "$CONFIG_FILE" && \
        # run_script "scripts/dataset_preprocess_2.py" "$CONFIG_FILE" && \
        # run_script "scripts/patches_3.py" "$CONFIG_FILE" && \
        # run_script "scripts/patches_filter_4.py" "$CONFIG_FILE" 
        # run_script "scripts/train_vae_model_5.py" "$CONFIG_FILE"
        # run_script "scripts/save_latents_6.py" "$CONFIG_FILE" && \
        # run_script "scripts/save_frames_7.py" "$CONFIG_FILE" 
        # run_script "scripts/evaluation_8.py" "$CONFIG_FILE" && \ 
        # run_script "scripts/frame_merging_8.py" "$CONFIG_FILE" && \ 
        # run_script "scripts/save_video_9.py" "$CONFIG_FILE"
        run_script "scripts/parse_csv_10.py" "$CONFIG_FILE"
        # run_script "scripts/transfer_learning_11.py" "$CONFIG_FILE"
    fi
done

print_message "All scripts have been executed successfully for all configuration files."
