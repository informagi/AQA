#!/bin/bash

# Check if the system type argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <system_type>"
    exit 1
fi

# Assign command-line argument to system_type
system_type=$1

# Check if environment variables are set
if [ -z "$INPUT_PATH" ] || [ -z "$BASE_CONFIG_FOLDER" ] || [ -z "$BASE_OUTPUT_FOLDER" ] || [ -z "$BASE_LOG_FOLDER" ]; then
    echo "Error: One or more required environment variables are not set."
    echo "Please set INPUT_PATH, BASE_CONFIG_FOLDER, BASE_OUTPUT_FOLDER, and BASE_LOG_FOLDER."
    exit 1
fi

# Define config paths based on system type
case $system_type in
  noR|nor)
    config_path="$BASE_CONFIG_FOLDER/nor_qa_flan_t5_xl_hotpotqa.jsonnet"
    ;;
  oner|oneR)
    config_path="$BASE_CONFIG_FOLDER/oner_qa_flan_t5_xl_hotpotqa.jsonnet"
    ;;
  ircot)
    config_path="$BASE_CONFIG_FOLDER/ircot_qa_flan_t5_xl_hotpotqa.jsonnet"
    ;;
  *)
    echo "Invalid system type: $system_type"
    exit 1
    ;;
esac

input_file_name=$(basename "$INPUT_PATH")
input_file_name_no_ext="${input_file_name%.*}"

output_path="$BASE_OUTPUT_FOLDER/$system_type/${system_type}_qa_flan_xl_${input_file_name_no_ext}.json"

log_path="$BASE_LOG_FOLDER/${system_type}_qa_flan_xl_${input_file_name_no_ext}.txt"

config_dir=$(dirname "$config_path")
input_dir=$(dirname "$INPUT_PATH")
output_dir=$(dirname "$output_path")
log_dir=$(dirname "$log_path")

mkdir -p "$config_dir"
mkdir -p "$input_dir"
mkdir -p "$output_dir"
mkdir -p "$log_dir"

echo "Running inference for system type: $system_type"
echo "Using config: $config_path"
echo "Input file: $INPUT_PATH"
echo "Output file: $output_path"
echo "Log file: $log_path"

echo "Command: python -m commaqa.inference.configurable_inference_AQA --config $config_path --input $INPUT_PATH --output $output_path > $log_path 2>&1"

python -m commaqa.inference.configurable_inference_AQA --config "$config_path" --input "$INPUT_PATH" --output "$output_path" > "$log_path" 2>&1
