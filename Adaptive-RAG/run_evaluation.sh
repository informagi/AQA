#!/bin/bash

# Check if the model type is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <model_type>"
    echo "Model types: NoR, OneR, IRCoT"
    exit 1
fi

MODEL_TYPE=$1
PREDICTION_DIR="/home/mhoveyda/AdaptiveQA/Adaptive-RAG/Agents_Executed_Final_test_w_Confidence"
OUTPUT_DIR="/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_TEST_Evaluation_Results"
LOG_DIR="LOGS_Final_Testl/$(date +'%Y-%m-%d')"


# Create the results and logs directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Set file paths based on the model type
case $MODEL_TYPE in
    NoR|nor)
        # PREDICTION_FILE="/home/mhoveyda/AdaptiveQA/Adaptive-RAG/Agents_Executed_Final_Train/nor/nor_qa_flan_xl_train_aware_210_51.json"
        PREDICTION_FILE="$PREDICTION_DIR/nor/nor_qa_flan_xl_test_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_nor.json" 
        LOG_FILE="$LOG_DIR/evaluation_nor.txt"
        ;;
    OneR|oner)
        # PREDICTION_FILE="/home/mhoveyda/AdaptiveQA/Adaptive-RAG/mohanna_test_runs_Last/oner/oner_qa_flan_xl_test_aware_210_51.json"
        PREDICTION_FILE="$PREDICTION_DIR/oner/oner_qa_flan_xl_test_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_oner.json"
        LOG_FILE="$LOG_DIR/evaluation_oner.txt"
        ;;
    IRCoT|ircot)
        # PREDICTION_FILE="/home/mhoveyda/AdaptiveQA/Adaptive-RAG/mohanna_test_runs_Last/ircot/ircot_qa_flan_xl_test_aware_210_51.json"
        PREDICTION_FILE="$PREDICTION_DIR/ircot/ircot_qa_flan_xl_test_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_ircot.json"
        LOG_FILE="$LOG_DIR/evaluation_ircot.txt"
        ;;
    *)
        echo "Invalid model type: $MODEL_TYPE"
        echo "Model types: NoR, OneR, IRCoT"
        exit 1
        ;;
esac

# Run the evaluation
python AQA_final_eval.py $PREDICTION_FILE /home/mhoveyda/AdaptiveQA/AdaptiveQA_Data_Final/test_aware_210_51.jsonl --output_file_path $OUTPUT_FILE > $LOG_FILE 2>&1

echo "Evaluation completed for model type: $MODEL_TYPE"
echo "Results saved to: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
