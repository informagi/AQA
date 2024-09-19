#!/bin/bash

# Check if all required environment variables are provided
if [ -z "$MODEL_TYPE" ] || [ -z "$GOLD_FILE" ] || [ -z "$PREDICTION_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$LOG_DIR" ]; then
    echo "Error: All environment variables must be set."
    echo "Required variables: MODEL_TYPE, GOLD_FILE, EVAL_TYPE, PREDICTION_DIR, OUTPUT_DIR, LOG_DIR"
    exit 1
fi

# Check if eval_type is valid
if [[ "$EVAL_TYPE" != "test" && "$EVAL_TYPE" != "train" ]]; then
    echo "Invalid eval_type: $EVAL_TYPE"
    echo "<eval_type> must be either 'test' or 'train'"
    exit 1
fi

echo "MODEL_TYPE: $MODEL_TYPE"
echo "GOLD_FILE: $GOLD_FILE"
echo "EVAL_TYPE: $EVAL_TYPE"
echo "PREDICTION_DIR: $PREDICTION_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "LOG_DIR: $LOG_DIR"


# Create the results and logs directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set file paths based on the model type and eval type
case $MODEL_TYPE in
    NoR|nor)
        PREDICTION_FILE="$PREDICTION_DIR/nor/nor_qa_flan_xl_${EVAL_TYPE}_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_nor_${EVAL_TYPE}.json"
        LOG_FILE="$LOG_DIR/evaluation_nor_${EVAL_TYPE}.txt"
        ;;
    OneR|oner)
        PREDICTION_FILE="$PREDICTION_DIR/oner/oner_qa_flan_xl_${EVAL_TYPE}_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_oner_${EVAL_TYPE}.json"
        LOG_FILE="$LOG_DIR/evaluation_oner_${EVAL_TYPE}.txt"
        ;;
    IRCoT|ircot)
        PREDICTION_FILE="$PREDICTION_DIR/ircot/ircot_qa_flan_xl_${EVAL_TYPE}_aware_210_51.json"
        OUTPUT_FILE="$OUTPUT_DIR/output_results_ircot_${EVAL_TYPE}.json"
        LOG_FILE="$LOG_DIR/evaluation_ircot_${EVAL_TYPE}.txt"
        ;;
    *)
        echo "Invalid model type: $MODEL_TYPE"
        echo "Model types: NoR, OneR, IRCoT"
        exit 1
        ;;
esac

# Run the evaluation
python AQA_final_eval.py "$PREDICTION_FILE" "$GOLD_FILE" --output_file_path "$OUTPUT_FILE" > "$LOG_FILE" 2>&1

echo

echo "----------------------------------------"
echo "Evaluation completed for model type: $MODEL_TYPE, eval type: $EVAL_TYPE"
echo "Results saved to: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"
