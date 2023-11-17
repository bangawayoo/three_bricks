export CUDA_VISIBLE_DEVICES="1"
huggingface-cli login --token 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 --token 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

MODEL_NAME="meta-llama/Llama-2-7b-hf"
B=16
METHOD="openai"
let PAYLOAD=2**B-1
JSON_PATH_DIR="${METHOD}-${B}bit"
OUTPUT_DIR="output/${METHOD}-${B}bit"

#MODEL_NAME="facebook/opt-1.3b"
python main_watermark.py --model_name $MODEL_NAME \
    --prompt_type guanaco --prompt_path data/alpaca_data.json --nsamples 100 --limit_rows 5000 --batch_size 1 \
    --method $METHOD --temperature 0.7 --seeding hash --ngram 1 --scoring_method v2 \
    --delta 2 --gamma 0.25 \
    --payload 0 --payload_max $PAYLOAD \
    --output_dir $OUTPUT_DIR \
    --method_detect "same" --do_eval T --overwrite T \
    --dataset_config_name "realnewslike" --dataset_name "c4"




#python main_eval.py \
#    --json_path examples/results.jsonl --text_key result --tokenizer_dir $MODEL_NAME \
#    --json_path_ori data/alpaca_data.json --text_key_ori output \
#    --do_wmeval True --method openai --seeding hash --ngram 2 --scoring_method v2 \
#    --payload 9 --payload_max 100 \
#    --output_dir output/
