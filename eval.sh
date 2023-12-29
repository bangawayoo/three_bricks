export CUDA_VISIBLE_DEVICES="0"
huggingface-cli login --token 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 --token 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

MODEL_NAME="meta-llama/Llama-2-7b-hf"
SRCP=.7
B=24
METHOD="openai"
let PAYLOAD=2**B-1
MBS=4
JSON_PATH_DIR="${METHOD}-${B}bit"
OUTPUT_DIR="output/${METHOD}-${B}bit"


python main_eval.py \
    --json_path "${OUTPUT_DIR}/results.jsonl" \
    --text_key result --tokenizer_dir $MODEL_NAME --do_wmeval True \
    --nsamples 1000 --limit_rows 1000 --batch_size 1 \
    --method $METHOD --temperature 0.7 --seeding hash --ngram 1 --scoring_method v2 \
    --delta 2 --gamma 0.25 \
    --payload 0 --payload_max $PAYLOAD \
    --output_dir $OUTPUT_DIR \
    --do_eval T \
    --dataset_config_name "realnewslike" --dataset_name "c4" --eval_human_text F \
    --message_block_size $MBS

    #--attack_name "copy-paste" --srcp $SRCP
