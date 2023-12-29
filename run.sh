export CUDA_VISIBLE_DEVICES="0"
huggingface-cli login --token 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 --token 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

MODEL_NAME="meta-llama/Llama-2-7b-hf"
#MODEL_NAME="facebook/opt-1.3b"
B=24
METHOD="mpac"
let PAYLOAD=2**B-1
N_SAM=100
MBS=4
OUTPUT_DIR="output/${METHOD}-${B}bit"

python main_watermark.py --model_name $MODEL_NAME \
    --prompt_type guanaco --prompt_path data/alpaca_data.json --nsamples $N_SAM --limit_rows 5000 --batch_size 1 \
    --method $METHOD --temperature 0.7 --seeding hash --ngram 1 --scoring_method v2 \
    --delta 2 --gamma 0.25 \
    --payload 0 --payload_max $PAYLOAD \
    --output_dir $OUTPUT_DIR \
    --method_detect "same" --do_eval T --overwrite T \
    --dataset_config_name "realnewslike" --dataset_name "c4" --message_block_size $MBS

