export CUDA_VISIBLE_DEVICES="1"
huggingface-cli login --token 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 --token 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

MODEL_NAME="meta-llama/Llama-2-7b-hf"
python main_watermark.py --model_name $MODEL_NAME \
    --prompt_type guanaco --prompt_path data/alpaca_data.json --nsamples 3 --batch_size 1 \
    --method "openai" --temperature 1.0 --seeding hash --ngram 2 --scoring_method v2 \
    --delta 4 --gamma 0.25 \
    --payload 3 --payload_max 4 \
    --output_dir output/ \
    --method_detect "same" --do_eval T
    #--overwrite T
    #--dataset_config_name "realnewslike" --dataset_name "c4"


#python main_eval.py \
#    --json_path examples/results.jsonl --text_key result --tokenizer_dir $MODEL_NAME \
#    --json_path_ori data/alpaca_data.json --text_key_ori output \
#    --do_wmeval True --method openai --seeding hash --ngram 2 --scoring_method v2 \
#    --payload 9 --payload_max 100 \
#    --output_dir output/
