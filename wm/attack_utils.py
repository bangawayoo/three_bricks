import random

import torch

from wm.dataset_utils import load_hf_dataset

OUTPUT_TEXT_COLUMN_NAMES = ['w_wm_output', 'baseline_completion']
def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            tokenized = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]

            if text_col == "baseline_completion":
                tokenized = tokenized[:args.max_gen_len]

            # empty tensors are float type by default
            # this leads to an error when constructing pyarrow table
            if not str(tokenized.dtype) == "torch.int64":
                tokenized = tokenized.long()
            example[f"{text_col}_tokd"] = tokenized
    return example

def single_insertion(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):

    top_insert_loc = min_token_count - attack_len
    # quick fix for when attack_len > min_token_count.
    if top_insert_loc <= 0:
        return torch.tensor(tokenized_no_wm_output)
    rand_insert_locs = torch.randint(low=0, high=top_insert_loc, size=(2,))

    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    tokenized_no_wm_output_cloned[
        rand_insert_locs[0].item() : rand_insert_locs[0].item() + attack_len
    ] = tokenized_w_wm_output[rand_insert_locs[1].item() : rand_insert_locs[1].item() + attack_len]

    return tokenized_no_wm_output_cloned

def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    # if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
    #     # # if not, copy the orig w_wm_output to w_wm_output_attacked
    #     # NOTE changing this to return "" so that those fail/we can filter out these examples
    #     example["w_wm_output_attacked"] = ""
    #     example["w_wm_output_attacked_length"] = 0
    #     return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))
    # input ids might have been converted to float if empty rows exist
    for key in example.keys():
        if "tokd" in key:
            example[key] = list(map(int, example[key]))

    tokenized_attacked_output = single_insertion(
        args.cp_wm_insertion_len,
        min_token_count,
        tokenized_dst,
        tokenized_src)

    # error occurred during attacking
    if tokenized_attacked_output is None:
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    tokenized_attacked_output = list(map(int, tokenized_attacked_output))

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example


def run_copy_paste_attack(args, results):
    args.return_type = "all"
    examples = load_hf_dataset(args)
    random.shuffle(examples)
    examples = examples[:len(results)]
    for ex, res in zip(examples, results):
        ex['w_wm_output'] = res

    tokenized = [tokenize_for_copy_paste(ex, args.tokenizer, args) for ex in examples]
    args.cp_wm_insertion_len = int(args.srcp * args.max_gen_len)
    args.cp_attack_dst_col = 'baseline_completion'
    args.cp_attack_src_col = 'w_wm_output'
    attacked_examples = [copy_paste_attack(ex, args.tokenizer, args) for ex in tokenized]
    results = [ex['w_wm_output_attacked'] for ex in attacked_examples]
    return results