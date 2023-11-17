from functools import partial

from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds

def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    args=None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # preprocess for model generation/completion
    example = tokenize_and_truncate(
        example,
        completion_length=max_new_tokens,
        prompt_length=min_prompt_tokens,
        hf_model_name=hf_model_name,
        tokenizer=tokenizer,
    )
    # Logic to parse the results of tokenzation and splitting to
    # construct string versions of the prompt and baseline completion
    inputs = example["input_ids"]
    prompt_len = inputs.shape[1]
    # for isolating the "gold" baseline completion
    untruncated_inputs = example.pop("untruncated_inputs")
    full_sample_len = untruncated_inputs.shape[1]
    # decode the preprocessed input to store for audit
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    # also decode the original suffix of the input for audit as the baseline
    baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
    decoded_baseline_completion = tokenizer.batch_decode(
        baseline_completion_tokens, skip_special_tokens=True
    )[0]
    baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "prompt": re_decoded_input,
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example

def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    # assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    # assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if hf_model_name:
        if "t5" in hf_model_name or "T0" in hf_model_name:
            inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def add_idx(example, idx):
    example.update({"idx": idx})
    return example

def load_hf_dataset(args):
    args.shuffle_dataset = False
    args.shuffle_seed = 1
    args.shufle_buffer_size = 64
    args.dataset_split = "train"
    args.stream_dataset = True
    args.columns_to_remove = []

    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name
    if dataset_name == "lfqa":
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./utils/data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    ## tokenize
    args.min_prompt_tokens = 50
    args.min_sample_tokens = 225
    args.max_new_tokens = 250
    args.model_name = ""
    tokenizer = args.tokenizer
    token_kwargs = dict(
        hf_model_name=args.model_name,
        tokenizer=tokenizer,
        args=args,
    )
    token_kwargs.update(dict(min_prompt_tokens=args.min_prompt_tokens))
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    input_check_kwargs = dict(
        min_sample_len=args.min_sample_tokens,
        max_new_tokens=args.max_new_tokens,
        min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens
    )
    input_check = partial(check_input_lengths, **input_check_kwargs)
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)
    dataset_iter = iter(dataset_input_len_filtered)

    if args.return_type == "prompt":
        prompts = []
        for _ in range(args.limit_rows):
            example = next(dataset_iter)['prompt']
            prompts.append(example)
        return prompts

    if args.return_type == "all":
        examples = []
        for _ in range(args.limit_rows):
            example = next(dataset_iter)
            examples.append(example)
        return examples