# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
python main_watermark.py --model_name guanaco-7b \
    --prompt_type guanaco --prompt_path data.json --nsamples 10 --batch_size 16 \
    --method openai --method_detect openai --seeding hash --ngram 2 --scoring_method v2 --temperature 1.0 \
    --payload 0 --payload_max 4 \
    --output_dir output/

python main_watermark.py --model_name guanaco-7b \
    --prompt_type guanaco --prompt_path data.json --nsamples 10 --batch_size 16 \
    --method maryland --seeding hash --ngram 2 --gamma 0.25 --delta 2.0 \
    --payload 0 --payload_max 4 \
    --output_dir output/
"""

import argparse
from typing import Dict, List
import os
import time
import json
import math 

import tqdm
import pandas as pd
import numpy as np

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from wm import (WmGenerator, OpenaiGenerator, OpenaiDetector, OpenaiNeymanPearsonDetector, 
                OpenaiDetectorZ, MarylandGenerator, MarylandDetector, MarylandDetectorZ)
from wm.wm_utils import compute_ber
from wm.dataset_utils import load_hf_dataset
import utils


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str)

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/alpaca_data.json")
    parser.add_argument('--prompt_type', type=str, default="alpaca", 
                        help='type of prompt formatting. Choose between: alpaca, oasst, guanaco')
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')
    # data parameters
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--dataset_config_name', type=str, default=None)
    parser.add_argument('--limit_rows', type=int, default=5000)

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=250)

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose between: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.)')
    parser.add_argument('--method_detect', type=str, default='same',
                        help='Statistical test to detect watermark. Choose between: same (same as method), openai, openaiz, openainp, maryland, marylandz')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=3.0,
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # multibit
    parser.add_argument('--payload', type=int, default=0, help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message, must be inferior to the vocab size at the moment')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--do_eval', type=utils.bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)

    parser.add_argument("--overwrite", type=utils.bool_inst, default=True)

    return parser


def format_prompts(prompts: List[Dict], prompt_type: str) -> List[str]:
    if prompt_type=='alpaca':
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }
    elif prompt_type=='guanaco':
        PROMPT_DICT = {
            "prompt_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Input:\n{input}\n\n### Assistant:"
            ),
            "prompt_no_input": (
                "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: {instruction}\n\n### Assistant:"
            )
        }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompts = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in prompts
    ]
    return prompts

def load_prompts(json_path: str, prompt_type: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        prompts = json.loads(f.read())
    new_prompts = prompts
    # new_prompts = [prompt for prompt in prompts if len(prompt["output"].split()) > 5]
    new_prompts = new_prompts[:nsamples]
    print(f"Filtered {len(new_prompts)} prompts from {len(prompts)}")
    new_prompts = format_prompts(new_prompts, prompt_type)
    return new_prompts

def load_results(json_path: str, nsamples: int=None, result_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            raw_data = json.loads(f.read())
        else:
            raw_data = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[result_key] for o in raw_data]
    new_prompts = new_prompts[:nsamples]
    return new_prompts, raw_data


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    if "llama-2" in args.model_name.lower():
        model_name = args.model_name
        adapters_name = None
    elif args.model_name == "llama-7b":
        model_name = "llama-7b"
        adapters_name = None
    elif args.model_name == "guanaco-7b":
        model_name = "llama-7b"
        adapters_name = 'timdettmers/guanaco-7b'
    elif args.model_name == "guanaco-13b":
        model_name = "llama-13b"
        adapters_name = 'timdettmers/guanaco-13b'
    else:
        model_name = args.model_name
        adapters_name = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.tokenizer = tokenizer
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={i: '32000MB' for i in range(args.ngpus)},
        offload_folder="offload",
    )
    if adapters_name is not None:
        model = PeftModel.from_pretrained(model, adapters_name)
    if not hasattr(model.config, 'max_sequence_length'):
        model.config.max_sequence_length = 4096
    if not hasattr(model.config, 'pad_token_id') or getattr(model.config, 'pad_token_id') is None:
        model.config.pad_token_id = 0
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Using {args.ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                    payload=args.payload, payload_max=args.payload_max)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                      payload=args.payload, gamma=args.gamma, delta=args.delta,
                                      payload_max=args.payload_max)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    else:
        if args.dataset_name:
            args.return_type = "prompt"
            prompts = load_hf_dataset(args)
        else:
            prompts = load_prompts(json_path=args.prompt_path, prompt_type=args.prompt_type, nsamples=args.limit_rows)

    # do splits
    if args.split is not None:
        nprompts = len(prompts)
        left = nprompts * args.split // args.nsplits 
        right = nprompts * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nprompts
        prompts = prompts[left:right]
        print(f"Creating prompts from {left} to {right}")
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if not args.overwrite:
        if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
            with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
                for _ in f:
                    start_point += 1
    print(f"Starting from {start_point}")

    # generate
    all_times = []
    write_type = "w" if args.overwrite else "a"
    import random
    writer = open(os.path.join(args.output_dir, f"results.jsonl"), write_type)
    valid_cnt = start_point
    with open(os.path.join(args.output_dir, f"all_results.jsonl"), write_type) as f:
        for ii in range(start_point, len(prompts), args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            gt_payload = random.randint(0, args.payload_max)
            results, token_lengths = generator.generate(
                prompts[ii:ii+chunk_size], 
                max_gen_len=args.max_gen_len, 
                temperature=args.temperature, 
                top_p=args.top_p,
                payload=gt_payload
            )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # check if output is valid
            filtered_prompts, filtered_results = [], []
            for g_idx, g_token in enumerate(token_lengths):
                if g_token > args.max_gen_len - 25:
                    filtered_prompts.append(prompts[ii + g_idx])
                    filtered_results.append(results[g_idx])
                    valid_cnt += 1
            
            for prompt, result in zip(filtered_prompts, filtered_results):
                writer.write(json.dumps({
                    "prompt": prompt, 
                    "result": result[len(prompt):],
                    "speed": speed,
                    'gt_payload': gt_payload,
                    "eta": eta}) + "\n")
                writer.flush()

            # log
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                f.write(json.dumps({
                    "prompt": prompt, 
                    "result": result[len(prompt):],
                    "speed": speed,
                    'gt_payload': gt_payload,
                    "eta": eta}) + "\n")
                f.flush()
            
            if valid_cnt >= args.nsamples:
                break 
                
    print(f"Average time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")

    if args.method_detect == 'same':
        args.method_detect = args.method
    if (not args.do_eval) or (args.method_detect not in ["openai", "maryland", "marylandz", "openaiz", "openainp"]):
        return

    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                  payload_max=args.payload_max)
    elif args.method_detect == "openaiz":
        detector = OpenaiDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                   payload_max=args.payload_max)
    elif args.method_detect == "openainp":
        detector = OpenaiNeymanPearsonDetector(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                               payload_max=args.payload_max)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                    gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)
    elif args.method_detect == "marylandz":
        detector = MarylandDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key,
                                     gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)

    # build sbert model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results_orig, _ = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="output")
    if args.split is not None:
        results_orig = results_orig[left:right]

    # evaluate
    results, raw_data = load_results(json_path=os.path.join(args.output_dir, f"results.jsonl"), nsamples=args.nsamples,
                           result_key="result")
    log_stats = []
    text_index = left if args.split is not None else 0
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
        for text, text_orig, example in tqdm.tqdm(zip(results, results_orig, raw_data)):
            # compute watermark score
            if args.method_detect == "openainp":
                scores_no_aggreg, probs = detector.get_scores_by_t([text], scoring_method=args.scoring_method, payload_max=args.payload_max)
                scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                pvalues = detector.get_pvalues(scores_no_aggreg, probs)
            else:
                scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method, payload_max=args.payload_max)
                scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                pvalues = detector.get_pvalues(scores_no_aggreg) 
            if args.payload_max:
                # decode payload and adjust pvalues
                payloads = np.argmin(pvalues, axis=1).tolist()
                all_pvalues = [pvalues[0].tolist()]
                pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
                scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
                # adjust pvalue to take into account the number of tests (2**payload_max)
                # use exact formula for high values and (more stable) upper bound for lower values
                M = args.payload_max+1
                pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
                bit_width = math.ceil(math.log2(args.payload_max))
                gold_msg = format(example['gt_payload'], f"0{bit_width}b")
                pred_msg = format(payloads[0], f"0{bit_width}b")
                correct, total = compute_ber(pred_msg, gold_msg)
                
            else:
                payloads = [ 0 ] * len(pvalues)
                pvalues = pvalues[:,0].tolist()
                all_pvalues = pvalues
                scores = [float(s[0]) for s in scores]
            num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
            # compute sbert score
            xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
            score_sbert = cossim(xs[0], xs[1]).item()
            # log stats and write
            log_stat = {
                'text_index': text_index,
                'num_token': num_tokens[0],
                'score': scores[0],
                'pvalue': pvalues[0], 
                'all_pvalues': all_pvalues[0],
                'score_sbert': score_sbert,
                'payload': payloads[0],
                'acc': correct / total
            }
            log_stats.append(log_stat)
            f.write(json.dumps(log_stat)+'\n')
            text_index += 1
        df = pd.DataFrame(log_stats)
        df['log10_pvalue'] = np.log10(df['pvalue'])
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
        with open(os.path.join(args.output_dir, "summary.log"), "w") as file:
            print(f">>> Scores: \n{df.describe(percentiles=[])}", file=file)
        print(f"Saved scores to {os.path.join(args.output_dir, 'scores.csv')}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
