import math

def compute_ber(pred, gt):
    correct, total = 0, 0
    if len(pred) != len(gt):
        print(f"Warning: length of prediction is {len(pred)} and gt is {len(gt)}")
        breakpoint()
    for p, g in zip(pred, gt):
        if p == g:
            correct += 1
        total += 1
    return correct, total