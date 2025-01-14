import math
from typing import List, Union

def estimate_pass_at_k(total_samples: int, num_correct: int, k: int) -> float:
    """
    Calculate unbiased estimate of pass@k as formulated by Chen et al. (HumanEval).
    Formula: 1 - comb(n - c, k) / comb(n, k)
    """
    if total_samples - num_correct < k:
        return 1.0
    if num_correct == 0:
        return 0.0
    
    # Numerically stable product formulation
    # comb(n - c, k) / comb(n, k) = prod_{i=1..k} (n - c - k + i) / (n - k + i)
    val = 1.0
    for i in range(1, k + 1):
        val *= (total_samples - num_correct - k + i) / (total_samples - k + i)
    return 1.0 - val
