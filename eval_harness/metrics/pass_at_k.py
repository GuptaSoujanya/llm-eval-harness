def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k: return 1.0
    if c == 0: return 0.0
    val = 1.0
    for i in range(1, k + 1):
        val *= (n - c - k + i) / (n - k + i)
    return 1.0 - val
