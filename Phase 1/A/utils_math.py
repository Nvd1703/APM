"""
utils_math.py

Deterministic numeric helpers for Living Cryptographic Ecosystem.

This module provides the mathematical primitives used by all other modules:
- Matrix-vector multiplication with arbitrary precision integers
- Modular arithmetic operations (reduction, exponentiation)
- Cyclic distance computation in finite fields
- Batch residue computation (memory-conscious)
- Robust statistical scaling (per-generation normalization)

All operations use Python's arbitrary-precision int for intermediate values to
prevent overflow, then convert to appropriate output types (int64, float, etc).

Design principles:
1. No overflow: use Python int internally
2. Deterministic: all operations reproducible given same inputs
3. Efficient: vectorize where safe, block for memory management
4. Transparent: clear docstrings with numerical properties

Example:
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]], dtype=int)  # 2×2 matrix
    >>> V = np.array([5, 7], dtype=int)              # 2-vector
    >>> S = safe_matmul_int(A, V)                    # Matrix-vector product
    >>> # S = [1*5 + 2*7, 3*5 + 4*7] = [19, 43]
    >>> moduli = np.array([7, 11], dtype=np.int64)
    >>> P = modular_vector_reduce(S, moduli)         # Apply moduli
    >>> # P = [19 mod 7, 43 mod 11] = [5, 10]
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Sequence
from dataclasses import dataclass


# ============================================================================
# MATRIX & MODULAR ARITHMETIC
# ============================================================================

def safe_matmul_int(
    A: np.ndarray,
    V: np.ndarray
) -> Union[List[int], np.ndarray]:
    """
    Matrix-vector multiplication with arbitrary-precision integers.
    
    Computes S = A · V using Python's unlimited-precision integers to
    prevent overflow. Suitable for large intermediate values that would
    overflow int64.
    
    Mathematical operation:
        S_i = Σ_j A[i,j] * V[j]  (unbounded arithmetic over ℤ)
    
    Args:
        A: np.ndarray of shape (d, k), any integer dtype - transformation matrix
        V: np.ndarray of shape (k,), any integer dtype - input vector
    
    Returns:
        List of Python ints (length d) representing S
        Each element is unbounded (no overflow guarantee)
    
    Raises:
        ValueError: If shapes are incompatible (A.shape[1] != len(V))
    
    Example:
        >>> A = np.array([[1, 2], [3, 4]], dtype=int)
        >>> V = np.array([5, 7], dtype=int)
        >>> S = safe_matmul_int(A, V)
        >>> assert S == [1*5 + 2*7, 3*5 + 4*7]
        >>> assert S == [19, 43]
        
        >>> # Large values that would overflow int64
        >>> A = np.array([[2**40, 2**40]], dtype=int)
        >>> V = np.array([2**40, 2**40]], dtype=int)
        >>> S = safe_matmul_int(A, V)
        >>> assert S[0] == 2 * (2**80)  # Correct despite overflow risk
    
    Properties:
        - Result is independent of input dtype (always Python int)
        - Numerically stable (no rounding)
        - Time: O(d * k * log(max_value)) due to big integer arithmetic
    """
    if A.shape[1] != len(V):
        raise ValueError(
            f"Incompatible shapes: A.shape[1]={A.shape[1]}, len(V)={len(V)}"
        )
    
    d = A.shape[0]
    S = []
    
    for i in range(d):
        # Compute S_i = Σ_j A[i,j] * V[j] using Python int
        s_i = int(0)
        for j in range(A.shape[1]):
            s_i += int(A[i, j]) * int(V[j])
        S.append(s_i)
    
    return S


def modular_vector_reduce(
    S: Sequence[Union[int, np.integer]],
    moduli: np.ndarray
) -> np.ndarray:
    """
    Apply per-coordinate modular reduction.
    
    Reduces each element S[i] modulo moduli[i] to produce coordinates
    in their respective finite fields Z_{m_i}.
    
    Mathematical operation:
        P_i = S_i mod m_i  for i = 1..d
    
    Args:
        S: Sequence of integers (Python int or np.integer) of length d
           Can be list, tuple, array, or any sequence
        moduli: np.ndarray of shape (d,), dtype int64 - moduli (each > 1)
    
    Returns:
        np.ndarray of shape (d,), dtype int64
        Each P_i in range [0, moduli[i]-1]
    
    Raises:
        ValueError: If S and moduli have different lengths or moduli contains 0
    
    Example:
        >>> S = [19, 43, 100]
        >>> moduli = np.array([7, 11, 13], dtype=np.int64)
        >>> P = modular_vector_reduce(S, moduli)
        >>> assert np.array_equal(P, np.array([5, 10, 9], dtype=np.int64))
    
    Properties:
        - P[i] ∈ [0, moduli[i]-1] always
        - Independent of input dtype (converts to Python int internally)
        - Handles negative values correctly (Python % operator)
    """
    if len(S) != len(moduli):
        raise ValueError(f"S length {len(S)} != moduli length {len(moduli)}")
    
    if np.any(moduli <= 0):
        raise ValueError(f"All moduli must be > 0, got {moduli}")
    
    P = np.array([int(S[i]) % int(moduli[i]) for i in range(len(S))], dtype=np.int64)
    return P


def modular_pow_vector(
    S: Sequence[Union[int, np.integer]],
    exponents: Sequence[Union[int, np.integer]],
    moduli: Sequence[Union[int, np.integer]]
) -> np.ndarray:
    """
    Vectorized modular exponentiation.
    
    Computes S_i' = S_i^{r_i} mod m_i for each coordinate, using Python's
    built-in pow(base, exp, mod) for efficient modular exponentiation.
    
    Mathematical operation:
        S_i' = pow(S_i, r_i, m_i)  (modular exponentiation)
    
    Args:
        S: Sequence of base values (length d)
        exponents: Sequence of exponents (length d)
        moduli: Sequence of moduli (length d)
    
    Returns:
        np.ndarray of shape (d,), dtype int64
        Each S_i' = S_i^{r_i} mod m_i
    
    Raises:
        ValueError: If sequences have different lengths or moduli contains 0
    
    Example:
        >>> S = [2, 3, 5]
        >>> exponents = [3, 2, 2]
        >>> moduli = [7, 11, 13]
        >>> result = modular_pow_vector(S, exponents, moduli)
        >>> # result[0] = 2^3 mod 7 = 8 mod 7 = 1
        >>> # result[1] = 3^2 mod 11 = 9 mod 11 = 9
        >>> # result[2] = 5^2 mod 13 = 25 mod 13 = 12
        >>> assert np.array_equal(result, np.array([1, 9, 12], dtype=np.int64))
    
    Properties:
        - Efficient: uses Python's optimized pow(b, e, m) which handles large exponents
        - No overflow: all arithmetic in modular field
        - Deterministic: identical to repeated multiplication mod m
    """
    if not (len(S) == len(exponents) == len(moduli)):
        raise ValueError(
            f"Sequences must have same length: len(S)={len(S)}, "
            f"len(exponents)={len(exponents)}, len(moduli)={len(moduli)}"
        )
    
    result = np.array(
        [pow(int(S[i]), int(exponents[i]), int(moduli[i])) for i in range(len(S))],
        dtype=np.int64
    )
    return result


def cyclic_distance(a: int, b: int, m: int) -> int:
    """
    Minimal cyclic distance between two residues in Z_m.
    
    For residues a, b ∈ Z_m, computes the minimal distance considering
    that Z_m is cyclic (wraps around at m).
    
    Mathematical definition:
        dist(a, b, m) = min(|a - b|, m - |a - b|)
    
    This is the "circular" distance, e.g., in Z_7:
        - distance(1, 6, 7) = min(5, 2) = 2 (shorter to go around)
    
    Args:
        a: First residue (int in [0, m))
        b: Second residue (int in [0, m))
        m: Modulus (int > 1)
    
    Returns:
        int in [0, floor(m/2)] representing minimum cyclic distance
    
    Example:
        >>> cyclic_distance(1, 6, 7)
        2
        >>> cyclic_distance(0, 3, 7)
        3
        >>> cyclic_distance(5, 5, 10)
        0
    
    Properties:
        - Symmetric: dist(a, b, m) == dist(b, a, m)
        - Bounded: 0 ≤ dist ≤ ⌊m/2⌋
        - dist(a, a, m) = 0 always
        - dist(a, b, m) + dist(b, c, m) ≥ dist(a, c, m) (triangle inequality)
    
    Normalized variant (for similarity):
        norm_dist = (m/2 - dist) / (m/2)  produces similarity in [0, 1]
    """
    a = int(a) % int(m)
    b = int(b) % int(m)
    m = int(m)
    
    # Direct distance
    direct = abs(a - b)
    # Wraparound distance
    wraparound = m - direct
    
    return min(direct, wraparound)


# ============================================================================
# BATCH & VECTORIZED OPERATIONS
# ============================================================================

def batch_residue_matrix(
    coords_array: np.ndarray,
    modulus: int,
    blocksize: int = 100
) -> np.ndarray:
    """
    Compute residue matrix for all genes under a given modulus.
    
    For population of N genes with dimension d, computes R where
    R[i, j] = coords[i, j] mod modulus for all i, j.
    
    Processes in blocks to manage memory for large N·d.
    
    Args:
        coords_array: np.ndarray of shape (N, d), dtype int64 - population coordinates
        modulus: int > 0 - modulus for reduction
        blocksize: int > 0 - number of genes per block (for memory management)
    
    Returns:
        np.ndarray of shape (N, d), dtype int64 - residue matrix
        Each entry R[i, j] in [0, modulus-1]
    
    Example:
        >>> coords = np.array([[5, 10], [15, 20]], dtype=np.int64)
        >>> modulus = 7
        >>> R = batch_residue_matrix(coords, modulus)
        >>> assert R[0, 0] == 5 % 7 == 5
        >>> assert R[0, 1] == 10 % 7 == 3
        >>> assert R[1, 0] == 15 % 7 == 1
        >>> assert R[1, 1] == 20 % 7 == 6
    
    Properties:
        - Time: O(N*d) operations
        - Memory: O(N*d) for output, O(blocksize*d) temporary
        - Vectorized: uses NumPy broadcasting for efficiency
    """
    N, d = coords_array.shape
    modulus = int(modulus)
    
    residue_matrix = np.zeros((N, d), dtype=np.int64)
    
    for block_start in range(0, N, blocksize):
        block_end = min(block_start + blocksize, N)
        block = coords_array[block_start:block_end, :]
        residue_matrix[block_start:block_end, :] = np.mod(block, modulus)
    
    return residue_matrix


# ============================================================================
# STATISTICAL OPERATIONS
# ============================================================================

def robust_scale(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Robust per-generation normalization (median + IQR based).
    
    Scales values using robust statistics to prevent outliers from dominating.
    Used in fitness arena normalization to ensure fair aggregation across
    arenas with different baseline scales.
    
    Algorithm:
        1. Compute median and interquartile range (IQR = Q75 - Q25)
        2. Normalize: (value - median) / max(IQR, ε)
        3. Clip: to [0, 1] range
        4. Return: normalized values and scaling parameters
    
    Args:
        values: np.ndarray of shape (N,) with arbitrary numeric values
    
    Returns:
        Tuple of:
            normalized: np.ndarray of shape (N,), dtype float in [0, 1]
            params: Dict with keys:
                'median': float - median of input values
                'q25': float - 25th percentile
                'q75': float - 75th percentile
                'iqr': float - interquartile range
                'epsilon': float - small value to prevent division by zero
    
    Example:
        >>> values = np.array([0.1, 0.2, 0.3, 0.5, 0.9], dtype=float)
        >>> normalized, params = robust_scale(values)
        >>> assert 0 <= np.min(normalized) and np.max(normalized) <= 1
        >>> assert 'median' in params
        >>> assert params['iqr'] > 0 or params['iqr'] == 0  # Valid even if iqr=0
    
    Properties:
        - Robust to outliers (uses median/IQR, not mean/std)
        - Output in [0, 1] by design (clipped)
        - Reproducible: same input → same output
        - Handles constant input (all equal): IQR=0 → normalized to constant
    
    Usage in fitness aggregation:
        Each arena normalized independently per generation,
        then aggregated as: F_total = Σ w_a * F_norm_a
    """
    values = np.array(values, dtype=float)
    epsilon = 1e-10
    
    median = float(np.median(values))
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    iqr = q75 - q25
    
    # Normalize
    denominator = max(iqr, epsilon)
    normalized = (values - median) / denominator
    
    # Clip to [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    
    params = {
        'median': median,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'epsilon': epsilon
    }
    
    return normalized, params


# ============================================================================
# DIAGNOSTIC & VALIDATION
# ============================================================================

def validate_modular_reduction(
    S: Sequence[int],
    moduli: np.ndarray,
    P: np.ndarray
) -> bool:
    """
    Validate that P = modular_reduce(S, moduli).
    
    Checks correctness of modular reduction operation.
    
    Args:
        S: Original sequence
        moduli: Moduli array
        P: Result array
    
    Returns:
        True if P[i] == S[i] mod moduli[i] for all i
    """
    if len(S) != len(moduli) or len(S) != len(P):
        return False
    
    for i in range(len(S)):
        expected = int(S[i]) % int(moduli[i])
        if P[i] != expected:
            return False
    
    return True


def validate_modular_pow(
    S: Sequence[int],
    exponents: Sequence[int],
    moduli: Sequence[int],
    result: np.ndarray
) -> bool:
    """
    Validate that result[i] == pow(S[i], exponents[i], moduli[i]).
    
    Args:
        S: Base values
        exponents: Exponents
        moduli: Moduli
        result: Result array
    
    Returns:
        True if all exponentiation results correct
    """
    if not (len(S) == len(exponents) == len(moduli) == len(result)):
        return False
    
    for i in range(len(S)):
        expected = pow(int(S[i]), int(exponents[i]), int(moduli[i]))
        if result[i] != expected:
            return False
    
    return True
