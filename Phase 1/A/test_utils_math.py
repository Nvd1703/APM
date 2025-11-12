"""
test_utils_math.py

Comprehensive tests for utils_math.py module.

Tests cover:
- Safe matrix-vector multiplication (no overflow)
- Modular vector operations
- Modular exponentiation
- Cyclic distance properties
- Batch residue computation
- Robust scaling
"""

import pytest
import numpy as np
from utils_math import (
    safe_matmul_int, modular_vector_reduce, modular_pow_vector,
    cyclic_distance, batch_residue_matrix, robust_scale,
    validate_modular_reduction, validate_modular_pow
)


# ============================================================================
# MATRIX MULTIPLICATION TESTS
# ============================================================================

class TestSafeMatmulInt:
    """Tests for safe_matmul_int function."""
    
    def test_safe_matmul_basic(self):
        """Test basic matrix-vector multiplication."""
        A = np.array([[1, 2], [3, 4]], dtype=int)
        V = np.array([5, 7], dtype=int)
        
        S = safe_matmul_int(A, V)
        
        # S[0] = 1*5 + 2*7 = 5 + 14 = 19
        # S[1] = 3*5 + 4*7 = 15 + 28 = 43
        assert S[0] == 19
        assert S[1] == 43
    
    def test_safe_matmul_large_values(self):
        """Test that large values don't overflow."""
        # Create values that would overflow int64
        A = np.array([[2**40, 2**40]], dtype=int)
        V = np.array([2**40, 2**40], dtype=int)
        
        S = safe_matmul_int(A, V)
        
        # S[0] = 2^40 * 2^40 + 2^40 * 2^40 = 2 * 2^80
        expected = 2 * (2**80)
        assert S[0] == expected
    
    def test_safe_matmul_single_element(self):
        """Test 1x1 matrix."""
        A = np.array([[5]], dtype=int)
        V = np.array([3], dtype=int)
        
        S = safe_matmul_int(A, V)
        
        assert S[0] == 15
    
    def test_safe_matmul_zero_result(self):
        """Test multiplication resulting in zero."""
        A = np.array([[0, 0], [0, 0]], dtype=int)
        V = np.array([5, 7], dtype=int)
        
        S = safe_matmul_int(A, V)
        
        assert S[0] == 0
        assert S[1] == 0
    
    def test_safe_matmul_negative_values(self):
        """Test with negative values."""
        A = np.array([[-1, 2], [3, -4]], dtype=int)
        V = np.array([5, 7], dtype=int)
        
        S = safe_matmul_int(A, V)
        
        # S[0] = -1*5 + 2*7 = -5 + 14 = 9
        # S[1] = 3*5 + -4*7 = 15 - 28 = -13
        assert S[0] == 9
        assert S[1] == -13
    
    def test_safe_matmul_incompatible_shapes(self):
        """Test error handling for incompatible shapes."""
        A = np.array([[1, 2], [3, 4]], dtype=int)  # 2x2
        V = np.array([5, 7, 9], dtype=int)  # 3-vector
        
        with pytest.raises(ValueError, match="Incompatible shapes"):
            safe_matmul_int(A, V)
    
    def test_safe_matmul_rectangular(self):
        """Test non-square matrices."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)  # 2x3
        V = np.array([1, 2, 3], dtype=int)  # 3-vector
        
        S = safe_matmul_int(A, V)
        
        # S[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        # S[1] = 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
        assert S[0] == 14
        assert S[1] == 32
    
    def test_safe_matmul_returns_python_int(self):
        """Test that result contains Python ints (unlimited precision)."""
        A = np.array([[1]], dtype=np.int64)
        V = np.array([1], dtype=np.int64)
        
        S = safe_matmul_int(A, V)
        
        assert isinstance(S[0], int)


# ============================================================================
# MODULAR VECTOR REDUCTION TESTS
# ============================================================================

class TestModularVectorReduce:
    """Tests for modular_vector_reduce function."""
    
    def test_modular_reduce_basic(self):
        """Test basic modular reduction."""
        S = [19, 43, 100]
        moduli = np.array([7, 11, 13], dtype=np.int64)
        
        P = modular_vector_reduce(S, moduli)
        
        assert P[0] == 19 % 7  # 5
        assert P[1] == 43 % 11  # 10
        assert P[2] == 100 % 13  # 9
    
    def test_modular_reduce_zero(self):
        """Test reduction of zero."""
        S = [0, 0, 0]
        moduli = np.array([7, 11, 13], dtype=np.int64)
        
        P = modular_vector_reduce(S, moduli)
        
        assert P[0] == 0
        assert P[1] == 0
        assert P[2] == 0
    
    def test_modular_reduce_exact_modulus(self):
        """Test when value equals modulus."""
        S = [7, 11, 13]
        moduli = np.array([7, 11, 13], dtype=np.int64)
        
        P = modular_vector_reduce(S, moduli)
        
        assert P[0] == 0
        assert P[1] == 0
        assert P[2] == 0
    
    def test_modular_reduce_negative(self):
        """Test reduction of negative values (Python % behavior)."""
        S = [-5, -1, -100]
        moduli = np.array([7, 11, 13], dtype=np.int64)
        
        P = modular_vector_reduce(S, moduli)
        
        # Python's % with negative dividend: -5 % 7 = 2
        assert P[0] == 2
        assert P[1] == 10
        assert P[2] == 9
    
    def test_modular_reduce_dtype(self):
        """Test output dtype is int64."""
        S = [1, 2, 3]
        moduli = np.array([5, 7, 11], dtype=np.int64)
        
        P = modular_vector_reduce(S, moduli)
        
        assert P.dtype == np.int64
    
    def test_modular_reduce_length_mismatch(self):
        """Test error when lengths don't match."""
        S = [1, 2, 3]
        moduli = np.array([5, 7], dtype=np.int64)  # Wrong length
        
        with pytest.raises(ValueError, match="different lengths"):
            modular_vector_reduce(S, moduli)
    
    def test_modular_reduce_invalid_modulus(self):
        """Test error with zero or negative modulus."""
        S = [1, 2]
        moduli = np.array([5, 0], dtype=np.int64)  # 0 is invalid
        
        with pytest.raises(ValueError, match="must be > 0"):
            modular_vector_reduce(S, moduli)


# ============================================================================
# MODULAR EXPONENTIATION TESTS
# ============================================================================

class TestModularPowVector:
    """Tests for modular_pow_vector function."""
    
    def test_modular_pow_basic(self):
        """Test basic modular exponentiation."""
        S = [2, 3, 5]
        exponents = [3, 2, 2]
        moduli = [7, 11, 13]
        
        result = modular_pow_vector(S, exponents, moduli)
        
        # 2^3 mod 7 = 8 mod 7 = 1
        # 3^2 mod 11 = 9 mod 11 = 9
        # 5^2 mod 13 = 25 mod 13 = 12
        assert result[0] == 1
        assert result[1] == 9
        assert result[2] == 12
    
    def test_modular_pow_zero_exponent(self):
        """Test with exponent 0 (should give 1)."""
        S = [5, 7, 11]
        exponents = [0, 0, 0]
        moduli = [13, 17, 19]
        
        result = modular_pow_vector(S, exponents, moduli)
        
        # Anything^0 = 1 (even 0^0 in Python's pow)
        assert result[0] == 1
        assert result[1] == 1
        assert result[2] == 1
    
    def test_modular_pow_one_exponent(self):
        """Test with exponent 1."""
        S = [5, 7, 11]
        exponents = [1, 1, 1]
        moduli = [13, 17, 19]
        
        result = modular_pow_vector(S, exponents, moduli)
        
        assert result[0] == 5
        assert result[1] == 7
        assert result[2] == 11
    
    def test_modular_pow_large_exponent(self):
        """Test with large exponent (efficiency of pow with mod)."""
        S = [2]
        exponents = [1000000]
        moduli = [1000000007]  # Large prime
        
        result = modular_pow_vector(S, exponents, moduli)
        
        # Should complete quickly due to built-in optimization
        expected = pow(2, 1000000, 1000000007)
        assert result[0] == expected
    
    def test_modular_pow_dtype(self):
        """Test output dtype is int64."""
        S = [2, 3]
        exponents = [2, 2]
        moduli = [7, 11]
        
        result = modular_pow_vector(S, exponents, moduli)
        
        assert result.dtype == np.int64
    
    def test_modular_pow_matches_scalar(self):
        """Test that vectorized matches scalar pow."""
        S = [2, 3, 5, 7]
        exponents = [3, 2, 4, 5]
        moduli = [11, 13, 17, 19]
        
        result = modular_pow_vector(S, exponents, moduli)
        
        for i in range(len(S)):
            expected = pow(S[i], exponents[i], moduli[i])
            assert result[i] == expected
    
    def test_modular_pow_length_mismatch(self):
        """Test error when lengths don't match."""
        S = [2, 3]
        exponents = [2, 2, 2]  # Wrong length
        moduli = [7, 11]
        
        with pytest.raises(ValueError, match="must have same length"):
            modular_pow_vector(S, exponents, moduli)


# ============================================================================
# CYCLIC DISTANCE TESTS
# ============================================================================

class TestCyclicDistance:
    """Tests for cyclic_distance function."""
    
    def test_cyclic_distance_direct(self):
        """Test direct distance (shorter than wraparound)."""
        # Distance 0->3 in Z_7: min(3, 7-3) = min(3, 4) = 3
        dist = cyclic_distance(0, 3, 7)
        assert dist == 3
    
    def test_cyclic_distance_wraparound(self):
        """Test wraparound distance (shorter than direct)."""
        # Distance 1->6 in Z_7: min(5, 7-5) = min(5, 2) = 2
        dist = cyclic_distance(1, 6, 7)
        assert dist == 2
    
    def test_cyclic_distance_identity(self):
        """Test distance from element to itself."""
        for m in [5, 7, 11, 100]:
            for a in [0, 1, m//2]:
                dist = cyclic_distance(a, a, m)
                assert dist == 0
    
    def test_cyclic_distance_symmetric(self):
        """Test symmetry property: dist(a,b) == dist(b,a)."""
        for a in range(7):
            for b in range(7):
                dist_ab = cyclic_distance(a, b, 7)
                dist_ba = cyclic_distance(b, a, 7)
                assert dist_ab == dist_ba
    
    def test_cyclic_distance_bounded(self):
        """Test that distance is bounded by floor(m/2)."""
        m = 13
        max_dist = m // 2
        
        for a in range(m):
            for b in range(m):
                dist = cyclic_distance(a, b, m)
                assert 0 <= dist <= max_dist
    
    def test_cyclic_distance_opposite_sides(self):
        """Test distance when elements are opposite (m/2 apart)."""
        # In Z_10: 0 and 5 are opposite, distance should be 5
        dist = cyclic_distance(0, 5, 10)
        assert dist == 5
        
        # In Z_11: no exact opposite, 0 and 5 have distance min(5, 6) = 5
        dist = cyclic_distance(0, 5, 11)
        assert dist == 5
    
    def test_cyclic_distance_modulo_reduction(self):
        """Test that inputs are reduced modulo m."""
        # cyclic_distance(8, 3, 7) should be same as cyclic_distance(1, 3, 7)
        # since 8 ≡ 1 (mod 7)
        dist1 = cyclic_distance(8, 3, 7)
        dist2 = cyclic_distance(1, 3, 7)
        assert dist1 == dist2


# ============================================================================
# BATCH RESIDUE MATRIX TESTS
# ============================================================================

class TestBatchResidueMatrix:
    """Tests for batch_residue_matrix function."""
    
    def test_batch_residue_basic(self):
        """Test basic batch residue computation."""
        coords = np.array([[5, 10], [15, 20]], dtype=np.int64)
        modulus = 7
        
        R = batch_residue_matrix(coords, modulus)
        
        assert R[0, 0] == 5 % 7  # 5
        assert R[0, 1] == 10 % 7  # 3
        assert R[1, 0] == 15 % 7  # 1
        assert R[1, 1] == 20 % 7  # 6
    
    def test_batch_residue_dtype(self):
        """Test output dtype is int64."""
        coords = np.array([[1, 2], [3, 4]], dtype=np.int64)
        R = batch_residue_matrix(coords, 7)
        
        assert R.dtype == np.int64
    
    def test_batch_residue_single_gene(self):
        """Test with single gene."""
        coords = np.array([[5, 10, 15]], dtype=np.int64)
        modulus = 7
        
        R = batch_residue_matrix(coords, modulus)
        
        assert R.shape == (1, 3)
        assert R[0, 0] == 5 % 7
    
    def test_batch_residue_single_coord(self):
        """Test with single coordinate."""
        coords = np.array([[5], [10], [15]], dtype=np.int64)
        modulus = 7
        
        R = batch_residue_matrix(coords, modulus)
        
        assert R.shape == (3, 1)
        assert R[0, 0] == 5 % 7
        assert R[1, 0] == 10 % 7
        assert R[2, 0] == 15 % 7
    
    def test_batch_residue_blocksize(self):
        """Test that blocksize parameter works."""
        coords = np.array([[i, i+1] for i in range(10)], dtype=np.int64)
        modulus = 7
        
        # Test different blocksizes
        for blocksize in [1, 3, 5, 10, 100]:
            R = batch_residue_matrix(coords, modulus, blocksize=blocksize)
            assert R.shape == (10, 2)
            assert np.all(R < modulus)


# ============================================================================
# ROBUST SCALING TESTS
# ============================================================================

class TestRobustScale:
    """Tests for robust_scale function."""
    
    def test_robust_scale_basic(self):
        """Test basic robust scaling."""
        values = np.array([0.1, 0.2, 0.3, 0.5, 0.9], dtype=float)
        
        normalized, params = robust_scale(values)
        
        # Check output range
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        
        # Check params exist
        assert 'median' in params
        assert 'iqr' in params
        assert 'q25' in params
        assert 'q75' in params
    
    def test_robust_scale_constant_input(self):
        """Test scaling of constant values."""
        values = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        
        normalized, params = robust_scale(values)
        
        # When all values are equal, IQR = 0, should get clipped result
        assert np.all((normalized == 0.0) | (normalized == 1.0))
    
    def test_robust_scale_two_values(self):
        """Test scaling with two distinct values."""
        values = np.array([0.0, 1.0], dtype=float)
        
        normalized, params = robust_scale(values)
        
        # Should produce valid output
        assert len(normalized) == 2
        assert 0.0 <= np.min(normalized) <= np.max(normalized) <= 1.0
    
    def test_robust_scale_median_centered(self):
        """Test that scaling centers on median."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        
        _, params = robust_scale(values)
        
        assert params['median'] == 3.0
    
    def test_robust_scale_large_range(self):
        """Test scaling values with large range."""
        values = np.array([1e-10, 1e10], dtype=float)
        
        normalized, params = robust_scale(values)
        
        # Should still produce valid [0, 1] output
        assert 0.0 <= np.min(normalized) <= 1.0
        assert 0.0 <= np.max(normalized) <= 1.0
    
    def test_robust_scale_negative_values(self):
        """Test with negative values."""
        values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=float)
        
        normalized, params = robust_scale(values)
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidationFunctions:
    """Tests for validation helper functions."""
    
    def test_validate_modular_reduction_correct(self):
        """Test validation of correct reduction."""
        S = [19, 43, 100]
        moduli = np.array([7, 11, 13], dtype=np.int64)
        P = np.array([5, 10, 9], dtype=np.int64)
        
        assert validate_modular_reduction(S, moduli, P)
    
    def test_validate_modular_reduction_incorrect(self):
        """Test validation detects incorrect reduction."""
        S = [19, 43]
        moduli = np.array([7, 11], dtype=np.int64)
        P = np.array([5, 9], dtype=np.int64)  # 43 % 11 = 10, not 9
        
        assert not validate_modular_reduction(S, moduli, P)
    
    def test_validate_modular_pow_correct(self):
        """Test validation of correct exponentiation."""
        S = [2, 3, 5]
        exponents = [3, 2, 2]
        moduli = [7, 11, 13]
        result = np.array([1, 9, 12], dtype=np.int64)
        
        assert validate_modular_pow(S, exponents, moduli, result)
    
    def test_validate_modular_pow_incorrect(self):
        """Test validation detects incorrect exponentiation."""
        S = [2, 3]
        exponents = [3, 2]
        moduli = [7, 11]
        result = np.array([1, 8], dtype=np.int64)  # 3^2 mod 11 = 9, not 8
        
        assert not validate_modular_pow(S, exponents, moduli, result)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipelines."""
    
    def test_gene_construction_pipeline(self):
        """Test complete gene construction: A·V -> pow -> mod."""
        # Setup
        A = np.array([[1, 2], [3, 4]], dtype=int)
        V = np.array([5, 7], dtype=int)
        exponents = np.array([2, 2], dtype=np.int64)
        moduli = np.array([7, 11], dtype=np.int64)
        
        # Pipeline
        S = safe_matmul_int(A, V)  # [19, 43]
        S_powered = modular_pow_vector(S, exponents, moduli)  # pow then mod
        P = modular_vector_reduce(S_powered, moduli)
        
        # Verify
        assert len(P) == 2
        assert all(P[i] < moduli[i] for i in range(2))
    
    def test_similarity_computation_pipeline(self):
        """Test computing similarity via cyclic distance."""
        # Two genes
        P = np.array([1, 2, 3], dtype=np.int64)
        Q = np.array([2, 3, 4], dtype=np.int64)
        modulus = 7
        
        # Compute pairwise cyclic distances
        distances = []
        for i in range(len(P)):
            d = cyclic_distance(P[i], Q[i], modulus)
            distances.append(d)
        
        # Similarity from distance
        max_dist = modulus // 2
        similarities = [1.0 - d / max_dist for d in distances]
        
        # Average similarity
        avg_similarity = np.mean(similarities)
        
        assert 0.0 <= avg_similarity <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
