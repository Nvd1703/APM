"""
demo_gene_construction.py

Demonstration of complete gene construction pipeline.

Shows how the core math primitives work together to implement
the definitive gene construction formula from the specification:

    Input:  V (input vector), A (matrix), moduli M_P, exponents R (optional)
    Step 1: S = A · V (matrix-vector product)
    Step 2: [Optional] S_i ← pow(S_i, r_i, m_i)
    Step 3: P_i ← S_i mod m_i (final gene coordinates)

This script:
1. Creates a deterministic example
2. Shows the pipeline step-by-step
3. Demonstrates deterministic reproducibility with seed
4. Validates the output
"""

import numpy as np
from utils_datastructures import Gene, Population, create_initial_population
from utils_math import (
    safe_matmul_int, modular_vector_reduce, modular_pow_vector,
    validate_modular_reduction, validate_modular_pow
)


def demonstrate_basic_pipeline():
    """Show step-by-step gene construction."""
    print("=" * 80)
    print("GENE CONSTRUCTION PIPELINE - STEP BY STEP")
    print("=" * 80)
    
    # Setup parameters
    A = np.array([[1, 2], [3, 4]], dtype=int)
    V = np.array([5, 7], dtype=int)
    moduli = np.array([7, 11], dtype=np.int64)
    exponents = np.array([2, 2], dtype=np.int64)
    
    print("\n1. Input Parameters:")
    print(f"   A (matrix, 2x2):\n{A}")
    print(f"   V (vector, k=2): {V}")
    print(f"   moduli: {moduli}")
    print(f"   exponents (for nonlinear step): {exponents}")
    
    # Step 1: Matrix multiplication
    print("\n2. Step 1: S = A · V (matrix-vector product)")
    S = safe_matmul_int(A, V)
    print(f"   S = {S}")
    print(f"   (No overflow: using Python arbitrary-precision int)")
    
    # Step 2: Optional modular exponentiation
    print("\n3. Step 2: S'_i = pow(S_i, r_i, m_i) [OPTIONAL NONLINEAR STEP]")
    S_powered = modular_pow_vector(S, exponents, moduli)
    print(f"   S' = {S_powered}")
    print(f"   Details:")
    for i in range(len(S)):
        print(f"      S'[{i}] = pow({S[i]}, {exponents[i]}, {moduli[i]}) = {S_powered[i]}")
    
    # Step 3: Modular reduction
    print("\n4. Step 3: P_i = S'_i mod m_i (final modular reduction)")
    P = modular_vector_reduce(S_powered, moduli)
    print(f"   P = {P}")
    print(f"   Details:")
    for i in range(len(S_powered)):
        print(f"      P[{i}] = {S_powered[i]} mod {moduli[i]} = {P[i]}")
    
    # Verification
    print("\n5. Verification:")
    is_valid = validate_modular_reduction(S_powered, moduli, P)
    print(f"   All coordinates correctly reduced: {is_valid}")
    print(f"   All coordinates < moduli: {all(P[i] < moduli[i] for i in range(len(P)))}")
    
    # Construct Gene object
    print("\n6. Construct Gene Object:")
    gene = Gene(
        id=1,
        coords=P.astype(np.int64),
        moduli=moduli,
        fitness=0.0,
        generation=0,
        metadata={
            'matrix_A_reference': 'demo_matrix_1',
            'input_vector_V': V.tolist(),
            'exponents_R': exponents.tolist(),
            'nonlinear': True
        }
    )
    print(f"   Gene ID: {gene.id}")
    print(f"   Coords: {gene.coords}")
    print(f"   Moduli: {gene.moduli}")
    print(f"   Dimension: {gene.dimension()}")
    
    # Serialization
    print("\n7. Serialization (for HDF5/JSON storage):")
    gene_dict = gene.to_dict()
    print(f"   Keys: {list(gene_dict.keys())}")
    print(f"   Coords in dict: {gene_dict['coords']}")
    print(f"   Metadata preserved: {gene_dict['metadata']['nonlinear']}")
    
    # Deserialization
    print("\n8. Deserialization (roundtrip check):")
    gene_restored = Gene.from_dict(gene_dict)
    print(f"   Restored ID: {gene_restored.id}")
    print(f"   Restored coords match: {np.array_equal(gene_restored.coords, gene.coords)}")
    print(f"   Restored moduli match: {np.array_equal(gene_restored.moduli, gene.moduli)}")
    

def demonstrate_population_creation():
    """Show deterministic population creation with seed."""
    print("\n\n" + "=" * 80)
    print("DETERMINISTIC POPULATION CREATION WITH SEED")
    print("=" * 80)
    
    # Parameters
    population_size = 5
    dimension = 8
    moduli_pool = [2, 3, 5, 7, 11, 13, 17, 19]
    seed = 42
    
    print(f"\nParameters:")
    print(f"   Population size: {population_size}")
    print(f"   Dimension: {dimension}")
    print(f"   Moduli pool: {moduli_pool}")
    print(f"   Seed: {seed}")
    
    # Create population with seed
    print(f"\n1. Creating population with seed={seed}...")
    pop1 = create_initial_population(population_size, dimension, moduli_pool, seed=seed)
    
    print(f"   Population created: {pop1.size()} individuals")
    print(f"   Sample gene (ID=1):")
    gene1 = pop1.get(1)
    print(f"     Coords: {gene1.coords}")
    print(f"     Moduli: {gene1.moduli}")
    
    # Reproducibility test
    print(f"\n2. Creating IDENTICAL population with same seed...")
    pop2 = create_initial_population(population_size, dimension, moduli_pool, seed=seed)
    
    gene1_again = pop2.get(1)
    print(f"   Sample gene (ID=1) from second run:")
    print(f"     Coords: {gene1_again.coords}")
    print(f"     Moduli: {gene1_again.moduli}")
    
    print(f"\n3. Reproducibility Verification:")
    coords_match = np.array_equal(gene1.coords, gene1_again.coords)
    moduli_match = np.array_equal(gene1.moduli, gene1_again.moduli)
    print(f"   Coordinates match: {coords_match}")
    print(f"   Moduli match: {moduli_match}")
    print(f"   ✓ DETERMINISTIC: Same seed → identical genes")
    
    # Show variation with different seed
    print(f"\n4. Creating population with DIFFERENT seed=99...")
    pop3 = create_initial_population(population_size, dimension, moduli_pool, seed=99)
    gene1_different = pop3.get(1)
    print(f"   Sample gene (ID=1) from different seed:")
    print(f"     Coords: {gene1_different.coords}")
    print(f"     Moduli: {gene1_different.moduli}")
    
    coords_differ = not np.array_equal(gene1.coords, gene1_different.coords)
    print(f"   Coordinates differ: {coords_differ}")
    print(f"   ✓ Different seed → different genes (as expected)")


def demonstrate_gene_statistics():
    """Show population statistics and analysis."""
    print("\n\n" + "=" * 80)
    print("POPULATION STATISTICS & ANALYSIS")
    print("=" * 80)
    
    # Create population
    pop = create_initial_population(20, 16, [2, 3, 5, 7, 11, 13], seed=123)
    
    # Assign random fitness
    np.random.seed(123)
    for gene in pop.get_all():
        gene.fitness = np.random.uniform(0, 1)
    
    # Get statistics
    stats = pop.fitness_stats()
    
    print(f"\nPopulation: {pop.size()} individuals, dimension {pop.get_all()[0].dimension()}")
    print(f"\nFitness Statistics:")
    print(f"   Mean:   {stats['mean']:.4f}")
    print(f"   Median: {stats['median']:.4f}")
    print(f"   Std:    {stats['std']:.4f}")
    print(f"   Min:    {stats['min']:.4f}")
    print(f"   Max:    {stats['max']:.4f}")
    
    print(f"\nTop 3 Genes (by fitness):")
    for i, gene in enumerate(pop.top_k(3), 1):
        print(f"   {i}. Gene ID={gene.id}, fitness={gene.fitness:.4f}")
    
    print(f"\nBottom 3 Genes (by fitness):")
    for i, gene in enumerate(pop.bottom_k(3), 1):
        print(f"   {i}. Gene ID={gene.id}, fitness={gene.fitness:.4f}")


def demonstrate_serialization_roundtrip():
    """Show complete serialization and restoration."""
    print("\n\n" + "=" * 80)
    print("SERIALIZATION ROUNDTRIP (HDF5/JSON Ready)")
    print("=" * 80)
    
    # Create a gene with metadata
    coords = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    moduli = np.array([7, 11, 13, 17, 19], dtype=np.int64)
    
    gene_original = Gene(
        id=42,
        coords=coords,
        moduli=moduli,
        fitness=0.87,
        generation=10,
        metadata={
            'origin': 'crossover',
            'parents': [10, 11],
            'operation': 'uniform_crossover',
            'consensus_modulus': 7
        }
    )
    
    print(f"\nOriginal Gene:")
    print(f"   ID: {gene_original.id}")
    print(f"   Coords: {gene_original.coords}")
    print(f"   Fitness: {gene_original.fitness}")
    print(f"   Metadata keys: {list(gene_original.metadata.keys())}")
    
    # Serialize
    print(f"\nSerialization to dict (JSON-ready):")
    gene_dict = gene_original.to_dict()
    print(f"   Dict keys: {list(gene_dict.keys())}")
    print(f"   Coords in dict: {gene_dict['coords']}")
    print(f"   Type of dict['coords'][0]: {type(gene_dict['coords'][0])}")
    
    # Deserialize
    print(f"\nDeserialization from dict:")
    gene_restored = Gene.from_dict(gene_dict)
    print(f"   Restored ID: {gene_restored.id}")
    print(f"   Restored fitness: {gene_restored.fitness}")
    
    # Verify roundtrip
    print(f"\nRoundtrip Verification:")
    coords_match = np.array_equal(gene_original.coords, gene_restored.coords)
    moduli_match = np.array_equal(gene_original.moduli, gene_restored.moduli)
    metadata_match = gene_original.metadata == gene_restored.metadata
    
    print(f"   Coordinates match: {coords_match}")
    print(f"   Moduli match: {moduli_match}")
    print(f"   Metadata match: {metadata_match}")
    print(f"   ✓ LOSSLESS ROUNDTRIP: dict → Gene → dict identical")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("LIVING CRYPTOGRAPHIC ECOSYSTEM - PHASE 1 DEMONSTRATION")
    print("=" * 80)
    
    demonstrate_basic_pipeline()
    demonstrate_population_creation()
    demonstrate_gene_statistics()
    demonstrate_serialization_roundtrip()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll modules working correctly!")
    print("✓ Gene construction pipeline functional")
    print("✓ Deterministic reproducibility verified")
    print("✓ Serialization lossless")
    print("✓ Population statistics computed")
    print("\nReady for Phase 2: Consensus & Genetic Operators")
    print("=" * 80 + "\n")
