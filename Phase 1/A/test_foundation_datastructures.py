"""
test_foundation_datastructures.py

Comprehensive tests for utils_datastructures.py module.

Tests cover:
- Gene serialization roundtrip
- Population creation and statistics
- Lineage tracking and queries
- Deterministic initialization with seeds
- Edge cases and error handling
"""

import pytest
import numpy as np
from datetime import datetime
from utils_datastructures import (
    Gene, Population, LineageGraph, FitnessRecord, LineageRecord,
    ConsensusGroup, GenerationSnapshot, OperationType,
    create_initial_population, estimate_population_size
)


# ============================================================================
# GENE TESTS
# ============================================================================

class TestGene:
    """Tests for Gene dataclass."""
    
    def test_gene_initialization_valid(self):
        """Test valid gene creation."""
        coords = np.array([5, 10, 15], dtype=np.int64)
        moduli = np.array([7, 11, 19], dtype=np.int64)
        
        gene = Gene(id=1, coords=coords, moduli=moduli, fitness=0.75, generation=1)
        
        assert gene.id == 1
        assert np.array_equal(gene.coords, coords)
        assert np.array_equal(gene.moduli, moduli)
        assert gene.fitness == 0.75
        assert gene.generation == 1
    
    def test_gene_initialization_invalid_coords_dtype(self):
        """Test that non-int64 coords dtype raises error."""
        coords = np.array([5, 10, 15], dtype=np.int32)  # Wrong dtype
        moduli = np.array([7, 11, 19], dtype=np.int64)
        
        with pytest.raises(ValueError, match="coords dtype must be int64"):
            Gene(id=1, coords=coords, moduli=moduli)
    
    def test_gene_initialization_invalid_moduli_dtype(self):
        """Test that non-int64 moduli dtype raises error."""
        coords = np.array([5, 10, 15], dtype=np.int64)
        moduli = np.array([7, 11, 19], dtype=np.float64)  # Wrong dtype
        
        with pytest.raises(ValueError, match="moduli dtype must be int64"):
            Gene(id=1, coords=coords, moduli=moduli)
    
    def test_gene_initialization_moduli_not_greater_than_one(self):
        """Test that moduli must be > 1."""
        coords = np.array([5, 10, 15], dtype=np.int64)
        moduli = np.array([7, 1, 19], dtype=np.int64)  # 1 is invalid
        
        with pytest.raises(ValueError, match="All moduli must be > 1"):
            Gene(id=1, coords=coords, moduli=moduli)
    
    def test_gene_dimension(self):
        """Test gene.dimension() method."""
        for d in [1, 16, 32, 128]:
            gene = Gene(
                id=1,
                coords=np.zeros(d, dtype=np.int64),
                moduli=np.full(d, 7, dtype=np.int64)
            )
            assert gene.dimension() == d
    
    def test_gene_copy(self):
        """Test gene.copy() creates independent copy."""
        coords = np.array([5, 10, 15], dtype=np.int64)
        moduli = np.array([7, 11, 19], dtype=np.int64)
        gene = Gene(id=1, coords=coords, moduli=moduli, fitness=0.75)
        
        copy = gene.copy(new_id=2, new_generation=1, new_metadata={'parent': 1})
        
        # Check new ID and generation
        assert copy.id == 2
        assert copy.generation == 1
        
        # Check data independence
        assert np.array_equal(copy.coords, gene.coords)
        copy.coords[0] = 999
        assert gene.coords[0] == 5  # Original unchanged
    
    def test_gene_serialization_roundtrip(self):
        """Test Gene.to_dict() and from_dict() roundtrip."""
        coords = np.array([5, 10, 15], dtype=np.int64)
        moduli = np.array([7, 11, 19], dtype=np.int64)
        metadata = {
            'origin': 'initialization',
            'parent_ids': [1, 2],
            'operation': 'crossover'
        }
        
        gene_orig = Gene(
            id=42,
            coords=coords,
            moduli=moduli,
            fitness=0.85,
            generation=10,
            metadata=metadata
        )
        
        # Serialize and deserialize
        gene_dict = gene_orig.to_dict()
        gene_restored = Gene.from_dict(gene_dict)
        
        # Verify equality
        assert gene_restored.id == gene_orig.id
        assert np.array_equal(gene_restored.coords, gene_orig.coords)
        assert np.array_equal(gene_restored.moduli, gene_orig.moduli)
        assert gene_restored.fitness == gene_orig.fitness
        assert gene_restored.generation == gene_orig.generation
        assert gene_restored.metadata == gene_orig.metadata
        
        # Second roundtrip should be identical
        gene_dict_2 = gene_restored.to_dict()
        assert gene_dict == gene_dict_2


# ============================================================================
# POPULATION TESTS
# ============================================================================

class TestPopulation:
    """Tests for Population class."""
    
    def test_population_initialization(self):
        """Test population initialization."""
        pop = Population()
        assert pop.size() == 0
        assert pop.generation == 0
        assert pop.max_id == 0
    
    def test_population_add_and_get(self):
        """Test adding and retrieving genes."""
        pop = Population()
        gene = Gene(
            id=1,
            coords=np.array([1, 2], dtype=np.int64),
            moduli=np.array([5, 7], dtype=np.int64)
        )
        
        pop.add(gene)
        assert pop.size() == 1
        
        retrieved = pop.get(1)
        assert retrieved.id == 1
        assert np.array_equal(retrieved.coords, gene.coords)
    
    def test_population_add_duplicate_id_raises(self):
        """Test that duplicate gene IDs raise error."""
        pop = Population()
        gene1 = Gene(id=1, coords=np.array([1], dtype=np.int64), 
                     moduli=np.array([5], dtype=np.int64))
        gene2 = Gene(id=1, coords=np.array([2], dtype=np.int64),
                     moduli=np.array([7], dtype=np.int64))
        
        pop.add(gene1)
        with pytest.raises(ValueError, match="already exists"):
            pop.add(gene2)
    
    def test_population_remove(self):
        """Test removing genes."""
        pop = Population()
        gene = Gene(id=1, coords=np.array([1], dtype=np.int64),
                    moduli=np.array([5], dtype=np.int64))
        pop.add(gene)
        
        removed = pop.remove(1)
        assert removed.id == 1
        assert pop.size() == 0
    
    def test_population_remove_nonexistent(self):
        """Test removing nonexistent gene raises error."""
        pop = Population()
        with pytest.raises(ValueError, match="not found"):
            pop.remove(999)
    
    def test_population_fitness_stats(self):
        """Test fitness statistics computation."""
        pop = Population()
        
        # Empty population
        stats = pop.fitness_stats()
        assert stats['mean'] == 0.0
        
        # Add genes with known fitness
        fitnesses = [0.1, 0.5, 0.9]
        for i, f in enumerate(fitnesses, 1):
            gene = Gene(
                id=i,
                coords=np.array([i], dtype=np.int64),
                moduli=np.array([11], dtype=np.int64),
                fitness=f
            )
            pop.add(gene)
        
        stats = pop.fitness_stats()
        assert stats['mean'] == pytest.approx(0.5)
        assert stats['median'] == pytest.approx(0.5)
        assert stats['min'] == pytest.approx(0.1)
        assert stats['max'] == pytest.approx(0.9)
    
    def test_population_sorted_by_fitness(self):
        """Test sorting by fitness."""
        pop = Population()
        fitnesses = [0.3, 0.1, 0.9, 0.5]
        
        for i, f in enumerate(fitnesses, 1):
            gene = Gene(
                id=i,
                coords=np.array([i], dtype=np.int64),
                moduli=np.array([11], dtype=np.int64),
                fitness=f
            )
            pop.add(gene)
        
        # Descending (highest first)
        sorted_desc = pop.sorted_by_fitness(ascending=False)
        assert [g.fitness for g in sorted_desc] == [0.9, 0.5, 0.3, 0.1]
        
        # Ascending (lowest first)
        sorted_asc = pop.sorted_by_fitness(ascending=True)
        assert [g.fitness for g in sorted_asc] == [0.1, 0.3, 0.5, 0.9]
    
    def test_population_top_k_bottom_k(self):
        """Test top-k and bottom-k retrieval."""
        pop = Population()
        for i in range(10):
            gene = Gene(
                id=i,
                coords=np.array([i], dtype=np.int64),
                moduli=np.array([11], dtype=np.int64),
                fitness=i / 10.0
            )
            pop.add(gene)
        
        top_3 = pop.top_k(3)
        assert len(top_3) == 3
        assert top_3[0].fitness == 0.9
        assert top_3[1].fitness == 0.8
        assert top_3[2].fitness == 0.7
        
        bottom_3 = pop.bottom_k(3)
        assert len(bottom_3) == 3
        assert bottom_3[0].fitness == 0.0
        assert bottom_3[1].fitness == 0.1
        assert bottom_3[2].fitness == 0.2
    
    def test_population_next_id(self):
        """Test ID generation."""
        pop = Population()
        id_1 = pop.next_id()
        id_2 = pop.next_id()
        id_3 = pop.next_id()
        
        assert id_1 == 1
        assert id_2 == 2
        assert id_3 == 3
        assert pop.max_id == 3
    
    def test_population_clear(self):
        """Test clearing population."""
        pop = Population()
        for i in range(5):
            gene = Gene(
                id=i,
                coords=np.array([i], dtype=np.int64),
                moduli=np.array([11], dtype=np.int64)
            )
            pop.add(gene)
        
        assert pop.size() == 5
        pop.clear()
        assert pop.size() == 0


# ============================================================================
# LINEAGE GRAPH TESTS
# ============================================================================

class TestLineageGraph:
    """Tests for LineageGraph class."""
    
    def test_lineage_add_and_query(self):
        """Test adding edges and querying relationships."""
        graph = LineageGraph()
        
        record = LineageRecord(
            child_id=2,
            parent_ids=[0, 1],
            operation=OperationType.CROSSOVER,
            generation=1
        )
        
        graph.add_edge(record)
        
        # Query parents
        assert graph.get_parents(2) == [0, 1]
        
        # Query children
        assert 2 in graph.get_children(0)
        assert 2 in graph.get_children(1)
    
    def test_lineage_ancestors(self):
        """Test ancestor query."""
        graph = LineageGraph()
        
        # Create chain: 0 -> 1 -> 2 -> 3
        graph.add_edge(LineageRecord(child_id=1, parent_ids=[0], 
                                     operation=OperationType.ASEXUAL))
        graph.add_edge(LineageRecord(child_id=2, parent_ids=[1],
                                     operation=OperationType.ASEXUAL))
        graph.add_edge(LineageRecord(child_id=3, parent_ids=[2],
                                     operation=OperationType.ASEXUAL))
        
        ancestors_3 = graph.get_ancestors(3)
        assert set(ancestors_3) == {0, 1, 2}
        
        ancestors_2 = graph.get_ancestors(2)
        assert set(ancestors_2) == {0, 1}
    
    def test_lineage_descendants(self):
        """Test descendant query."""
        graph = LineageGraph()
        
        # Create tree: 0 -> 1 -> 2
        #             0 -> 3 -> 4
        graph.add_edge(LineageRecord(child_id=1, parent_ids=[0],
                                     operation=OperationType.ASEXUAL))
        graph.add_edge(LineageRecord(child_id=2, parent_ids=[1],
                                     operation=OperationType.ASEXUAL))
        graph.add_edge(LineageRecord(child_id=3, parent_ids=[0],
                                     operation=OperationType.ASEXUAL))
        graph.add_edge(LineageRecord(child_id=4, parent_ids=[3],
                                     operation=OperationType.ASEXUAL))
        
        descendants_0 = graph.get_descendants(0)
        assert set(descendants_0) == {1, 2, 3, 4}
        
        descendants_1 = graph.get_descendants(1)
        assert set(descendants_1) == {2}
    
    def test_lineage_serialization(self):
        """Test lineage graph serialization."""
        graph = LineageGraph()
        
        record1 = LineageRecord(
            child_id=2, parent_ids=[0, 1],
            operation=OperationType.CROSSOVER, generation=1
        )
        record2 = LineageRecord(
            child_id=3, parent_ids=[2],
            operation=OperationType.MUTATION, generation=2
        )
        
        graph.add_edge(record1)
        graph.add_edge(record2)
        
        # Serialize and deserialize
        edge_list = graph.to_list()
        graph_restored = LineageGraph.from_list(edge_list)
        
        assert graph_restored.num_edges() == 2
        assert graph_restored.get_parents(2) == [0, 1]
        assert graph_restored.get_parents(3) == [2]


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_initial_population_deterministic(self):
        """Test that same seed produces identical populations."""
        moduli_pool = [2, 3, 5, 7, 11, 13]
        seed = 12345
        
        pop1 = create_initial_population(20, 16, moduli_pool, seed=seed)
        pop2 = create_initial_population(20, 16, moduli_pool, seed=seed)
        
        # Check size
        assert pop1.size() == 20
        assert pop2.size() == 20
        
        # Check dimension
        for gene in pop1.get_all():
            assert gene.dimension() == 16
        
        # Check reproducibility: same seed -> same genes
        genes1 = pop1.sorted_by_fitness(ascending=True)  # Sort for consistent order
        genes2 = pop2.sorted_by_fitness(ascending=True)
        
        # Compare first gene (and a few more)
        for g1, g2 in zip(genes1[:3], genes2[:3]):
            assert np.array_equal(g1.coords, g2.coords)
            assert np.array_equal(g1.moduli, g2.moduli)
    
    def test_create_initial_population_size(self):
        """Test population size parameter."""
        for N in [10, 50, 100]:
            pop = create_initial_population(N, 16, [2, 3, 5, 7])
            assert pop.size() == N
    
    def test_create_initial_population_dimension(self):
        """Test dimension parameter."""
        for d in [8, 16, 32, 64]:
            pop = create_initial_population(10, d, [2, 3, 5, 7])
            for gene in pop.get_all():
                assert gene.dimension() == d
    
    def test_create_initial_population_invalid_moduli_pool(self):
        """Test error handling for empty moduli pool."""
        with pytest.raises(ValueError, match="moduli_pool cannot be empty"):
            create_initial_population(10, 16, [])
    
    def test_estimate_population_size(self):
        """Test population size estimation."""
        assert estimate_population_size('easy') == 100
        assert estimate_population_size('medium') == 200
        assert estimate_population_size('hard') == 400
        assert estimate_population_size('unknown') == 200  # Default


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple components."""
    
    def test_workflow_create_evaluate_track(self):
        """Test workflow: create pop, evaluate, track lineage."""
        # Create initial population
        pop = create_initial_population(10, 16, [2, 3, 5, 7, 11], seed=42)
        
        # Assign fitness
        for i, gene in enumerate(pop.get_all()):
            gene.fitness = i / 10.0
        
        # Track lineage
        lineage = LineageGraph()
        
        # Simulate crossover
        parent1 = pop.get(1)
        parent2 = pop.get(2)
        child_id = pop.next_id()
        
        record = LineageRecord(
            child_id=child_id,
            parent_ids=[parent1.id, parent2.id],
            operation=OperationType.CROSSOVER,
            generation=1,
            parent_coords=[parent1.coords, parent2.coords],
            parent_moduli=[parent1.moduli, parent2.moduli]
        )
        lineage.add_edge(record)
        
        # Verify
        assert lineage.get_parents(child_id) == [1, 2]
        assert pop.max_id >= child_id
    
    def test_snapshot_creation_and_restoration(self):
        """Test generation snapshot creation and restoration."""
        # Create population and lineage
        pop = create_initial_population(5, 8, [2, 3, 5], seed=99)
        lineage = LineageGraph()
        
        # Add some fitness and lineage
        for i, gene in enumerate(pop.get_all()):
            gene.fitness = i / 5.0
        
        # Create snapshot
        snapshot = GenerationSnapshot(
            generation=1,
            population=[g.to_dict() for g in pop.get_all()],
            lineage_edges=lineage.to_list(),
            consensus_groups=[],
            fitness_stats=pop.fitness_stats(),
            metadata={'config': 'test', 'seed': 99}
        )
        
        # Serialize and deserialize
        snapshot_dict = snapshot.to_dict()
        snapshot_restored = GenerationSnapshot.from_dict(snapshot_dict)
        
        # Verify
        assert snapshot_restored.generation == 1
        assert len(snapshot_restored.population) == 5
        assert snapshot_restored.fitness_stats['mean'] == pytest.approx(0.4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
