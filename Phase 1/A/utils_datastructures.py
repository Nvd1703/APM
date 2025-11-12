"""
utils_datastructures.py

Core data structures for Living Cryptographic Ecosystem.

This module defines the fundamental data structures used throughout the project:
- Gene: polynomial representation with coordinates, moduli, fitness, metadata
- Population: collection management with indexing, sorting, statistics
- LineageGraph: genealogy tracking as a directed acyclic graph (DAG)
- FitnessRecord: arena-based fitness tracking
- ConsensusGroup: interaction group representation
- GenerationSnapshot: complete generation serialization

All data structures are designed for:
1. Deterministic reproducibility (via metadata tracking)
2. HDF5 serialization (via to_dict/from_dict)
3. Scientific audit trails (via lineage tracking)
4. Type safety (full type hints)

Example:
    >>> pop = create_initial_population(200, 32, [2,3,5,7,11,13], seed=12345)
    >>> gene = pop.get_all()[0]
    >>> assert gene.dimension() == 32
    >>> assert all(m > 1 for m in gene.moduli)
    >>> gene_dict = gene.to_dict()
    >>> gene_restored = Gene.from_dict(gene_dict)
    >>> assert gene_restored.coords[0] == gene.coords[0]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
from datetime import datetime
import json


class OperationType(Enum):
    """Types of genetic operations."""
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    SELF_REPLICATION = "self_replication"
    ASEXUAL = "asexual"


@dataclass
class Gene:
    """
    Represents a single polynomial (gene) in the population.
    
    A gene is a d-dimensional vector of coordinates in modular arithmetic spaces.
    Each coordinate i lives in Z_{m_i}, where m_i is the modulus for that coordinate.
    
    Attributes:
        id: Unique identifier for this gene within the population
        coords: np.ndarray of shape (d,), dtype int64 - polynomial coefficients
        moduli: np.ndarray of shape (d,), dtype int64 - per-coordinate moduli (each > 1)
        fitness: float - current fitness score (higher is better), default 0.0
        generation: int - generation at which this gene was created
        metadata: dict - additional information (parents, operation, construction params, etc)
        experience: Optional[np.ndarray] - optional experience vector for grafting
    
    Constraints:
        - coords.dtype == np.int64
        - moduli.dtype == np.int64
        - len(coords) == len(moduli) == d
        - all(m > 1 for m in moduli)
        - all(m.bit_length() <= config.max_modulus_bits for m in moduli)
    
    Properties:
        - Each coordinate coords[i] is in [0, moduli[i]-1]
        - Fitness is typically in [0, 1]
    
    Example:
        >>> gene = Gene(
        ...     id=1,
        ...     coords=np.array([5, 10, 15], dtype=np.int64),
        ...     moduli=np.array([7, 11, 19], dtype=np.int64),
        ...     fitness=0.75,
        ...     generation=1,
        ...     metadata={'origin': 'initialization'}
        ... )
        >>> assert gene.dimension() == 3
        >>> gene_dict = gene.to_dict()
        >>> gene_restored = Gene.from_dict(gene_dict)
        >>> assert np.array_equal(gene_restored.coords, gene.coords)
    """
    id: int
    coords: np.ndarray
    moduli: np.ndarray
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    experience: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate gene after initialization."""
        # Type validation
        if self.coords.dtype != np.int64:
            raise ValueError(f"coords dtype must be int64, got {self.coords.dtype}")
        if self.moduli.dtype != np.int64:
            raise ValueError(f"moduli dtype must be int64, got {self.moduli.dtype}")
        
        # Dimension validation
        if self.coords.ndim != 1:
            raise ValueError(f"coords must be 1-D array, got shape {self.coords.shape}")
        if self.moduli.ndim != 1:
            raise ValueError(f"moduli must be 1-D array, got shape {self.moduli.shape}")
        if len(self.coords) != len(self.moduli):
            raise ValueError(
                f"coords and moduli must have same length, got {len(self.coords)} vs {len(self.moduli)}"
            )
        
        # Moduli validation
        if not all(m > 1 for m in self.moduli):
            raise ValueError(f"All moduli must be > 1, got {self.moduli}")
        
        # Bit length validation (default max: 64 bits)
        max_modulus_bits = 64
        for i, m in enumerate(self.moduli):
            if m.bit_length() > max_modulus_bits:
                raise ValueError(
                    f"Modulus at index {i} ({m}) exceeds max_modulus_bits ({max_modulus_bits})"
                )
        
        # Add creation timestamp if not present
        if 'creation_timestamp' not in self.metadata:
            self.metadata['creation_timestamp'] = datetime.now().isoformat()
    
    def dimension(self) -> int:
        """Return dimensionality of the gene (d)."""
        return len(self.coords)
    
    def copy(self, new_id: int, new_generation: int, new_metadata: Optional[Dict] = None) -> 'Gene':
        """
        Create a deep copy of this gene with new ID and generation.
        
        Args:
            new_id: Unique ID for the copy
            new_generation: Generation of the copy
            new_metadata: Optional metadata override (merged with parent metadata)
        
        Returns:
            New Gene object with identical coords/moduli but new ID and metadata
        
        Example:
            >>> gene = Gene(id=1, coords=np.array([1,2], dtype=np.int64), ...)
            >>> copy = gene.copy(new_id=2, new_generation=1)
            >>> assert copy.id == 2
            >>> assert np.array_equal(copy.coords, gene.coords)
        """
        merged_metadata = self.metadata.copy()
        if new_metadata:
            merged_metadata.update(new_metadata)
        
        return Gene(
            id=new_id,
            coords=self.coords.copy(),
            moduli=self.moduli.copy(),
            fitness=self.fitness,
            generation=new_generation,
            metadata=merged_metadata,
            experience=self.experience.copy() if self.experience is not None else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert gene to dictionary for storage (HDF5, JSON, etc).
        
        Returns:
            Dictionary with all gene data in JSON-serializable format
        
        Example:
            >>> gene_dict = gene.to_dict()
            >>> assert isinstance(gene_dict['coords'], list)
            >>> assert gene_dict['fitness'] == gene.fitness
        """
        return {
            'id': int(self.id),
            'coords': self.coords.tolist(),
            'moduli': self.moduli.tolist(),
            'fitness': float(self.fitness),
            'generation': int(self.generation),
            'metadata': self.metadata,
            'experience': self.experience.tolist() if self.experience is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Gene':
        """
        Reconstruct gene from dictionary (reverse of to_dict).
        
        Args:
            data: Dictionary created by to_dict()
        
        Returns:
            Reconstructed Gene object with identical state
        
        Example:
            >>> gene_dict = gene.to_dict()
            >>> restored = Gene.from_dict(gene_dict)
            >>> assert restored.to_dict() == gene_dict
        """
        return cls(
            id=data['id'],
            coords=np.array(data['coords'], dtype=np.int64),
            moduli=np.array(data['moduli'], dtype=np.int64),
            fitness=data.get('fitness', 0.0),
            generation=data.get('generation', 0),
            metadata=data.get('metadata', {}),
            experience=np.array(data['experience']) if data.get('experience') is not None else None
        )


@dataclass
class FitnessRecord:
    """
    Records fitness scores for different arenas and overall fitness.
    
    Attributes:
        total: float - weighted sum of all arena fitnesses
        arena_scores: Dict[str, float] - individual arena scores (e.g., 'hash', 'modular')
        timestamp: datetime - when this fitness was computed
        generation: int - generation of evaluation
    
    Example:
        >>> record = FitnessRecord(
        ...     total=0.75,
        ...     arena_scores={'hash': 0.8, 'modular': 0.7},
        ...     generation=10
        ... )
        >>> assert record.total == 0.75
    """
    total: float
    arena_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    generation: int = 0
    
    def __repr__(self) -> str:
        arenas = ", ".join(f"{k}={v:.4f}" for k, v in self.arena_scores.items())
        return f"FitnessRecord(total={self.total:.4f}, [{arenas}], gen={self.generation})"


@dataclass
class LineageRecord:
    """
    Records parent-child relationship and operation details.
    
    This record captures the complete context of a reproductive event, enabling
    full audit trail and ancestry reconstruction.
    
    Attributes:
        child_id: int - ID of offspring
        parent_ids: List[int] - IDs of parents (1 for asexual, 2 for sexual)
        operation: OperationType - type of genetic operation
        consensus_modulus: Optional[int] - modulus used for interaction (if sexual)
        generation: int - generation at which reproduction occurred
        parent_coords: List[np.ndarray] - snapshots of parent coordinates
        parent_moduli: List[np.ndarray] - snapshots of parent moduli
        mutations: List[Dict] - mutation log for each parent
        additional_data: Dict - operation-specific data
    
    Example:
        >>> record = LineageRecord(
        ...     child_id=42,
        ...     parent_ids=[10, 11],
        ...     operation=OperationType.CROSSOVER,
        ...     consensus_modulus=7,
        ...     generation=5
        ... )
        >>> assert record.is_sexual == True
    """
    child_id: int
    parent_ids: List[int]
    operation: OperationType
    consensus_modulus: Optional[int] = None
    generation: int = 0
    parent_coords: List[np.ndarray] = field(default_factory=list)
    parent_moduli: List[np.ndarray] = field(default_factory=list)
    mutations: List[Dict[str, Any]] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_sexual(self) -> bool:
        """Whether this is sexual (2 parents) or asexual (1 parent) reproduction."""
        return len(self.parent_ids) == 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'child_id': self.child_id,
            'parent_ids': self.parent_ids,
            'operation': self.operation.value,
            'consensus_modulus': self.consensus_modulus,
            'generation': self.generation,
            'parent_coords': [pc.tolist() for pc in self.parent_coords],
            'parent_moduli': [pm.tolist() for pm in self.parent_moduli],
            'mutations': self.mutations,
            'additional_data': self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageRecord':
        """Reconstruct from dictionary."""
        return cls(
            child_id=data['child_id'],
            parent_ids=data['parent_ids'],
            operation=OperationType(data['operation']),
            consensus_modulus=data.get('consensus_modulus'),
            generation=data.get('generation', 0),
            parent_coords=[np.array(pc, dtype=np.int64) for pc in data.get('parent_coords', [])],
            parent_moduli=[np.array(pm, dtype=np.int64) for pm in data.get('parent_moduli', [])],
            mutations=data.get('mutations', []),
            additional_data=data.get('additional_data', {})
        )


@dataclass
class ConsensusGroup:
    """
    Represents a group of genes that achieved consensus to interact.
    
    Attributes:
        members: List[int] - gene IDs in this group
        agreed_modulus: int - modulus agreed upon for interaction
        compatibility_scores: Dict[Tuple[int, int], float] - pairwise compatibility
        generation: int - generation at which consensus was formed
    
    Example:
        >>> group = ConsensusGroup(
        ...     members=[1, 2, 3],
        ...     agreed_modulus=7,
        ...     generation=5
        ... )
        >>> assert group.size() == 3
    """
    members: List[int]
    agreed_modulus: int
    compatibility_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    generation: int = 0
    
    def size(self) -> int:
        """Number of members in group."""
        return len(self.members)
    
    def avg_compatibility(self) -> float:
        """Average compatibility within group."""
        if not self.compatibility_scores:
            return 0.0
        return np.mean(list(self.compatibility_scores.values()))


class Population:
    """
    Manages the population of genes across generations.
    
    Provides collection management, indexing, fitness statistics, and selection utilities.
    
    Attributes:
        individuals: Dict[int, Gene] - ID → Gene mapping
        generation: int - current generation
        max_id: int - highest ID assigned so far
    
    Example:
        >>> pop = Population()
        >>> gene = Gene(id=1, coords=np.array([1,2], dtype=np.int64), ...)
        >>> pop.add(gene)
        >>> assert pop.size() == 1
        >>> stats = pop.fitness_stats()
        >>> assert 'mean' in stats
    """
    
    def __init__(self):
        self.individuals: Dict[int, Gene] = {}
        self.generation: int = 0
        self.max_id: int = 0
    
    def add(self, gene: Gene) -> None:
        """
        Add a gene to population.
        
        Args:
            gene: Gene object to add
        
        Raises:
            ValueError: If gene ID already exists
        """
        if gene.id in self.individuals:
            raise ValueError(f"Gene ID {gene.id} already exists in population")
        self.individuals[gene.id] = gene
        self.max_id = max(self.max_id, gene.id)
    
    def remove(self, gene_id: int) -> Gene:
        """
        Remove and return a gene from population.
        
        Args:
            gene_id: ID of gene to remove
        
        Returns:
            Removed Gene object
        
        Raises:
            ValueError: If gene_id not found
        """
        if gene_id not in self.individuals:
            raise ValueError(f"Gene ID {gene_id} not found in population")
        return self.individuals.pop(gene_id)
    
    def get(self, gene_id: int) -> Optional[Gene]:
        """
        Get gene by ID.
        
        Args:
            gene_id: ID to look up
        
        Returns:
            Gene if found, None otherwise
        """
        return self.individuals.get(gene_id)
    
    def get_all(self) -> List[Gene]:
        """Return all genes in population."""
        return list(self.individuals.values())
    
    def size(self) -> int:
        """Return population size."""
        return len(self.individuals)
    
    def sorted_by_fitness(self, ascending: bool = False) -> List[Gene]:
        """
        Return genes sorted by fitness.
        
        Args:
            ascending: If False (default), highest fitness first
        
        Returns:
            Sorted list of genes
        """
        return sorted(self.individuals.values(), key=lambda g: g.fitness, reverse=not ascending)
    
    def top_k(self, k: int) -> List[Gene]:
        """
        Return top k genes by fitness.
        
        Args:
            k: Number of top genes to return
        
        Returns:
            Top k genes (highest fitness first)
        """
        return self.sorted_by_fitness(ascending=False)[:k]
    
    def bottom_k(self, k: int) -> List[Gene]:
        """
        Return bottom k genes by fitness.
        
        Args:
            k: Number of bottom genes to return
        
        Returns:
            Bottom k genes (lowest fitness first)
        """
        return self.sorted_by_fitness(ascending=True)[:k]
    
    def fitness_stats(self) -> Dict[str, float]:
        """
        Compute fitness statistics.
        
        Returns:
            Dict with keys: mean, median, std, min, max
        
        Example:
            >>> stats = pop.fitness_stats()
            >>> assert 0 <= stats['min'] <= stats['max']
        """
        if not self.individuals:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        fitnesses = np.array([g.fitness for g in self.individuals.values()])
        return {
            'mean': float(np.mean(fitnesses)),
            'median': float(np.median(fitnesses)),
            'std': float(np.std(fitnesses)),
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses))
        }
    
    def next_id(self) -> int:
        """
        Generate next unique ID.
        
        Returns:
            New unique ID (guaranteed > all previous IDs)
        """
        self.max_id += 1
        return self.max_id
    
    def clear(self) -> None:
        """Remove all individuals from population."""
        self.individuals.clear()


class LineageGraph:
    """
    Tracks genealogical relationships as a directed acyclic graph (DAG).
    
    Structure: parent → child (directed edges)
    Enables full ancestry/descendant queries and pedigree reconstruction.
    
    Example:
        >>> graph = LineageGraph()
        >>> record = LineageRecord(child_id=2, parent_ids=[0, 1], operation=OperationType.CROSSOVER)
        >>> graph.add_edge(record)
        >>> parents = graph.get_parents(2)
        >>> assert parents == [0, 1]
    """
    
    def __init__(self):
        self.edges: List[LineageRecord] = []
        self.node_to_children: Dict[int, List[int]] = {}
        self.node_to_parents: Dict[int, List[int]] = {}
    
    def add_edge(self, record: LineageRecord) -> None:
        """
        Add a parent-child relationship.
        
        Args:
            record: LineageRecord describing reproduction event
        """
        self.edges.append(record)
        
        # Map child to parents
        if record.child_id not in self.node_to_parents:
            self.node_to_parents[record.child_id] = []
        self.node_to_parents[record.child_id].extend(record.parent_ids)
        
        # Map parents to child
        for parent_id in record.parent_ids:
            if parent_id not in self.node_to_children:
                self.node_to_children[parent_id] = []
            self.node_to_children[parent_id].append(record.child_id)
    
    def get_parents(self, node_id: int) -> List[int]:
        """Get immediate parents of a node."""
        return self.node_to_parents.get(node_id, [])
    
    def get_children(self, node_id: int) -> List[int]:
        """Get immediate children of a node."""
        return self.node_to_children.get(node_id, [])
    
    def get_ancestors(self, node_id: int, max_generations: Optional[int] = None) -> List[int]:
        """
        Get all ancestors up to max_generations.
        
        Args:
            node_id: Starting node
            max_generations: Maximum generations to traverse (None = all)
        
        Returns:
            List of ancestor node IDs
        """
        ancestors = set()
        queue = [node_id]
        generations = 0
        
        while queue and (max_generations is None or generations < max_generations):
            next_queue = []
            for node in queue:
                parents = self.get_parents(node)
                ancestors.update(parents)
                next_queue.extend(parents)
            queue = next_queue
            generations += 1
        
        return list(ancestors)
    
    def get_descendants(self, node_id: int, max_generations: Optional[int] = None) -> List[int]:
        """
        Get all descendants up to max_generations.
        
        Args:
            node_id: Starting node
            max_generations: Maximum generations to traverse (None = all)
        
        Returns:
            List of descendant node IDs
        """
        descendants = set()
        queue = [node_id]
        generations = 0
        
        while queue and (max_generations is None or generations < max_generations):
            next_queue = []
            for node in queue:
                children = self.get_children(node)
                descendants.update(children)
                next_queue.extend(children)
            queue = next_queue
            generations += 1
        
        return list(descendants)
    
    def num_edges(self) -> int:
        """Total number of lineage edges."""
        return len(self.edges)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert lineage to list of dictionaries for storage."""
        return [record.to_dict() for record in self.edges]
    
    @classmethod
    def from_list(cls, records: List[Dict[str, Any]]) -> 'LineageGraph':
        """Reconstruct lineage graph from list of dictionaries."""
        graph = cls()
        for record_dict in records:
            record = LineageRecord.from_dict(record_dict)
            graph.add_edge(record)
        return graph


@dataclass
class GenerationSnapshot:
    """
    Complete state snapshot of a generation for storage/replay.
    
    Contains all data needed to reconstruct a generation for analysis,
    reproducibility verification, or system recovery.
    
    Attributes:
        generation: int - generation number
        population: List[Dict] - serialized genes
        lineage_edges: List[Dict] - genealogy records
        consensus_groups: List[Dict] - interaction groups that formed
        fitness_stats: Dict - population fitness statistics
        timestamp: datetime - when snapshot was created
        metadata: Dict - additional experimental data (config, RNG seeds, etc)
    
    Example:
        >>> snapshot = GenerationSnapshot(
        ...     generation=10,
        ...     population=[...],
        ...     lineage_edges=[...],
        ...     consensus_groups=[],
        ...     fitness_stats={'mean': 0.75, 'std': 0.1},
        ... )
        >>> snapshot_dict = snapshot.to_dict()
    """
    generation: int
    population: List[Dict]
    lineage_edges: List[Dict]
    consensus_groups: List[Dict]
    fitness_stats: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON/HDF5 serialization.
        
        Returns:
            Dictionary with all snapshot data
        """
        return {
            'generation': self.generation,
            'population': self.population,
            'lineage_edges': self.lineage_edges,
            'consensus_groups': self.consensus_groups,
            'fitness_stats': self.fitness_stats,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationSnapshot':
        """
        Reconstruct from dictionary.
        
        Args:
            data: Dictionary created by to_dict()
        
        Returns:
            Reconstructed GenerationSnapshot
        """
        return cls(
            generation=data['generation'],
            population=data['population'],
            lineage_edges=data['lineage_edges'],
            consensus_groups=data['consensus_groups'],
            fitness_stats=data['fitness_stats'],
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            metadata=data.get('metadata', {})
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_initial_population(
    population_size: int,
    dimension: int,
    moduli_pool: List[int],
    seed: Optional[int] = None
) -> Population:
    """
    Factory function to create initial population with random genes.
    
    Creates a population of random genes initialized with provided parameters.
    When seed is provided, initialization is deterministic and reproducible.
    
    Args:
        population_size: Number of individuals (N)
        dimension: Polynomial dimensionality (d)
        moduli_pool: List of possible moduli to sample from
        seed: Optional random seed for deterministic generation
    
    Returns:
        Population with random genes
    
    Raises:
        ValueError: If moduli_pool empty or dimension < 1
    
    Example:
        >>> pop1 = create_initial_population(200, 32, [2,3,5,7,11,13], seed=123)
        >>> pop2 = create_initial_population(200, 32, [2,3,5,7,11,13], seed=123)
        >>> assert pop1.get_all()[0].coords[0] == pop2.get_all()[0].coords[0]
    """
    if not moduli_pool:
        raise ValueError("moduli_pool cannot be empty")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    
    if seed is not None:
        np.random.seed(seed)
    
    pop = Population()
    moduli_pool_arr = np.array(moduli_pool, dtype=np.int64)
    
    for _ in range(population_size):
        # Random coordinates in [0, max(moduli)-1]
        coords = np.random.randint(0, int(np.max(moduli_pool_arr)), size=dimension, dtype=np.int64)
        
        # Random moduli sampled from pool
        moduli = np.random.choice(moduli_pool_arr, size=dimension, replace=True).astype(np.int64)
        
        gene = Gene(
            id=pop.next_id(),
            coords=coords,
            moduli=moduli,
            fitness=0.0,
            generation=0,
            metadata={'origin': 'initialization', 'seed': seed}
        )
        pop.add(gene)
    
    return pop


def estimate_population_size(problem_difficulty: str = 'medium') -> int:
    """
    Estimate reasonable population size based on problem difficulty.
    
    Based on research findings (Section 4.2 in refined_specification),
    optimal population sizes for cryptographic tasks are typically 100-200.
    
    Args:
        problem_difficulty: 'easy', 'medium', or 'hard'
    
    Returns:
        Recommended population size
    
    Example:
        >>> size_medium = estimate_population_size('medium')
        >>> assert size_medium == 200
    """
    size_map = {
        'easy': 100,
        'medium': 200,
        'hard': 400
    }
    return size_map.get(problem_difficulty, 200)
