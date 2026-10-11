"""
Subset-Preserving Memetic Algorithm (SPMA).

Selects exactly ``n`` items out of a pool of ``N`` items so that a black-box fitness function is maximised. Every
operator (initialisation, crossover, local search) preserves the fixed cardinality ``n``, so no penalty function is
needed to enforce the subset-size constraint.

The engine is intentionally independent of federated learning: the COMPASS-specific Cobb-Douglas utility is injected
through the :class:`FitnessEvaluator` interface (see ``dlsim.compass.fitness``).
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional


class Chromosome:
    """
    A fixed-cardinality selection proposal: a binary gene array of length ``N`` with exactly ``n`` ones.
    The indices of the selected (one) and unselected (zero) genes are kept in two lists so that a swap is O(1).
    """
    __slots__ = ("genes", "ones", "zeros", "fitness")

    def __init__(self, genes: bytearray, ones: List[int], zeros: List[int], fitness: float = -math.inf) -> None:
        self.genes: bytearray = genes
        self.ones: List[int] = ones
        self.zeros: List[int] = zeros
        self.fitness: float = fitness

    @classmethod
    def from_selection(cls, pool_size: int, selected: List[int]) -> "Chromosome":
        """
        Build a chromosome from a list of selected indices.
        """
        genes = bytearray(pool_size)
        for index in selected:
            genes[index] = 1
        ones = [index for index in range(pool_size) if genes[index]]
        zeros = [index for index in range(pool_size) if not genes[index]]
        return cls(genes, ones, zeros)

    @property
    def size(self) -> int:
        return len(self.ones)

    def selected(self) -> List[int]:
        """
        Return the (sorted) indices of the selected genes.
        """
        return sorted(self.ones)

    def key(self) -> bytes:
        """
        Return a hashable representation of the selection, used for fitness memoization.
        """
        return bytes(self.genes)

    def swap(self, one_pos: int, zero_pos: int) -> None:
        """
        Swap the gene at position ``one_pos`` of the ones-list with the gene at position ``zero_pos`` of the
        zeros-list. Calling this method twice with the same arguments reverts the swap.
        """
        one_index = self.ones[one_pos]
        zero_index = self.zeros[zero_pos]
        self.genes[one_index] = 0
        self.genes[zero_index] = 1
        self.ones[one_pos] = zero_index
        self.zeros[zero_pos] = one_index

    def copy(self) -> "Chromosome":
        return Chromosome(bytearray(self.genes), list(self.ones), list(self.zeros), self.fitness)

    def __repr__(self) -> str:
        return "Chromosome(selected=%s, fitness=%g)" % (self.selected(), self.fitness)


class FitnessEvaluator(ABC):
    """
    Interface for the fitness function maximised by the SPMA engine.
    """

    @abstractmethod
    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Return the fitness of a chromosome (higher is better).
        """


@dataclass
class SPMAResult:
    """
    The outcome of an SPMA run.
    """
    best: Chromosome
    evaluations: int
    generations: int


class SPMASolver:
    """
    Memetic engine: random fixed-cardinality initialisation, elitism, tournament selection, Greedy Intersect
    Crossover and Hill-Climbing Swap local search.
    """

    def __init__(self,
                 population_size: int = 20,
                 generations: int = 10,
                 elitism_count: int = 2,
                 mutation_rate: float = 0.3,
                 local_search_depth: int = 3,
                 tournament_size: int = 3,
                 rng: Optional[Random] = None,
                 memoize: bool = True) -> None:
        if population_size < 1:
            raise ValueError("population_size must be at least 1")
        if generations < 0:
            raise ValueError("generations cannot be negative")
        if elitism_count < 0:
            raise ValueError("elitism_count cannot be negative")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if local_search_depth < 0:
            raise ValueError("local_search_depth cannot be negative")
        if tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")

        self.population_size = population_size
        self.generations = generations
        self.elitism_count = min(elitism_count, population_size)
        self.mutation_rate = mutation_rate
        self.local_search_depth = local_search_depth
        self.tournament_size = tournament_size
        self.rng = rng or Random()
        self.memoize = memoize

        # Per-run state
        self._evaluator: Optional[FitnessEvaluator] = None
        self._cache: Dict[bytes, float] = {}
        self._evaluations: int = 0

    # ------------------------------------------------------------------ helpers

    def _evaluate(self, chromosome: Chromosome) -> float:
        if self.memoize:
            key = chromosome.key()
            cached = self._cache.get(key)
            if cached is not None:
                chromosome.fitness = cached
                return cached
        fitness = float(self._evaluator.evaluate(chromosome))
        if math.isnan(fitness):
            fitness = -math.inf
        self._evaluations += 1
        if self.memoize:
            self._cache[chromosome.key()] = fitness
        chromosome.fitness = fitness
        return fitness

    def random_chromosome(self, pool_size: int, subset_size: int) -> Chromosome:
        """
        Create a uniformly random chromosome with exactly ``subset_size`` selected genes.
        """
        return Chromosome.from_selection(pool_size, self.rng.sample(range(pool_size), subset_size))

    def tournament(self, population: List[Chromosome]) -> Chromosome:
        """
        Tournament selection: the fittest of ``tournament_size`` uniformly drawn individuals.
        """
        best: Optional[Chromosome] = None
        for _ in range(min(self.tournament_size, len(population))):
            contestant = population[self.rng.randrange(len(population))]
            if best is None or contestant.fitness > best.fitness:
                best = contestant
        return best

    def crossover(self, parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
        """
        Greedy Intersect Crossover.
        1. Genes selected in both parents are inherited directly (k genes).
        2. The pool consists of the genes selected in exactly one parent.
        3. The remaining n - k genes are drawn from the pool without replacement, a gene originating from parent A
           (resp. B) being drawn with a weight proportional to F_A / (F_A + F_B) (resp. F_B / (F_A + F_B)).
        """
        pool_size = len(parent_a.genes)
        subset_size = parent_a.size
        genes_a, genes_b = parent_a.genes, parent_b.genes

        offspring: List[int] = []
        only_a: List[int] = []
        only_b: List[int] = []
        for index in parent_a.ones:
            if genes_b[index]:
                offspring.append(index)
            else:
                only_a.append(index)
        for index in parent_b.ones:
            if not genes_a[index]:
                only_b.append(index)

        missing = subset_size - len(offspring)
        if missing > 0:
            weight_a, weight_b = self._parent_weights(parent_a.fitness, parent_b.fitness)
            # Efraimidis-Spirakis weighted sampling without replacement: draw key u^(1/w), keep the largest keys.
            keyed = []
            for index in only_a:
                keyed.append((self._sampling_key(weight_a), index))
            for index in only_b:
                keyed.append((self._sampling_key(weight_b), index))
            keyed.sort(reverse=True)
            offspring.extend(index for _, index in keyed[:missing])

        return Chromosome.from_selection(pool_size, offspring)

    @staticmethod
    def _parent_weights(fitness_a: float, fitness_b: float):
        # Fitness values are utilities in (0, 1]; guard against non-positive or non-finite values.
        f_a = fitness_a if math.isfinite(fitness_a) and fitness_a > 0 else 0.0
        f_b = fitness_b if math.isfinite(fitness_b) and fitness_b > 0 else 0.0
        total = f_a + f_b
        if total <= 0:
            return 0.5, 0.5
        return f_a / total, f_b / total

    def _sampling_key(self, weight: float) -> float:
        if weight <= 0:
            return -self.rng.random()  # Only drawn when no positively weighted gene is left
        return self.rng.random() ** (1.0 / weight)

    def hill_climb(self, chromosome: Chromosome) -> None:
        """
        Hill-Climbing Swap local search: swap a selected gene with an unselected one and keep the swap only if the
        fitness strictly improves. Repeated ``local_search_depth`` times.
        """
        if not chromosome.ones or not chromosome.zeros:
            return  # n == 0 or n == N: no swap possible

        for _ in range(self.local_search_depth):
            old_fitness = chromosome.fitness
            one_pos = self.rng.randrange(len(chromosome.ones))
            zero_pos = self.rng.randrange(len(chromosome.zeros))
            chromosome.swap(one_pos, zero_pos)
            new_fitness = self._evaluate(chromosome)
            if new_fitness <= old_fitness:
                chromosome.swap(one_pos, zero_pos)  # Revert
                chromosome.fitness = old_fitness

    # ------------------------------------------------------------------ engine

    def solve(self, pool_size: int, subset_size: int, evaluator: FitnessEvaluator,
              seeds: Optional[List[List[int]]] = None) -> SPMAResult:
        """
        Run the memetic algorithm.
        :param pool_size: N, the number of items in the pool.
        :param subset_size: n, the exact number of items to select.
        :param evaluator: the fitness function.
        :param seeds: optional selections (lists of n indices) injected in the initial population.
        :return: the best chromosome found.
        """
        if pool_size < 0:
            raise ValueError("Pool size cannot be negative")
        if subset_size < 0:
            raise ValueError("Subset size cannot be negative")
        if subset_size > pool_size:
            raise ValueError("Cannot select %d items out of a pool of %d" % (subset_size, pool_size))
        if evaluator is None:
            raise ValueError("A fitness evaluator is required")

        self._evaluator = evaluator
        self._cache = {}
        self._evaluations = 0

        # Trivial instances: there is exactly one feasible solution.
        if subset_size == 0 or subset_size == pool_size:
            only = Chromosome.from_selection(pool_size, list(range(subset_size)))
            self._evaluate(only)
            return SPMAResult(only, self._evaluations, 0)

        population: List[Chromosome] = []
        for seed in (seeds or [])[:self.population_size]:
            if len(set(seed)) == subset_size and all(0 <= index < pool_size for index in seed):
                population.append(Chromosome.from_selection(pool_size, list(seed)))
        while len(population) < self.population_size:
            population.append(self.random_chromosome(pool_size, subset_size))
        for chromosome in population:
            self._evaluate(chromosome)

        best = max(population, key=lambda c: c.fitness).copy()

        for _ in range(self.generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            next_population: List[Chromosome] = [c.copy() for c in population[:self.elitism_count]]

            while len(next_population) < self.population_size:
                parent_a = self.tournament(population)
                parent_b = self.tournament(population)
                child = self.crossover(parent_a, parent_b)
                self._evaluate(child)
                if self.rng.random() < self.mutation_rate:
                    self.hill_climb(child)
                next_population.append(child)

            population = next_population
            generation_best = max(population, key=lambda c: c.fitness)
            if generation_best.fitness > best.fitness:
                best = generation_best.copy()

        return SPMAResult(best, self._evaluations, self.generations)
