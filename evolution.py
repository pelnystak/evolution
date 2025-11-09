"""
Evolution Algorithm Implementation
Based on genetic algorithms principles with configurable parameters
"""

import random
import numpy as np
from typing import List, Callable, Tuple, Any
from dataclasses import dataclass


@dataclass
class EvolutionConfig:
    """Configuration for the evolution algorithm"""
    population_size: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    max_generations: int = 100
    convergence_threshold: float = 0.001


class Individual:
    """Represents a single individual in the population"""

    def __init__(self, genome: Any):
        self.genome = genome
        self.fitness: float = 0.0

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, genome={self.genome})"


class EvolutionAlgorithm:
    """
    Generic Evolution Algorithm implementation using genetic algorithm principles.

    This algorithm can be adapted to various optimization problems by providing
    custom genome generation, fitness, mutation, and crossover functions.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Individual = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize the population with random genomes"""
        self.population = [
            Individual(genome_factory())
            for _ in range(self.config.population_size)
        ]
        self.generation = 0

    def evaluate_fitness(self, fitness_function: Callable[[Any], float]) -> None:
        """Evaluate fitness for all individuals in the population"""
        for individual in self.population:
            individual.fitness = fitness_function(individual.genome)

        # Update best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(self.population[0].genome)
            self.best_individual.fitness = self.population[0].fitness

    def selection_tournament(self) -> Individual:
        """Tournament selection - select best from random subset"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def selection_roulette(self) -> Individual:
        """Roulette wheel selection based on fitness proportions"""
        total_fitness = sum(ind.fitness for ind in self.population)
        if total_fitness == 0:
            return random.choice(self.population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in self.population:
            current += individual.fitness
            if current > pick:
                return individual
        return self.population[-1]

    def crossover(self, parent1: Individual, parent2: Individual,
                  crossover_function: Callable) -> Tuple[Any, Any]:
        """Perform crossover between two parents"""
        if random.random() < self.config.crossover_rate:
            return crossover_function(parent1.genome, parent2.genome)
        return parent1.genome, parent2.genome

    def mutate(self, genome: Any, mutation_function: Callable) -> Any:
        """Apply mutation to a genome"""
        if random.random() < self.config.mutation_rate:
            return mutation_function(genome)
        return genome

    def evolve_generation(self,
                         fitness_function: Callable,
                         mutation_function: Callable,
                         crossover_function: Callable,
                         selection_method: str = 'tournament') -> dict:
        """
        Evolve one generation

        Returns statistics about the generation
        """
        # Evaluate fitness
        self.evaluate_fitness(fitness_function)

        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_genome': self.best_individual.genome
        }

        # Store history
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['diversity'].append(stats['std_fitness'])

        # Create new population
        new_population = []

        # Elitism - keep best individuals
        elite_count = min(self.config.elitism_count, len(self.population))
        new_population.extend([
            Individual(ind.genome) for ind in self.population[:elite_count]
        ])

        # Selection method
        selection_func = (self.selection_tournament if selection_method == 'tournament'
                         else self.selection_roulette)

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = selection_func()
            parent2 = selection_func()

            # Crossover
            child1_genome, child2_genome = self.crossover(
                parent1, parent2, crossover_function
            )

            # Mutation
            child1_genome = self.mutate(child1_genome, mutation_function)
            child2_genome = self.mutate(child2_genome, mutation_function)

            # Add to new population
            new_population.append(Individual(child1_genome))
            if len(new_population) < self.config.population_size:
                new_population.append(Individual(child2_genome))

        self.population = new_population[:self.config.population_size]
        self.generation += 1

        return stats

    def run(self,
            fitness_function: Callable,
            mutation_function: Callable,
            crossover_function: Callable,
            genome_factory: Callable,
            selection_method: str = 'tournament',
            callback: Callable = None) -> Individual:
        """
        Run the complete evolution algorithm

        Args:
            fitness_function: Function that evaluates genome fitness
            mutation_function: Function that mutates a genome
            crossover_function: Function that crosses over two genomes
            genome_factory: Function that creates a random genome
            selection_method: 'tournament' or 'roulette'
            callback: Optional function called after each generation with stats

        Returns:
            Best individual found
        """
        # Initialize population
        self.initialize_population(genome_factory)

        # Evolution loop
        for gen in range(self.config.max_generations):
            stats = self.evolve_generation(
                fitness_function,
                mutation_function,
                crossover_function,
                selection_method
            )

            # Callback for progress updates
            if callback:
                callback(stats)

            # Check convergence
            if len(self.history['best_fitness']) > 10:
                recent_improvement = (
                    self.history['best_fitness'][-1] -
                    self.history['best_fitness'][-10]
                )
                if abs(recent_improvement) < self.config.convergence_threshold:
                    print(f"Converged at generation {gen}")
                    break

        return self.best_individual


# Example problem implementations

class StringMatchingProblem:
    """Evolve a string to match a target string"""

    def __init__(self, target: str):
        self.target = target
        self.charset = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,;:'

    def create_genome(self) -> str:
        """Create a random string genome"""
        return ''.join(random.choice(self.charset) for _ in range(len(self.target)))

    def fitness(self, genome: str) -> float:
        """Calculate fitness as number of correct characters"""
        return sum(1 for a, b in zip(genome, self.target) if a == b)

    def mutate(self, genome: str) -> str:
        """Randomly change one character"""
        genome_list = list(genome)
        idx = random.randint(0, len(genome_list) - 1)
        genome_list[idx] = random.choice(self.charset)
        return ''.join(genome_list)

    def crossover(self, genome1: str, genome2: str) -> Tuple[str, str]:
        """Single-point crossover"""
        point = random.randint(1, len(genome1) - 1)
        child1 = genome1[:point] + genome2[point:]
        child2 = genome2[:point] + genome1[point:]
        return child1, child2


class FunctionOptimizationProblem:
    """Optimize a mathematical function"""

    def __init__(self, dimension: int = 2, bounds: Tuple[float, float] = (-10, 10)):
        self.dimension = dimension
        self.bounds = bounds

    def create_genome(self) -> np.ndarray:
        """Create random point in search space"""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)

    def fitness_sphere(self, genome: np.ndarray) -> float:
        """Sphere function (minimize sum of squares)"""
        return -np.sum(genome ** 2)  # Negative because we maximize fitness

    def fitness_rastrigin(self, genome: np.ndarray) -> float:
        """Rastrigin function (many local minima)"""
        A = 10
        n = len(genome)
        return -(A * n + np.sum(genome**2 - A * np.cos(2 * np.pi * genome)))

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to genome"""
        mutation = genome + np.random.normal(0, 0.5, self.dimension)
        # Clip to bounds
        return np.clip(mutation, self.bounds[0], self.bounds[1])

    def crossover(self, genome1: np.ndarray, genome2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Arithmetic crossover"""
        alpha = random.random()
        child1 = alpha * genome1 + (1 - alpha) * genome2
        child2 = (1 - alpha) * genome1 + alpha * genome2
        return child1, child2


class BinaryOptimizationProblem:
    """Optimize a binary string (e.g., OneMax problem)"""

    def __init__(self, length: int = 50):
        self.length = length

    def create_genome(self) -> List[int]:
        """Create random binary string"""
        return [random.randint(0, 1) for _ in range(self.length)]

    def fitness(self, genome: List[int]) -> float:
        """Count number of 1s (OneMax problem)"""
        return sum(genome)

    def mutate(self, genome: List[int]) -> List[int]:
        """Bit flip mutation"""
        genome_copy = genome.copy()
        idx = random.randint(0, len(genome_copy) - 1)
        genome_copy[idx] = 1 - genome_copy[idx]
        return genome_copy

    def crossover(self, genome1: List[int], genome2: List[int]) -> Tuple[List[int], List[int]]:
        """Two-point crossover"""
        point1 = random.randint(0, len(genome1) - 1)
        point2 = random.randint(point1, len(genome1))

        child1 = genome1[:point1] + genome2[point1:point2] + genome1[point2:]
        child2 = genome2[:point1] + genome1[point1:point2] + genome2[point2:]

        return child1, child2
