"""
Example script demonstrating programmatic usage of the evolution algorithm
"""

from evolution import (
    EvolutionAlgorithm,
    EvolutionConfig,
    StringMatchingProblem,
    FunctionOptimizationProblem,
    BinaryOptimizationProblem
)


def example_string_matching():
    """Example: Evolve a string to match 'Hello Evolution!'"""
    print("=" * 60)
    print("Example 1: String Matching")
    print("=" * 60)

    # Setup problem
    target = "Hello Evolution!"
    problem = StringMatchingProblem(target)

    # Configure algorithm
    config = EvolutionConfig(
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_count=2,
        max_generations=100
    )

    # Create and run algorithm
    evolution = EvolutionAlgorithm(config)

    def progress_callback(stats):
        if stats['generation'] % 10 == 0:
            print(f"Gen {stats['generation']:3d} | "
                  f"Best: {stats['best_fitness']:2.0f}/{len(target)} | "
                  f"Genome: {stats['best_genome']}")

    best = evolution.run(
        fitness_function=problem.fitness,
        mutation_function=problem.mutate,
        crossover_function=problem.crossover,
        genome_factory=problem.create_genome,
        selection_method='tournament',
        callback=progress_callback
    )

    print(f"\nFinal Result:")
    print(f"Target:  '{target}'")
    print(f"Evolved: '{best.genome}'")
    print(f"Fitness: {best.fitness}/{len(target)}")
    print()


def example_function_optimization():
    """Example: Minimize the sphere function"""
    print("=" * 60)
    print("Example 2: Function Optimization (Sphere)")
    print("=" * 60)

    # Setup problem
    problem = FunctionOptimizationProblem(dimension=5, bounds=(-10, 10))

    # Configure algorithm
    config = EvolutionConfig(
        population_size=50,
        mutation_rate=0.05,
        crossover_rate=0.8,
        elitism_count=3,
        max_generations=100
    )

    # Create and run algorithm
    evolution = EvolutionAlgorithm(config)

    def progress_callback(stats):
        if stats['generation'] % 20 == 0:
            print(f"Gen {stats['generation']:3d} | "
                  f"Best Fitness: {stats['best_fitness']:10.6f} | "
                  f"Actual Value: {-stats['best_fitness']:10.6f}")

    best = evolution.run(
        fitness_function=problem.fitness_sphere,
        mutation_function=problem.mutate,
        crossover_function=problem.crossover,
        genome_factory=problem.create_genome,
        selection_method='tournament',
        callback=progress_callback
    )

    print(f"\nFinal Result:")
    print(f"Best Position: {best.genome}")
    print(f"Function Value: {-best.fitness:.6f}")
    print(f"(Optimal is 0.0 at origin)")
    print()


def example_binary_optimization():
    """Example: Maximize number of 1s (OneMax problem)"""
    print("=" * 60)
    print("Example 3: Binary Optimization (OneMax)")
    print("=" * 60)

    # Setup problem
    length = 50
    problem = BinaryOptimizationProblem(length=length)

    # Configure algorithm
    config = EvolutionConfig(
        population_size=30,
        mutation_rate=0.02,
        crossover_rate=0.9,
        elitism_count=1,
        max_generations=50
    )

    # Create and run algorithm
    evolution = EvolutionAlgorithm(config)

    def progress_callback(stats):
        if stats['generation'] % 10 == 0:
            ones = int(stats['best_fitness'])
            print(f"Gen {stats['generation']:3d} | "
                  f"Ones: {ones}/{length} | "
                  f"Progress: {ones/length*100:.1f}%")

    best = evolution.run(
        fitness_function=problem.fitness,
        mutation_function=problem.mutate,
        crossover_function=problem.crossover,
        genome_factory=problem.create_genome,
        selection_method='tournament',
        callback=progress_callback
    )

    print(f"\nFinal Result:")
    print(f"Ones: {int(best.fitness)}/{length}")
    print(f"Binary String (first 50 bits): {''.join(str(x) for x in best.genome[:50])}")
    print()


if __name__ == "__main__":
    print("\n🧬 Evolution Algorithm Examples\n")

    # Run examples
    example_string_matching()
    example_function_optimization()
    example_binary_optimization()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo run the interactive UI, execute: streamlit run app.py")
