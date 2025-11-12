"""
Interactive Evolution Algorithm UI
Streamlit application for visualizing and controlling the evolution algorithm
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from evolution import (
    EvolutionAlgorithm,
    EvolutionConfig,
    StringMatchingProblem,
    FunctionOptimizationProblem,
    BinaryOptimizationProblem
)


def plot_evolution_progress(history):
    """Create interactive plots showing evolution progress"""
    if not history['best_fitness']:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Best Fitness Over Time', 'Average Fitness Over Time',
                       'Fitness Diversity', 'Best vs Average Fitness'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    generations = list(range(len(history['best_fitness'])))

    # Best fitness
    fig.add_trace(
        go.Scatter(x=generations, y=history['best_fitness'],
                  mode='lines+markers', name='Best Fitness',
                  line=dict(color='green', width=2)),
        row=1, col=1
    )

    # Average fitness
    fig.add_trace(
        go.Scatter(x=generations, y=history['avg_fitness'],
                  mode='lines+markers', name='Average Fitness',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )

    # Diversity
    fig.add_trace(
        go.Scatter(x=generations, y=history['diversity'],
                  mode='lines+markers', name='Diversity (Std)',
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )

    # Best vs Average
    fig.add_trace(
        go.Scatter(x=generations, y=history['best_fitness'],
                  mode='lines', name='Best', line=dict(color='green', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=generations, y=history['avg_fitness'],
                  mode='lines', name='Average', line=dict(color='blue', width=2)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Fitness")
    fig.update_layout(height=700, showlegend=True)

    return fig


def plot_function_optimization_2d(problem, best_genome, history):
    """Plot 2D function optimization progress"""
    if len(best_genome) != 2:
        return None

    # Create grid for function visualization
    x = np.linspace(problem.bounds[0], problem.bounds[1], 100)
    y = np.linspace(problem.bounds[0], problem.bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Calculate function values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            genome = np.array([X[i, j], Y[i, j]])
            Z[i, j] = -problem.fitness_sphere(genome)  # Show actual function value

    fig = go.Figure(data=[
        go.Contour(x=x, y=y, z=Z, colorscale='Viridis', name='Function'),
        go.Scatter(x=[best_genome[0]], y=[best_genome[1]],
                  mode='markers', marker=dict(size=15, color='red', symbol='star'),
                  name='Best Solution')
    ])

    fig.update_layout(
        title='Function Optimization Landscape (2D)',
        xaxis_title='x1',
        yaxis_title='x2',
        height=500
    )

    return fig


def main():
    st.set_page_config(
        page_title="Evolution Algorithm Explorer",
        page_icon="🧬",
        layout="wide"
    )

    st.title("🧬 Evolution Algorithm Explorer")
    st.markdown("""
    Explore genetic algorithms and evolutionary computation!
    Based on principles from **Andrzej Dragan's "Quo vAIdis"**.

    Configure parameters, select a problem, and watch evolution in action!
    """)

    # Sidebar for configuration
    st.sidebar.header("⚙️ Algorithm Parameters")

    # Problem selection
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        ["String Matching", "Function Optimization (Sphere)",
         "Function Optimization (Rastrigin)", "Binary Optimization (OneMax)"]
    )

    # Population parameters
    st.sidebar.subheader("Population Settings")
    population_size = st.sidebar.slider("Population Size", 10, 500, 100, 10)
    max_generations = st.sidebar.slider("Max Generations", 10, 1000, 100, 10)

    # Genetic operators
    st.sidebar.subheader("Genetic Operators")
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.01, 0.01)
    crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.7, 0.05)

    # Selection
    st.sidebar.subheader("Selection")
    selection_method = st.sidebar.radio("Selection Method", ["tournament", "roulette"])
    tournament_size = st.sidebar.slider("Tournament Size", 2, 10, 3, 1) if selection_method == "tournament" else 3

    # Elitism
    elitism_count = st.sidebar.slider("Elitism Count", 0, 20, 2, 1)

    # Problem-specific parameters
    st.sidebar.subheader("Problem Configuration")

    if problem_type == "String Matching":
        target_string = st.sidebar.text_input("Target String", "Hello Evolution!")
        problem = StringMatchingProblem(target_string)
        fitness_func = problem.fitness
        max_possible_fitness = len(target_string)

    elif "Function Optimization" in problem_type:
        dimension = st.sidebar.slider("Dimension", 2, 10, 2, 1)
        bounds_min = st.sidebar.number_input("Bounds Min", value=-10.0)
        bounds_max = st.sidebar.number_input("Bounds Max", value=10.0)
        problem = FunctionOptimizationProblem(dimension, (bounds_min, bounds_max))

        if "Rastrigin" in problem_type:
            fitness_func = problem.fitness_rastrigin
            max_possible_fitness = 0
        else:
            fitness_func = problem.fitness_sphere
            max_possible_fitness = 0

    else:  # Binary Optimization
        binary_length = st.sidebar.slider("Binary String Length", 10, 200, 50, 10)
        problem = BinaryOptimizationProblem(binary_length)
        fitness_func = problem.fitness
        max_possible_fitness = binary_length

    # Create config
    config = EvolutionConfig(
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism_count=elitism_count,
        tournament_size=tournament_size,
        max_generations=max_generations,
        convergence_threshold=0.001
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()

        st.subheader("🏆 Best Solution")
        best_solution_placeholder = st.empty()

    with col1:
        st.subheader("📈 Evolution Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

    # Function plot for 2D optimization
    if problem_type.startswith("Function Optimization") and dimension == 2:
        st.subheader("🗺️ Solution Space")
        function_plot_placeholder = st.empty()

    # Run button
    if st.sidebar.button("▶️ Run Evolution", type="primary"):
        # Initialize algorithm
        evolution = EvolutionAlgorithm(config)
        evolution.initialize_population(problem.create_genome)

        # Statistics tracking
        stats_data = []

        # Progress callback
        def update_progress(stats):
            gen = stats['generation']
            progress = min(gen / max_generations, 1.0)
            progress_bar.progress(progress)

            status_text.text(
                f"Generation {gen}/{max_generations} | "
                f"Best Fitness: {stats['best_fitness']:.4f} | "
                f"Avg Fitness: {stats['avg_fitness']:.4f}"
            )

            stats_data.append(stats)

            # Update statistics display
            with stats_placeholder.container():
                st.metric("Generation", gen)
                st.metric("Best Fitness", f"{stats['best_fitness']:.4f}")
                st.metric("Average Fitness", f"{stats['avg_fitness']:.4f}")
                st.metric("Diversity (Std)", f"{stats['std_fitness']:.4f}")

                if max_possible_fitness > 0:
                    progress_pct = (stats['best_fitness'] / max_possible_fitness) * 100
                    st.metric("Progress", f"{progress_pct:.1f}%")

            # Update best solution display
            with best_solution_placeholder.container():
                if problem_type == "String Matching":
                    st.code(stats['best_genome'])
                elif problem_type.startswith("Function Optimization"):
                    st.write(f"Position: {stats['best_genome']}")
                    st.write(f"Value: {-stats['best_fitness']:.6f}")
                else:
                    ones_count = sum(stats['best_genome'])
                    st.write(f"Ones: {ones_count}/{len(stats['best_genome'])}")
                    st.text(''.join(str(x) for x in stats['best_genome'][:50]))

            # Update chart every few generations
            if gen % 5 == 0 or gen == max_generations - 1:
                fig = plot_evolution_progress(evolution.history)
                if fig:
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Update function plot for 2D optimization
                if problem_type.startswith("Function Optimization") and dimension == 2:
                    func_fig = plot_function_optimization_2d(
                        problem, stats['best_genome'], evolution.history
                    )
                    if func_fig:
                        function_plot_placeholder.plotly_chart(func_fig, use_container_width=True)

        # Run evolution
        with st.spinner("Evolution in progress..."):
            best = evolution.run(
                fitness_func,
                problem.mutate,
                problem.crossover,
                problem.create_genome,
                selection_method,
                callback=update_progress
            )

        # Final results
        st.success(f"Evolution completed! Best fitness: {best.fitness:.4f}")

        # Show final statistics
        st.subheader("📋 Final Results")

        results_col1, results_col2 = st.columns(2)

        with results_col1:
            st.write("**Best Individual:**")
            if problem_type == "String Matching":
                st.code(best.genome)
                st.write(f"Matches: {int(best.fitness)}/{len(target_string)}")
            elif problem_type.startswith("Function Optimization"):
                st.write(f"Position: {best.genome}")
                st.write(f"Function Value: {-best.fitness:.6f}")
            else:
                ones_count = sum(best.genome)
                st.write(f"Ones: {ones_count}/{len(best.genome)}")
                st.text(''.join(str(x) for x in best.genome))

        with results_col2:
            st.write("**Evolution Statistics:**")
            st.write(f"Total Generations: {evolution.generation}")
            st.write(f"Final Best Fitness: {best.fitness:.4f}")
            st.write(f"Final Avg Fitness: {evolution.history['avg_fitness'][-1]:.4f}")

            if max_possible_fitness > 0:
                success_rate = (best.fitness / max_possible_fitness) * 100
                st.write(f"Success Rate: {success_rate:.1f}%")

        # Download results
        if stats_data:
            df = pd.DataFrame(stats_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics CSV",
                data=csv,
                file_name="evolution_stats.csv",
                mime="text/csv"
            )

    # Information section
    with st.expander("ℹ️ About Evolution Algorithms"):
        st.markdown("""
        ### What are Genetic/Evolution Algorithms?

        Evolution algorithms are optimization techniques inspired by natural selection and genetics:

        **Key Components:**
        - **Population**: A set of candidate solutions
        - **Fitness Function**: Evaluates how good each solution is
        - **Selection**: Chooses better solutions to reproduce
        - **Crossover**: Combines two parent solutions to create offspring
        - **Mutation**: Introduces random changes to maintain diversity

        **How it Works:**
        1. Initialize a random population
        2. Evaluate fitness of all individuals
        3. Select parents based on fitness
        4. Create offspring through crossover and mutation
        5. Replace old population with new generation
        6. Repeat until convergence or max generations

        **Parameters Explained:**
        - **Population Size**: Number of individuals in each generation
        - **Mutation Rate**: Probability of random changes
        - **Crossover Rate**: Probability of combining parent genes
        - **Elitism**: Number of best individuals preserved unchanged
        - **Tournament Size**: Number of individuals competing in selection
        """)


if __name__ == "__main__":
    main()
