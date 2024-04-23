import random
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

class ImprovedGeneticAlgorithm:
    """
    An improved genetic algorithm with enhanced reliability and performance.

    Attributes:
        population_size (int): The size of the population.
        mutation_rate (float): The mutation rate for genetic operators.
        elitism_ratio (float): The ratio of elitism in selection.
        pruning_ratio (float): The ratio of individuals to prune in each generation.
        dynamic_mutation_rate (bool): Whether to dynamically adjust the mutation rate.
        logger (logging.Logger): Logger for logging algorithm progress.
    """

    def __init__(self, population_size, mutation_rate, elitism_ratio=0.1, pruning_ratio=0.2):
        """
        Initialize the genetic algorithm with specified parameters.

        Args:
            population_size (int): The size of the population.
            mutation_rate (float): The mutation rate for genetic operators.
            elitism_ratio (float, optional): The ratio of elitism in selection. Defaults to 0.1.
            pruning_ratio (float, optional): The ratio of individuals to prune in each generation. Defaults to 0.2.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.pruning_ratio = pruning_ratio
        self.dynamic_mutation_rate = True
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up the logger for logging algorithm progress.

        Returns:
            logging.Logger: Logger instance.
        """
        logger = logging.getLogger('GeneticAlgorithmLogger')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('genetic_algorithm.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def initialize_population(self, chromosome_length):
        """
        Initialize the population of solutions.

        Args:
            chromosome_length (int): Length of the chromosome for each solution.

        Returns:
            list: List of initial solutions.
        """
        population = [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(self.population_size)]
        return population

    def calculate_fitness(self, solution):
        """
        Calculate the fitness of a solution.

        Args:
            solution (list): Binary representation of a solution.

        Returns:
            float: Fitness value of the solution.
        """
        # Example: Evaluate fitness based on a multidimensional optimization problem
        fitness = np.sum(np.array(solution) ** 2)
        return fitness

    def dynamic_adjust_mutation_rate(self, generation):
        """
        Dynamically adjust the mutation rate based on the generation.

        Args:
            generation (int): Current generation number.
        """
        if self.dynamic_mutation_rate:
            self.mutation_rate = max(0.01, 1.0 / (1 + np.sqrt(generation)))

    def select_parents(self, population):
        """
        Select parent solutions for crossover.

        Args:
            population (list): List of solutions.

        Returns:
            list: List of selected parent solutions.
        """
        tournament_size = max(2, int(self.population_size * 0.1))
        tournament = random.sample(population, tournament_size)
        parents = max(tournament, key=self.calculate_fitness)
        return parents

    def crossover_uniform(self, parent1, parent2):
        """
        Perform uniform crossover to create offspring.

        Args:
            parent1 (list): First parent solution.
            parent2 (list): Second parent solution.

        Returns:
            list: Offspring solution.
        """
        offspring = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
        return offspring

    def mutate(self, solution):
        """
        Mutate a solution with a certain probability.

        Args:
            solution (list): Binary representation of a solution.

        Returns:
            list: Mutated solution.
        """
        mutated_solution = [bit ^ (random.random() < self.mutation_rate) for bit in solution]
        return mutated_solution

    def parallel_calculate_fitness(self, population):
        """
        Calculate fitness scores for a population in parallel.

        Args:
            population (list): List of solutions.

        Returns:
            list: List of fitness scores.
        """
        with ThreadPoolExecutor() as executor:
            fitness_scores = list(executor.map(self.calculate_fitness, population))
        return fitness_scores

    def evolve_population(self, population):
        """
        Evolve the population through selection, crossover, and mutation.

        Args:
            population (list): List of solutions.

        Returns:
            list: New population after evolution.
        """
        elitism_count = int(self.population_size * self.elitism_ratio)
        pruning_count = int(self.population_size * self.pruning_ratio)
        new_population = population[:elitism_count]

        for generation in range(1, 51):
            self.dynamic_adjust_mutation_rate(generation)
            fitness_scores = self.parallel_calculate_fitness(population)
            selected_population = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda x: fitness_scores[x], reverse=True)[:self.population_size]]
            new_population = self.parallel_evolve_population(selected_population)
            population = new_population
            population = self.prune_population(population, pruning_count)

            # Log population diversity
            diversity = self.calculate_population_diversity(population)
            self.logger.info(f'Generation {generation} - Population Diversity: {diversity}')

            # Check for stagnation and terminate if necessary
            if generation > 1:
                if diversity < 0.01 or self.detect_stagnation(fitness_scores):
                    self.logger.info(f'Stopping early at generation {generation} due to stagnation or lack of diversity.')
                    break

        return new_population

    def calculate_population_diversity(self, population):
        """
        Calculate the diversity of the population.

        Args:
            population (list): List of solutions.

        Returns:
            float: Population diversity measure.
        """
        diversity = np.mean(np.std(population, axis=0))
        return diversity

    def detect_stagnation(self, fitness_scores):
        """
        Detect stagnation in fitness scores.

        Args:
            fitness_scores (list): List of fitness scores.

        Returns:
            bool: True if stagnation is detected, False otherwise.
        """
        if len(fitness_scores) < 10:
            return False
        recent_scores = fitness_scores[-10:]
        recent_std_dev = np.std(recent_scores)
        return recent_std_dev < 1e-6

    def parallel_evolve_population(self, population):
        """
        Evolve the population in parallel.

        Args:
            population (list): List of solutions.

        Returns:
            list: New population after evolution.
        """
        with ThreadPoolExecutor() as executor:
            offspring = list(executor.map(self.select_parents, [population] * (self.population_size - 1)))
            mutated_offspring = list(executor.map(self.mutate, offspring))
            new_population = population[:1] + mutated_offspring
        return new_population

    def prune_population(self, population, count):
        """
        Prune the population to maintain diversity.

        Args:
            population (list): List of solutions.
            count (int): Number of individuals to prune.

        Returns:
            list: Pruned population.
        """
        if len(population) <= count:
            return population
        distances = self.calculate_population_distances(population)
        sorted_indices = np.argsort(distances)
        pruned_population = [population[i] for i in sorted_indices[count:]]
        return pruned_population

    def calculate_population_distances(self, population):
        """
        Calculate distances between individuals in the population.

        Args:
            population (list): List of solutions.

        Returns:
            numpy.ndarray: Array of distances.
        """
        distances = []
        for i, solution1 in enumerate(population):
            for solution2 in population[i + 1:]:
                distance = np.linalg.norm(np.array(solution1) - np.array(solution2))
                distances.append(distance)
        return np.array(distances)

    def run_genetic_algorithm(self, chromosome_length):
        """
        Run the genetic algorithm.

        Args:
            chromosome_length (int): Length of the chromosome for each solution.

        Returns:
            list: Final population after evolution.
        """
        population = self.initialize_population(chromosome_length)
        final_population = self.evolve_population(population)
        return final_population

def main():
    """
    Main function to demonstrate the usage of the improved genetic algorithm.
    """
    # Genetic algorithm parameters
    population_size = 100
    mutation_rate = 0.01
    chromosome_length = 10

    # Initialize and run the genetic algorithm
    improved_genetic_algorithm = ImprovedGeneticAlgorithm(population_size, mutation_rate)
    final_population = improved_genetic_algorithm.run_genetic_algorithm(chromosome_length)

    # Log final population
    logging.info(f'Final Population: {final_population}')

if __name__ == "__main__":
    main()
