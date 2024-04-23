
# /AI_ML_Modules/genetic_algorithms.py

import numpy as np
import random

def generate_initial_population(size, gene_length):
    return [np.random.randint(0, 2, gene_length).tolist() for _ in range(size)]

def fitness_function(individual):
    # Simple fitness function that calculates the sum of the elements
    return sum(individual)

def crossover(parent_1, parent_2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent_1) - 1)
        return parent_1[:point] + parent_2[point:], parent_2[:point] + parent_1[point:]
    return parent_1, parent_2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]

def run_genetic_algorithm(population_size, gene_length, generations):
    population = generate_initial_population(population_size, gene_length)
    for _ in range(generations):
        population = sorted(population, key=lambda x: -fitness_function(x))
        new_generation = population[:2]  # Elitism: carry forward the top 2
        
        while len(new_generation) < population_size:
            parent_1, parent_2 = random.choices(population[:10], k=2)  # Tournament selection
            offspring_1, offspring_2 = crossover(parent_1, parent_2)
            mutate(offspring_1)
            mutate(offspring_2)
            new_generation.extend([offspring_1, offspring_2])
        
        population = new_generation
    return population

# Example of running the genetic algorithm
if __name__ == '__main__':
    final_population = run_genetic_algorithm(50, 10, 100)
    print(f"Best individual in the final population: {max(final_population, key=fitness_function)}")
