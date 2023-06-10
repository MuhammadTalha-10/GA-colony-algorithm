import random
import numpy as np

# Genetic Algorithm Parameters
population_size = 10
mutation_rate = 0.01
generations = 100

# Ant Colony Optimization Parameters
colony_size = 10
pheromone_evaporation = 0.5
alpha = 1
beta = 2
Q = 100

# Problem-specific Parameters
num_cities = 10
distance_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))

# Initialize Population for Genetic Algorithm
population = []
for _ in range(population_size):
    individual = np.random.permutation(num_cities)
    population.append(individual)

# Evaluate Fitness of an Individual
def evaluate_fitness(individual):
    total_distance = 0
    for i in range(num_cities-1):
        city_a = individual[i]
        city_b = individual[i+1]
        total_distance += distance_matrix[city_a][city_b]
    return total_distance

# Genetic Algorithm: Selection
def selection(population):
    fitness_scores = [evaluate_fitness(individual) for individual in population]
    selected = random.choices(population, weights=fitness_scores, k=2)
    return selected[0], selected[1]

# Genetic Algorithm: Crossover
def crossover(parent1, parent2):
    child = np.zeros(num_cities, dtype=int)
    start = random.randint(0, num_cities-1)
    end = random.randint(0, num_cities-1)
    if start > end:
        start, end = end, start
    child[start:end] = parent1[start:end]
    for city in parent2:
        if city not in child:
            for i in range(num_cities):
                if child[i] == 0:
                    child[i] = city
                    break
    return child

# Genetic Algorithm: Mutation
def mutation(individual):
    if random.random() < mutation_rate:
        indices = random.sample(range(num_cities), 2)
        individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]
    return individual

# Ant Colony Optimization: Update Pheromone
def update_pheromone(pheromone_matrix, paths, distances):
    pheromone_matrix *= pheromone_evaporation
    num_ants = len(paths)
    for i in range(num_ants):
        path = paths[i]
        distance = distances[i]
        for j in range(num_cities-1):
            city_a = path[j]
            city_b = path[j+1]
            pheromone_matrix[city_a][city_b] += Q / distance

# Ant Colony Optimization: Ant Construction
def construct_solution(pheromone_matrix):
    solutions = []
    distances = []
    for _ in range(colony_size):
        start_city = random.randint(0, num_cities-1)
        solution = [start_city]
        visited = set([start_city])
        while len(solution) < num_cities:
            current_city = solution[-1]
            probabilities = []
            denominator = 0
            for city in range(num_cities):
                if city not in visited:
                    pheromone = pheromone_matrix[current_city][city]
                    distance = distance_matrix[current_city][city]
                    probabilities.append((pheromone ** alpha) * ((1 / distance) ** beta))
                    denominator += probabilities[-1]
                else:
                    probabilities.append(0)
            probabilities = [p / denominator for p in probabilities]
            next_city = random.choices(range(num_cities), weights=probabilities)[0]
            solution.append(next_city)
            visited.add(next_city)
        distance = evaluate_fitness(solution)
        solutions.append(solution)
        distances.append(distance)
    return solutions, distances

# Main Algorithm
for generation in range(generations):
    # Genetic Algorithm
    selected_parent1, selected_parent2 = selection(population)
    child = crossover(selected_parent1, selected_parent2)
    mutated_child = mutation(child)

    # Ant Colony Optimization
    pheromone_matrix = np.ones((num_cities, num_cities))
    paths, distances = construct_solution(pheromone_matrix)
    update_pheromone(pheromone_matrix, paths, distances)

    # Find the index of the worst individual
    worst_index = max(range(len(population)), key=lambda i: evaluate_fitness(population[i]))

    # Replace the worst individual with the mutated child
    population[worst_index] = mutated_child

    # Elitism: Select the best individual from the population
    best_individual = min(population, key=evaluate_fitness)

    # Print the best individual in each generation
    print(f"Generation {generation+1}: Best Distance = {evaluate_fitness(best_individual)}")

# Print the final best individual
best_individual = min(population, key=evaluate_fitness)
print(f"\nFinal Best Individual: {best_individual}")
print(f"Final Best Distance: {evaluate_fitness(best_individual)}")
