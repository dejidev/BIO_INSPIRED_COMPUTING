import numpy as np
import random
import matplotlib.pyplot as plt

# Objective Function to Minimize
def objective_function(x1, x2, x3):
    return 2*x1*x2*x3 - 4*x1*x3 - 2*x2*x3 + x1**2 + x2**2 + x3**2 - 2*x1 - 4*x2 + 4*x3

# Decode denary chromosome into real values within bounds
def decode_denary(chromosome, bounds, decimal_places):
    decoded = []
    var_len = len(chromosome) // 3
    for i, (lower, upper) in enumerate(bounds):
        var_str = ''.join(str(bit) for bit in chromosome[i*var_len:(i+1)*var_len])
        integer_part = int(var_str[:2])
        decimal_part = float(var_str[2:]) / (10 ** decimal_places)
        value = integer_part + decimal_part
        decoded.append(np.clip(value, lower, upper))
    return decoded

# Encode real number into denary string of fixed length
def encode_denary(value, decimal_places):
    integer_part = int(value)
    decimal_part = round((value - integer_part) * (10 ** decimal_places))
    return f"{integer_part:02d}{decimal_part:0{decimal_places}d}"

# Initialize denary population
def initialize_denary_population(pop_size, decimal_places):
    population = []
    for _ in range(pop_size):
        individual = []
        for _ in range(3):  # Three variables
            val = f"{random.randint(0, 99):02d}{random.randint(0, 10**decimal_places - 1):0{decimal_places}d}"
            individual.extend(list(val))
        population.append(individual)
    return population

# Roulette Wheel Selection (adjusted for minimization)
def roulette_wheel_selection(population, fitnesses):
    max_fitness = max(fitnesses)
    adjusted_fitness = [max_fitness - f for f in fitnesses]
    total = sum(adjusted_fitness)
    if total == 0:
        return random.choice(population)
    probs = [af / total for af in adjusted_fitness]
    return population[np.random.choice(len(population), p=probs)]

# One-point crossover
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Mutation function (digit-wise mutation)
def mutate(chromosome, mutation_rate, decimal_places):
    chrom_list = chromosome.copy()
    for i in range(len(chrom_list)):
        if random.random() < mutation_rate:
            if i % (2 + decimal_places) < 2:
                chrom_list[i] = str(random.randint(0, 9))  # Integer part digit
            else:
                chrom_list[i] = str(random.randint(0, 9))  # Decimal part digit
    return chrom_list

# Main Genetic Algorithm
def genetic_algorithm(bounds, decimal_places=5, pop_size=50, generations=50,
                      crossover_prob=0.85, mutation_prob=0.25, threshold=0.05):
    chromosome_length = 3 * (2 + decimal_places)
    population = initialize_denary_population(pop_size, decimal_places)
    best_fitness_progress = []

    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        decoded_pop = [decode_denary(ind, bounds, decimal_places) for ind in population]
        fitnesses = [objective_function(*ind) for ind in decoded_pop]

        gen_best_idx = np.argmin(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]

        best_fitness_progress.append(gen_best_fitness)

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = decoded_pop[gen_best_idx]

        if gen > 0 and abs(best_fitness_progress[-1] - best_fitness_progress[-2]) < threshold:
            break

        new_population = []
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)

            if random.random() < crossover_prob:
                offspring1, offspring2 = one_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]

            new_population.append(mutate(offspring1, mutation_prob, decimal_places))
            if len(new_population) < pop_size:
                new_population.append(mutate(offspring2, mutation_prob, decimal_places))

        population = new_population

    return best_solution, best_fitness, best_fitness_progress

# Bounds for x1, x2, x3
bounds = [(10, 90), (0, 90), (-20, 60)]

# Task vi–vii: Run GA with different string lengths
string_lengths = [7, 8, 9, 10]  # Corresponds to 5, 6, 7, 8 decimal places
results_by_length = {}

print("Running GA with varying string lengths...")
for sl in string_lengths:
    dp = sl - 2  # decimal places = string length - 2
    sol, fit, progress = genetic_algorithm(bounds, decimal_places=dp, pop_size=50)
    results_by_length[sl] = progress
    print(f"String Length {sl}: Best Fitness = {fit:.6f}, Solution = [{sol[0]:.5f}, {sol[1]:.5f}, {sol[2]:.5f}]")

# Plot Fitness vs Generation for all string lengths
plt.figure(figsize=(10, 6))
for sl in string_lengths:
    plt.plot(results_by_length[sl], label=f"Length {sl}")
plt.title("Fitness vs Generation (Varying String Lengths)")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid(True)
plt.show()

# Task viii–ix: Run GA with different population sizes
population_sizes = [20, 30, 40, 50, 60, 70]
results_by_population = {}

print("\nRunning GA with varying population sizes...")
for size in population_sizes:
    sol, fit, progress = genetic_algorithm(bounds, decimal_places=5, pop_size=size)
    results_by_population[size] = progress
    print(f"Population Size {size}: Best Fitness = {fit:.6f}, Solution = [{sol[0]:.5f}, {sol[1]:.5f}, {sol[2]:.5f}]")

# Plot Fitness vs Generation for all population sizes
plt.figure(figsize=(10, 6))
for size in population_sizes:
    plt.plot(results_by_population[size], label=f"Size {size}")
plt.title("Fitness vs Generation (Varying Population Sizes)")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid(True)
plt.show()
# import numpy as np