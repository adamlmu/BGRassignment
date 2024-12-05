import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from deap import base, creator, tools, algorithms
from scipy.interpolate import make_interp_spline, splprep, splev

# Step 1: Load and preprocess the cone data
# Replace 'BrandsHatchLayout.csv' with the path to your CSV file
df = pd.read_csv('BrandsHatchLayout.csv')
x_coords = df.iloc[:, 0].to_numpy()
y_coords = df.iloc[:, 1].to_numpy()

# Remove any NaN or missing values
valid_indices = ~(np.isnan(x_coords) | np.isnan(y_coords))
x_coords = x_coords[valid_indices]
y_coords = y_coords[valid_indices]

# Split the cones into inner and outer boundaries
# Adjust the indices based on your data
x_inner = x_coords[157:]
y_inner = y_coords[157:]
x_outer = x_coords[:156]
y_outer = y_coords[:156]

# Define the width of the track boundaries
track_width = 8  # Adjust as needed

# Compute the inner and outer limits of the track
x_inner_limit = []
y_inner_limit = []
x_outer_limit = []
y_outer_limit = []

for i in range(len(x_inner)):
    dx = x_outer[i] - x_inner[i]
    dy = y_outer[i] - y_inner[i]
    distance = math.hypot(dx, dy)
    if distance == 0:
        distance = 1e-6  # Prevent division by zero
    scale = track_width / distance
    x_inner_limit.append(x_inner[i] + dx * scale)
    y_inner_limit.append(y_inner[i] + dy * scale)
    x_outer_limit.append(x_outer[i] - dx * scale)
    y_outer_limit.append(y_outer[i] - dy * scale)

# Step 2: Prepare the waypoints for the path
middle_line_x = []
middle_line_y = []

half_length = int(len(x_coords) / 2)
for i in range(half_length - 1):  # Adjusted to prevent index out of bounds
    index_outer = i
    index_inner = i + 1 + half_length
    if index_inner < len(x_coords):
        middle_x = (x_coords[index_outer] + x_coords[index_inner]) / 2
        middle_y = (y_coords[index_outer] + y_coords[index_inner]) / 2
        middle_line_x.append(middle_x)
        middle_line_y.append(middle_y)
    else:
        print(f"Index out of bounds: index_inner = {index_inner}, len(x_coords) = {len(x_coords)}")
        break

# Create splines for the inner and outer limits and the middle line
t_values = np.linspace(0, len(middle_line_x) - 1, len(middle_line_x))
spline_x_inner = make_interp_spline(t_values, x_inner_limit[:len(t_values)], k=3)
spline_y_inner = make_interp_spline(t_values, y_inner_limit[:len(t_values)], k=3)
spline_x_outer = make_interp_spline(t_values, x_outer_limit[:len(t_values)], k=3)
spline_y_outer = make_interp_spline(t_values, y_outer_limit[:len(t_values)], k=3)
spline_x_middle = make_interp_spline(t_values, middle_line_x, k=3)
spline_y_middle = make_interp_spline(t_values, middle_line_y, k=3)

# Set the number of waypoints
num_waypoints = 250  # Adjust this value as needed

# Generate parameter values for interpolation
t_dense = np.linspace(0, len(middle_line_x) - 1, num_waypoints)

# Interpolate the middle line and boundaries
dense_x = spline_x_middle(t_dense)
dense_y = spline_y_middle(t_dense)
dense_x_inner = spline_x_inner(t_dense)
dense_y_inner = spline_y_inner(t_dense)
dense_x_outer = spline_x_outer(t_dense)
dense_y_outer = spline_y_outer(t_dense)

# Prepare the bounds for each waypoint
bounds = []
for i in range(num_waypoints):
    x_min = min(dense_x_inner[i], dense_x_outer[i])
    x_max = max(dense_x_inner[i], dense_x_outer[i])
    y_min = min(dense_y_inner[i], dense_y_outer[i])
    y_max = max(dense_y_inner[i], dense_y_outer[i])
    bounds.append((x_min, x_max, y_min, y_max))

# Step 3: Set up the evolutionary algorithm with DEAP
# Define the fitness function to minimize the total length and curvature of the path
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    individual = []
    for i in range(num_waypoints):
        x_min, x_max, y_min, y_max = bounds[i]
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        individual.extend([x, y])
    return individual

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Set the curvature weight to balance path length and smoothness
curvature_weight = 1  # Adjust this weight as needed

def evaluate(individual):
    x_vals = np.array(individual[::2])  # Even indices
    y_vals = np.array(individual[1::2])  # Odd indices

    # Create a spline representation of the path
    try:
        tck, u = splprep([x_vals, y_vals], s=0)
        unew = np.linspace(0, 1.0, num_waypoints * 10)
        out = splev(unew, tck)

        # Compute distances between consecutive points on the spline
        dx = np.diff(out[0])
        dy = np.diff(out[1])
        distances = np.hypot(dx, dy)
        total_length = np.sum(distances)

        # Compute curvature of the spline
        dx_dt, dy_dt = splev(unew, tck, der=1)
        d2x_dt2, d2y_dt2 = splev(unew, tck, der=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(dx_dt**2 + dy_dt**2, 1.5)
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        mean_curvature = np.mean(curvature)

        # Penalty for going outside the bounds
        penalty = 0
        for i in range(num_waypoints):
            x = individual[2 * i]
            y = individual[2 * i + 1]
            x_min, x_max, y_min, y_max = bounds[i]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                penalty += 1000  # High penalty for violating bounds

        # Combine length and curvature in the fitness function
        total_fitness = total_length + mean_curvature * curvature_weight + penalty
    except Exception as e:
        # If spline fitting fails, assign a high fitness value
        total_fitness = 1e6

    return (total_fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    population_size = 120
    num_generations = 100
    crossover_prob = 0.7
    mutation_prob = 0.3

    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Statistics to keep track of the progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Evolutionary Algorithm
    for gen in range(1, num_generations + 1):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                for i in range(num_waypoints):
                    x_min, x_max, y_min, y_max = bounds[i]
                    mutant[2 * i] = min(max(mutant[2 * i], x_min), x_max)
                    mutant[2 * i + 1] = min(max(mutant[2 * i + 1], y_min), y_max)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

        # Gather statistics
        record = stats.compile(population)
        print(f"Generation {gen}: {record}")

    # Get the best individual
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

best_individual = main()

# Extract the best path and create a smooth spline for visualization
x_vals = np.array(best_individual[::2])
y_vals = np.array(best_individual[1::2])

# Fit the spline for the best path
tck, u = splprep([x_vals, y_vals], s=0)
unew = np.linspace(0, 1.0, num_waypoints * 10)
out = splev(unew, tck)
best_x_spline = out[0]
best_y_spline = out[1]

# Visualize the Results
plt.figure(figsize=(10, 10))
plt.plot(dense_x_outer, dense_y_outer, label='Outer Limit', color='black')
plt.plot(dense_x_inner, dense_y_inner, label='Inner Limit', color='black')
plt.scatter(x_coords, y_coords, marker='o', color='gray', label='Cones', s=10)
plt.plot(best_x_spline, best_y_spline, label='Optimized Smooth Path', color='red', linewidth=2)
plt.legend()
plt.title('Optimized Smooth Path using Evolutionary Algorithm')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()
