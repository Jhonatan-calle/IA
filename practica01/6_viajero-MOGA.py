import random
import math
from deap import base, creator, tools, algorithms

# --- Datos del problema ---
cities = [(0, 0), (1, 5), (5, 2), (7, 8), (2, 7)]  # ejemplo de 5 ciudades


# Matriz de tráfico (mayor valor = más lento el tramo)
traffic = [
    [1, 1.2, 1.1, 1.5, 1],
    [1, 1, 1.3, 1.1, 1.4],
    [1.2, 1.1, 1, 1.3, 1],
    [1, 1.5, 1.1, 1, 1.2],
    [1, 1.4, 1, 1.3, 1],
]


def distancia_time_total(individual):
    distance = 0
    time = 0
    for i in range(len(individual)):
        c1 = cities[individual[i]]
        c2 = cities[individual[(i + 1) % len(individual)]]  # vuelta a la ciudad inicial
        distance += math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        time += traffic[individual[i]][individual[(i + 1) % len(individual)]]

    return (distance, time)


# --- Crear clases ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

toolbox = base.Toolbox()

# --- Generadores ---
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # type: ignore
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

# --- Operadores ---
toolbox.register("evaluate", distancia_time_total)
toolbox.register("mate", tools.cxOrdered)  # cruce para permutaciones
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # mutación permutacional
toolbox.register("select", tools.selNSGA2)

# --- Estadísticas ---
stats_dist = tools.Statistics(lambda ind: ind.fitness.values[0])
stats_time = tools.Statistics(lambda ind: ind.fitness.values[1])

stats_dist.register("min", min)
stats_dist.register("avg", lambda x: sum(x) / len(x))
stats_time.register("min", min)
stats_time.register("avg", lambda x: sum(x) / len(x))

mstats = tools.MultiStatistics(dist=stats_dist, time=stats_time)


# --- Algoritmo ---
def main():
    random.seed(42)
    pop = toolbox.population(n=50)  # type: ignore
    hof = tools.HallOfFame(1)  # mejor individuo global

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=100,
        stats=mstats,  # aqui como le paseo los dos ?
        halloffame=hof,
        verbose=True,
    )

    print("Mejor recorrido:", hof[0])
    print("Distancia total:", hof[0].fitness.values[0])
    print("Tiempo total:", hof[0].fitness.values[1])


if __name__ == "__main__":
    main()
