import random
import math
from deap import base, creator, tools, algorithms

# --- Datos del problema ---
cities = [(0,0), (1,5), (5,2), (7,8), (2,7)]  # ejemplo de 5 ciudades

def distancia_total(individual):
    total = 0
    for i in range(len(individual)):
        c1 = cities[individual[i]]
        c2 = cities[individual[(i+1) % len(individual)]]  # vuelta a la ciudad inicial
        total += math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    return (total,)  # DEAP requiere tupla

# --- Crear clases ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# --- Generadores ---
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Operadores ---
toolbox.register("evaluate", distancia_total)
toolbox.register("mate", tools.cxOrdered)  # cruce para permutaciones
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # mutación permutacional
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Estadísticas ---
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", min)
stats.register("avg", lambda x: sum(x)/len(x))

# --- Algoritmo ---
def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)  # mejor individuo global

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100,
                        stats=stats, halloffame=hof, verbose=True)

    print("Mejor recorrido:", hof[0])
    print("Distancia total:", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()
