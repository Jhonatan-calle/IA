import random
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

# 8 posibles tipos de conexiones 

# Matriz de capacidades (Mbps)
capacidades = [
    [50,  80,  50, 120,  60,  70,  90, 100],
    [80,  55,  90,  65,  65, 100,  75,  85],
    [50,  90,  60,  70,  55,  80, 110,  95],
    [120, 65,  70,  60,  60,  75,  85, 150],
    [60,  65,  55,  60,  50,  80,  75,  70],
    [70, 100,  80,  75,  80,  60,  70, 140],
    [90,  75, 110,  85,  75,  70,  65, 130],
    [100, 85,  95, 150,  70, 140, 130, 60]
]

# Matriz de costos (USD) sin ceros
costos = [
    [750, 1000,  700, 1800,  950,  800,  900, 1000],
    [1000, 770, 1200,  900,  950, 1300, 1100,  950],
    [700, 1200,  780,  900,  850,  880, 1600,  970],
    [1800, 900,  900,  850,  850,  800,  920, 2000],
    [950,  950,  850,  800,  790, 1100, 1000,  880],
    [800, 1300,  880,  800, 1100,  770,  900, 1900],
    [900, 1100, 1600,  920, 1000,  900,  750, 1750],
    [1000, 950,  970, 2000,  880, 1900, 1750, 800]
]

def cost_ability(individual):
    cost  = 0 
    ability = 0 
    for i in range(len(individual)):
        orig = individual[i]
        dest = individual[(i + 1) % len(individual)]
        cost += costos[orig][dest]
        ability += capacidades[orig][dest]
    return cost,ability

creator.create("FitnessMinMax",base.Fitness, weights=(-1.0,1.0))
creator.create("Individual",list, fitness=creator.FitnessMinMax ) #type: ignore

toolbox = base.Toolbox()

toolbox.register("indices", lambda: random.choices(range(8), k=10))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # type: ignore
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

# --- Operadores ---
toolbox.register("evaluate", cost_ability)
# Cruce uniforme (cada posición se mezcla con probabilidad 0.5)
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# Mutación: cambiar aleatoriamente valores de la lista
def mutRandomReplace(individual, prob=0.2):
    for i in range(len(individual)):
        if random.random() < prob:
            individual[i] = random.randint(0, 7)  # rango de nodos
    return individual,

toolbox.register("mutate", mutRandomReplace)
toolbox.register("select", tools.selNSGA2)

# --- Estadísticas ---
stats_cost = tools.Statistics(lambda ind: ind.fitness.values[0])
stats_ability = tools.Statistics(lambda ind: ind.fitness.values[1])

stats_cost.register("min", min)
stats_ability.register("max", max)

mstats = tools.MultiStatistics(costo=stats_cost, potencia=stats_ability)



def plot_pareto(population):
    costs = [ind.fitness.values[0] for ind in population]
    capacities = [ind.fitness.values[1] for ind in population]

    plt.figure(figsize=(8,6))
    plt.scatter(costs, capacities, c="blue", label="Individuos")
    
    # Opcional: marcar HOF
    hof_cost = min(costs)
    hof_capacity = max(capacities)
    plt.scatter(hof_cost, hof_capacity, c="red", label="Hall of Fame", marker="*")
    
    plt.xlabel("Costo")
    plt.ylabel("Capacidad")
    plt.title("Frente de Pareto - Última generación")
    plt.legend()
    plt.grid(True)
    plt.show()



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
        ngen=50,
        stats=mstats,  # aqui como le paseo los dos ?
        halloffame=hof,
        verbose=True,
    )
    plot_pareto(pop)

    print("costo:", hof[0].fitness.values[0])
    print("capacidad:", hof[0].fitness.values[1])


if __name__ == "__main__":
    main()

