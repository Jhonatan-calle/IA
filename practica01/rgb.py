import math
import random

from deap import base, creator, tools, algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_int",random.randint,0,255)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_int,3)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)


def distancia_euclidiana(Individual):
    muestra=[0,0,0]
    return math.sqrt((Individual[0]-muestra[0])**2 + (Individual[1]-muestra[1])**2 + (Individual[2]-muestra[2])**2),


toolbox.register("evaluate",distancia_euclidiana)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 

# Registrar estadísticas de fitness

# Función que devuelve el individuo completo
stats = tools.Statistics(lambda ind: ind)  # registramos todo el individuo
stats.register("best", lambda inds: max(inds, key=lambda ind: ind.fitness.values[0]))

def main():
    random.seed(42)
    
    pop = toolbox.population(n=50)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=50, 
                        stats=stats, verbose=True)

if __name__ == "__main__":
        main()
