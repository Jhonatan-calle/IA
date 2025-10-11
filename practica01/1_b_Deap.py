import random

from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

toolbox = base.Toolbox()

toolbox.register("attr_bool",random.randint,0,1)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,20) # type: ignore
toolbox.register("population",tools.initRepeat,list,toolbox.individual)# type: ignore


def eval_maxONes(individual):
    return sum(individual),


toolbox.register("evaluate",eval_maxONes)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", lambda x: sum(x)/len(x))
stats.register("max", max)

def main():
    random.seed(42)
    pop = toolbox.population(n=50) # Población inicial # type: ignore
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20,  stats=stats, verbose=True)

if __name__ == "__main__":
        main()
