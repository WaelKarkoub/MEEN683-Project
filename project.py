from deap import base, creator, tools,algorithms
from bitstring import BitArray
import numpy as np
from scipy.stats import bernoulli
import pickle
import matplotlib.pyplot as plt
import time
import multiprocessing
pool = multiprocessing.Pool()

def decode(individual):
    dl1 = (10*10**-3 - 0)/2**12
    dl2 = (10*10**-3 - 0)/2**12
    dl_sma = (10*10**-3 - 0)/2**12
    dd_spring = (10*10**-3 - 0)/2**12
    dl_spring = (10*10**-3 - 0)/2**12
    dD_spring = (10*10**-3 - 0)/2**12
    dN_spring = 1
    d_dielectric = (1*10**-3 - 0)/2**12
    dl_lever = (10*10**-3 - 0)/2**12

    l1 = dl1*BitArray(individual[0:12]).uint
    l2 = dl2*BitArray(individual[12:24]).uint
    d_sma = BitArray(individual[24:28]).uint
    l_sma = dl_sma*BitArray(individual[28:40]).uint
    d_spring = dd_spring*BitArray(individual[40:52]).uint
    l_spring = dl_spring*BitArray(individual[52:64]).uint
    D_spring = dD_spring*BitArray(individual[64:76]).uint
    N_spring = dN_spring*BitArray(individual[76:88]).uint
    dielectric = d_dielectric*BitArray(individual[88:100]).uint
    material = BitArray(individual[100:102]).uint
    l_lever = dl_spring*BitArray(individual[102:114]).uint

    dd_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.1*10**-3, 0.13*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]
    

    if d_sma >= 0.51*10**-3:
        d_sma = 0.51*10**-3

    if material == 0: # Steel
        E = 207.0*10**9
        G = 79.3*10**9
        v = 0.292
    
    elif material == 1: # Copper
        E = 119.0*10**9
        G = 44.7*10**9
        v = 0.326
    
    elif material == 2: # Aluminum
        E = 71.7*10**9
        G = 26.9*10**9
        v = 0.333
    
    elif material == 3: # Titanium
        E = 114.0*10**9
        G = 42.4*10**9
        v = 0.34
    
    F_spring = (G*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    
    if d_sma == 0:
        F_heating_sma = 8.9 * 10**-3
        F_rest_sma = 3.0 * 10**-3
        I_supply = 45 * 10**-3
        LT = 0.18
        HT = 0.15
    
    elif d_sma == 1:
        F_heating_sma = 20 * 10**-3
        F_rest_sma = 8 * 10**-3
        I_supply = 55 * 10**-3
        LT = 0.24
        HT = 0.20
    
    elif d_sma == 2:
        F_heating_sma = 36 * 10**-3
        F_rest_sma = 14 * 10**-3
        I_supply = 85 * 10**-3
        LT = 0.4
        HT = 0.30
    
    elif d_sma == 3:
        F_heating_sma = 80 * 10**-3
        F_rest_sma = 32 * 10**-3
        I_supply = 150 * 10**-3
        LT = 0.8
        HT = 0.7
    
    elif d_sma == 4:
        F_heating_sma = 143 * 10**-3
        F_rest_sma = 57 * 10**-3
        I_supply = 200 * 10**-3
        LT = 1.1
        HT = 0.9
    
    elif d_sma == 5:
        F_heating_sma = 223 * 10**-3
        F_rest_sma = 89 * 10**-3
        I_supply = 320 * 10**-3
        LT = 1.6
        HT = 1.4
    
    elif d_sma == 6:
        F_heating_sma = 321 * 10**-3
        F_rest_sma = 128 * 10**-3
        I_supply = 410 * 10**-3
        LT = 2.0
        HT = 1.7
    
    elif d_sma == 7:
        F_heating_sma = 570 * 10**-3
        F_rest_sma = 228 * 10**-3
        I_supply = 660 * 10**-3
        LT = 3.2
        HT = 2.7
    
    elif d_sma == 8:
        F_heating_sma = 891 * 10**-3
        F_rest_sma = 356 * 10**-3
        I_supply = 1050 * 10**-3
        LT = 5.4
        HT = 4.5
    
    elif d_sma == 9:
        F_heating_sma = 1280 * 10**-3
        F_rest_sma = 512 * 10**-3
        I_supply = 1500 * 10**-3
        LT = 8.1
        HT = 6.8
    
    elif d_sma == 10:
        F_heating_sma = 2004 * 10**-3
        F_rest_sma = 802 * 10**-3
        I_supply = 2250 * 10**-3
        LT = 10.5
        HT = 8.8
    
    elif d_sma == 11:
        F_heating_sma = 3560 * 10**-3
        F_rest_sma = 1424 * 10**-3
        I_supply = 4000 * 10**-3
        LT = 16.8
        HT = 114.0
    
    F = [F_spring,F_heating_sma,F_rest_sma,I_supply,LT,HT]
    mat = [E,G,v]

    return [l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F]

def geom_cons_1(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if (l1+l2) < 10**-3:
        return True
    else:
        return False

def geom_cons_1_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    return (l1+l2) - 10**-3


def geom_cons_2(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if 2*l_sma < 10**-3:
        return True
    else:
        return False

def geom_cons_2_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    return 2*l_sma - 10**-3

def geom_cons_3(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if (l2/l1)>=1:
        return True
    else:
        return False

def geom_cons_3_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if (l2/l1)>=1:
        return True
    else:
        return False

def geom_cons_4(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if (l_spring - dielectric)<= l2:
        return True
    else:
        return False

def geom_cons_4_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    return (l_spring - dielectric) - l2

def geom_cons_5(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    if (l_spring - (N_spring*d_spring))<=dielectric:
        return True
    else:
        return False

def geom_cons_5_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    return (l_spring - (N_spring*d_spring)) - dielectric

def geom_cons_6(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    eps_max = 0.04
    if l_lever>=(2*eps_max*l_sma*l2/l1):
        return True
    else:
        return False

def geom_cons_6_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    eps_max = 0.04
    return l_lever - (2*eps_max*l_sma*l2/l1)

def geom_cons_7(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    eps_max = 0.04
    if np.sqrt(l_spring**2 - (l_spring-dielectric)**2)>=(eps_max*l_sma*l2/l1):
        return True
    else:
        return False

def geom_cons_7_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)
    eps_max = 0.04
    return np.sqrt(l_spring**2 - (l_spring-dielectric)**2) - (eps_max*l_sma*l2/l1)

def geom_cons_8(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)

    if l_lever < 10*10**-3:
        return True
    else:
        return False

def geom_cons_8_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,F = decode(individual)

    return l_lever - 10*10**-3 + 10**-6

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

population_size = 450
num_generations = 100
gene_length = 114

toolbox = base.Toolbox()
toolbox.register("map", pool.map)
hof = tools.HallOfFame(1)
toolbox.register("binary", bernoulli.rvs,0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.binary, n=gene_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxTwoPoint)
# toolbox.register('crossover', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.5)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('evaluate', eggcrate)
# toolbox.decorate("evaluate", tools.DeltaPenalty(cons_f1, 1000.0, cons_f1_dist))
population = toolbox.population(n = population_size)
pop,logbook , h= algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.5, ngen = num_generations,halloffame=hof, verbose = False)
print("Best individual is: %s\nwith fitness: %s" % (h[0], h[0].fitness))
best_individuals = tools.selBest(population,k = 1)