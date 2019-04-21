from deap import base, creator, tools,algorithms
from bitstring import BitArray
import numpy as np
from scipy.stats import bernoulli
import pickle
import matplotlib.pyplot as plt
import time
import multiprocessing
import random 
pool = multiprocessing.Pool()
import itertools
import matplotlib.pyplot as plt

def decode(individual):
    vblock = 500
    eps_air = 10**6

    dl1 = ((10*10**-3)/2 - 0.5*10**-3)/2**12
    dl2 = ((10*10**-3)/2 - 0.5*10**-3)/2**12
    dl_sma = ((10*10**-3)/2 - 2*10**-3)/2**12
    dd_spring = (2*10**-3 - 0.025*10**-3)/2**12
    dl_spring = ((10*10**-3)/2 - 0.0001*10**-3)/2**12
    dD_spring = ((10*10**-3)/2 - 0.05*10**-3)/2**12
    dN_spring = 1
    d_dielectric = ((10*10**-3)/4 - vblock/eps_air)/2**12
    dl_lever = ((10*10**-3)/2 - 0)/2**12
    dI = (4000 * 10**-3 - 5*10**-3)/2**12


    l1 = dl1*BitArray(individual[0:12]).uint + 0.5*10**-3
    l2 = dl2*BitArray(individual[12:24]).uint+ 0.5*10**-3
    d_sma = BitArray(individual[24:28]).uint
    l_sma = dl_sma*BitArray(individual[28:40]).uint+  2*10**-3
    d_spring = dd_spring*BitArray(individual[40:52]).uint+ 0.025*10**-3
    l_spring = dl_spring*BitArray(individual[52:64]).uint+ 0.025*10**-3
    D_spring = dD_spring*BitArray(individual[64:76]).uint+ 1*10**-3
    N_spring = dN_spring*BitArray(individual[76:80]).uint+4
    dielectric = d_dielectric*BitArray(individual[80:92]).uint + vblock/eps_air
    material = BitArray(individual[92:95]).uint
    l_lever = dl_spring*BitArray(individual[95:107]).uint+0
    I = dI*BitArray(individual[107:119]).uint+45*10**-3

    dd_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.1*10**-3, 0.13*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]

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

    elif material >= 3: # Titanium
        E = 114.0*10**9
        G = 42.4*10**9
        v = 0.34

    F_spring = (G*d_spring**4)/(8*N_spring*D_spring**2) * dielectric

    if d_sma == 0:
        F_heating_sma = 9.81*8.9 * 10**-3
        F_rest_sma = 9.81*3.0 * 10**-3
        I_supply = 45 * 10**-3
        R = 1425
        LT = 0.18
        HT = 0.15

    elif d_sma == 1:
        F_heating_sma = 9.81*20 * 10**-3
        F_rest_sma = 9.81*8 * 10**-3
        I_supply = 55 * 10**-3
        R = 890
        LT = 0.24
        HT = 0.20

    elif d_sma == 2:
        F_heating_sma = 9.81*36 * 10**-3
        F_rest_sma = 9.81*14 * 10**-3
        I_supply = 85 * 10**-3
        R = 500
        LT = 0.4
        HT = 0.30

    elif d_sma == 3:
        F_heating_sma = 9.81*80 * 10**-3
        F_rest_sma = 9.81*32 * 10**-3
        I_supply = 150 * 10**-3
        R = 232
        LT = 0.8
        HT = 0.7

    elif d_sma == 4:
        F_heating_sma = 9.81*143 * 10**-3
        F_rest_sma = 9.81*57 * 10**-3
        I_supply = 200 * 10**-3
        R = 126
        LT = 1.1
        HT = 0.9

    elif d_sma == 5:
        F_heating_sma = 9.81*223 * 10**-3
        F_rest_sma = 9.81*89 * 10**-3
        I_supply = 320 * 10**-3
        R = 75
        LT = 1.6
        HT = 1.4

    elif d_sma == 6:
        F_heating_sma = 9.81*321 * 10**-3
        F_rest_sma = 9.81*128 * 10**-3
        I_supply = 410 * 10**-3
        R = 55
        LT = 2.0
        HT = 1.7

    elif d_sma == 7:
        F_heating_sma = 9.81*570 * 10**-3
        F_rest_sma = 9.81*228 * 10**-3
        I_supply = 660 * 10**-3
        R = 29
        LT = 3.2
        HT = 2.7

    elif d_sma == 8:
        F_heating_sma = 9.81*891 * 10**-3
        F_rest_sma = 9.81*356 * 10**-3
        I_supply = 1050 * 10**-3
        R = 18.5
        LT = 5.4
        HT = 4.5

    elif d_sma == 9:
        F_heating_sma = 9.81*1280 * 10**-3
        F_rest_sma = 9.81*512 * 10**-3
        I_supply = 1500 * 10**-3
        R = 12.2
        LT = 8.1
        HT = 6.8

    elif d_sma == 10:
        F_heating_sma = 9.81*2004 * 10**-3
        F_rest_sma = 9.81*802 * 10**-3
        I_supply = 2250 * 10**-3
        LT = 10.5
        R = 8.3
        HT = 8.8

    elif d_sma >= 11:
        F_heating_sma = 9.81*3560 * 10**-3
        F_rest_sma = 9.81*1424 * 10**-3
        I_supply = 4000 * 10**-3
        R = 4.3
        LT = 16.8
        HT = 114.0

    F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    mat = [material,E,G,v]
    return [l1,l2,l_sma,dd_sma[d_sma if d_sma<=len(dd_sma)-1 else 11],d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F]

def tol(value,tolerence=10**-1):
    u = value+tolerence
    l = value-tolerence
    return [u,l]

def geom_cons_1(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)
    if (l1+l2+dielectric) <bu:
        return True
    else:
        return False

def geom_cons_1_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)
    return (l1+l2+dielectric) - bu


def geom_cons_2(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.05
    bu,bl = tol(10*10**-3)
    if (2+eps_max)*l_sma < bu:
        return True
    else:
        return False

def geom_cons_2_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.05
    bu,bl = tol(10*10**-3)
    return (2+eps_max)*l_sma - bu

def geom_cons_3(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(1)
    if (l2/l1)>=bl:
        return True
    else:
        return False

def geom_cons_3_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)
    return (l2/l1)-bl

def geom_cons_4(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(l2)
    if (l_spring)<= bu:
        return True
    else:
        return False

def geom_cons_4_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)
    return (l_spring) - bu

def geom_cons_5(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(2,tolerence=10**1)
    if (l_spring - (N_spring*d_spring))/dielectric>=bl:
        return True
    else:
        return False

def geom_cons_5_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(2)
    return (l_spring - (N_spring*d_spring)) - bl*dielectric

def geom_cons_6(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.05
    bu,bl = tol(2*eps_max*l_sma*l2/l1)
    if l_lever>=bl:
        return True
    else:
        return False

def geom_cons_6_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.04
    bu,bl = tol(2*eps_max*l_sma*l2/l1)
    return l_lever - bl

def geom_cons_7(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.05
    bu,bl = tol(eps_max*l_sma*l2/l1)
    if l_spring**2 - (l_spring-dielectric) <= 0:
        return False
    elif np.sqrt(l_spring**2 - (l_spring-dielectric)**2)>=bl:
        return True
    else:
        return False

def geom_cons_7_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    eps_max = 0.05
    bu,bl = tol(eps_max*l_sma*l2/l1)
    if l_spring**2 - (l_spring-dielectric) < 0:
        return 1
    return np.sqrt(l_spring**2 - (l_spring-dielectric)**2) - bl

def geom_cons_8(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)
    if l_lever < bu:
        return True
    else:
        return False

def geom_cons_8_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(10*10**-3)

    return l_lever - bu

def geom_cons_9(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(6)
    if D_spring/d_spring >= bl:
        return True
    else:
        return False

def geom_cons_9_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(6)
    return D_spring/d_spring - bl

def current(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    
    return (10**7)*d_sma**2 + 275.92*d_sma + 0.0301

def func_cons_1(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # mat = [material,E,G,v]
    bu,bl = tol(current(individual),tolerence=1)
    if I < bu:
        return True
    else:
        return False


def func_cons_1_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(current(individual))

    return I - bu

def func_cons_2(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # mat = [material,E,G,v]
    bu,bl = tol(0)

    if F[1]*l1 - (F[2]*l1 + 0.41*F[0]*l2) > bl:
        return True
    else:
        return False

def func_cons_2_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(0)
   
    return F[1]*l1 - F[2]*l1 + 0.41*F[0]*l2 -bl

def func_cons_3(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # mat = [material,E,G,v]
    bu,bl = tol(0)

    if F[2]*l1 - (F[0]*l2*0.51 )< bu:
        return True
    else:
        return False

def func_cons_3_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    bu,bl = tol(0)
    return F[2]*l1 - F[0]*l2*0.51 - bu

def func_cons_4(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # mat = [material,E,G,v]
    vblock = 500
    eps_air = 10**6
    bu,bl = tol(vblock/eps_air)
    if 2*dielectric >=bl:
        return True
    else:
        return False

def func_cons_4_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    vblock = 500
    eps_air = 10**6
    bu,bl = tol(vblock/eps_air)
    return 2*dielectric - bl

def func_cons_5(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # mat = [material,E,G,v]
    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    if k > 0:
        return True
    else:
        return False

def func_cons_5_dist(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    return k

def power(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    if k < 0:
        k = 0
    t = -((rho*d_sma*cp)/(4*h))*np.log(k)
    return F[4]*t*I**2

def height(individual):
    l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(individual)
    return np.argmax([D_spring,d_sma])

def J(individual):
    return (power(individual),height(individual))

def valid(individual):
    if geom_cons_1(individual) and geom_cons_2(individual) and geom_cons_3(individual) and geom_cons_4(individual) and geom_cons_5(individual) \
        and geom_cons_6(individual) and geom_cons_7(individual) and geom_cons_9(individual) and func_cons_1(individual) \
            and func_cons_2(individual) and func_cons_3(individual) and func_cons_4(individual):

        return True
    else:
        return False


population_size = 500
num_generations = 50
gene_length = 119
def permutations(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    for indices in itertools.product(range(n), repeat=r):
        if len(set(indices)) == r:
            yield tuple(pool[i] for i in indices)
i = 0
# for item in permutations([0,1],119):
#     print(item)
#     if i > 10:
#         break

# while True:
#     possible = np.random.choice([0, 1], size=(gene_length,), p=[1./2, 1./2])
#     possible = list(possible)
#     if valid(possible):
#         l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(possible)
#         print("valid")
#         print("l1: {}, l2: {}, l_sma: {}, d_sma: {}, d_spring: {}, l_spring: {}, D_spring: {}, N_spring: {}, dielectric: {}, l_lever: {}, mat: {}, I: {}, Energy: {}".format(l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat[0],I, power(possible)))
#         break
#     else:
#         print("not valid")
#     i+=1
#     print(i)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# toolbox.register("map", pool.map)
toolbox.register("binary", bernoulli.rvs,0.5)
# toolbox.register("binary", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.binary, n=gene_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.6)
toolbox.register('select', tools.selNSGA2)
toolbox.register('evaluate', J)


listCons = [geom_cons_1,geom_cons_2,geom_cons_3,geom_cons_4,geom_cons_5,geom_cons_6,geom_cons_7,geom_cons_9,func_cons_1,func_cons_2,func_cons_3,func_cons_4]
listDist = [geom_cons_1_dist,geom_cons_2_dist,geom_cons_3_dist,geom_cons_4_dist,geom_cons_5_dist,geom_cons_6_dist,geom_cons_7_dist,geom_cons_9_dist,func_cons_1_dist,func_cons_2_dist,func_cons_3_dist,func_cons_4_dist]
for i in range(len(listCons)):
    toolbox.decorate("evaluate", tools.DeltaPenalty(listCons[i], 12.0))

# pareto=tools.ParetoFront()
population = toolbox.population(n = population_size)
# fits=toolbox.map(toolbox.evaluate,population)
# for fit,ind in zip(fits,population):
#     ind.fitness.values=fit
# pareto.update(population)
# counter = 0
# for gen in range(num_generations):
#     offspring=algorithms.varAnd(population,toolbox,cxpb=1.0,mutpb=0.1)
#     fits=toolbox.map(toolbox.evaluate,population)
#     for fit,ind in zip(fits,offspring):
#         ind.fitness.values=fit
#     population=toolbox.select(offspring+population,k=population_size)
#     pareto.update(population)
#     counter += 1
#     print(counter)

pop,logbook = algorithms.eaSimple(population, toolbox, cxpb = 1, mutpb = 0.5, ngen = num_generations, verbose = True)
# pareto_ind=pareto.items
# p=np.array(map(J,pareto_ind))
# h=np.array(map(J,pareto_ind))
# print(p)

# plt.plot(p,h,'ro')

best_individuals = tools.selBest(pop,k = 1)
l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(best_individuals[0])
print("l1: {}, l2: {}, l_sma: {}, d_sma: {}, d_spring: {}, l_spring: {}, D_spring: {}, N_spring: {}, dielectric: {}, l_lever: {}, mat: {}, I: {}, J: {}".format(l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat[0],I, J(best_individuals[0])))

for i,cons in enumerate(listCons):
    print(list(map(cons,[best_individuals[0]])))

for ind in pop:
    if valid(ind):
        l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat,I,F = decode(ind)
        print("valid")
        print("l1: {}, l2: {}, l_sma: {}, d_sma: {}, d_spring: {}, l_spring: {}, D_spring: {}, N_spring: {}, dielectric: {}, l_lever: {}, mat: {}, I: {}, Energy: {}".format(l1,l2,l_sma,d_sma,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,mat[0],I, power(pop)))
