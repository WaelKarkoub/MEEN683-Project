from deap import base, creator, tools,algorithms
from bitstring import BitArray
import numpy as np
from scipy.stats import bernoulli
import pickle
import matplotlib.pyplot as plt
import time
import random 
import matplotlib.pyplot as plt
import pandas as pd 
import itertools
doe = pd.read_csv("random_doe.csv")
experiment = doe.values.tolist()



def materialProp(material):
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
    
    return [material,E,G,v]

def smaProp(d_sma):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    dd_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]
    ind = find_nearest(dd_sma,d_sma)
    if ind == 0:
        F_heating_sma = 9.81*8.9 * 10**-3
        F_rest_sma = 9.81*3.0 * 10**-3
        I_supply = 45 * 10**-3
        R = 1425
        LT = 0.18
        HT = 0.15

    elif ind == 1:
        F_heating_sma = 9.81*20 * 10**-3
        F_rest_sma = 9.81*8 * 10**-3
        I_supply = 55 * 10**-3
        R = 890
        LT = 0.24
        HT = 0.20

    elif ind == 2:
        F_heating_sma = 9.81*36 * 10**-3
        F_rest_sma = 9.81*14 * 10**-3
        I_supply = 85 * 10**-3
        R = 500
        LT = 0.4
        HT = 0.30

    elif ind == 3:
        F_heating_sma = 9.81*80 * 10**-3
        F_rest_sma = 9.81*32 * 10**-3
        I_supply = 150 * 10**-3
        R = 232
        LT = 0.8
        HT = 0.7

    elif ind == 4:
        F_heating_sma = 9.81*143 * 10**-3
        F_rest_sma = 9.81*57 * 10**-3
        I_supply = 200 * 10**-3
        R = 126
        LT = 1.1
        HT = 0.9

    elif ind == 5:
        F_heating_sma = 9.81*223 * 10**-3
        F_rest_sma = 9.81*89 * 10**-3
        I_supply = 320 * 10**-3
        R = 75
        LT = 1.6
        HT = 1.4

    elif ind == 6:
        F_heating_sma = 9.81*321 * 10**-3
        F_rest_sma = 9.81*128 * 10**-3
        I_supply = 410 * 10**-3
        R = 55
        LT = 2.0
        HT = 1.7

    elif ind == 7:
        F_heating_sma = 9.81*570 * 10**-3
        F_rest_sma = 9.81*228 * 10**-3
        I_supply = 660 * 10**-3
        R = 29
        LT = 3.2
        HT = 2.7

    elif ind == 8:
        F_heating_sma = 9.81*891 * 10**-3
        F_rest_sma = 9.81*356 * 10**-3
        I_supply = 1050 * 10**-3
        R = 18.5
        LT = 5.4
        HT = 4.5

    elif ind == 9:
        F_heating_sma = 9.81*1280 * 10**-3
        F_rest_sma = 9.81*512 * 10**-3
        I_supply = 1500 * 10**-3
        R = 12.2
        LT = 8.1
        HT = 6.8

    elif ind == 10:
        F_heating_sma = 9.81*2004 * 10**-3
        F_rest_sma = 9.81*802 * 10**-3
        I_supply = 2250 * 10**-3
        LT = 10.5
        R = 8.3
        HT = 8.8

    elif ind >= 11:
        F_heating_sma = 9.81*3560 * 10**-3
        F_rest_sma = 9.81*1424 * 10**-3
        I_supply = 4000 * 10**-3
        R = 4.3
        LT = 16.8
        HT = 114.0
    
    F = [F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    return F

def tol(value,tolerence=10**-1):
    u = value+tolerence
    l = value-tolerence
    return [u,l]

def geom_cons_1(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma = individual
    Mat = materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)
    if (l1+l2+dielectric) <bu:
        return True
    else:
        return False

def geom_cons_1_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)
    return (l1+l2+dielectric) - bu

def geom_cons_2(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.05
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)
    if (2+eps_max)*l_sma < bu:
        return True
    else:
        return False

def geom_cons_2_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.05
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)
    return (2+eps_max)*l_sma - bu

def geom_cons_3(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(1)
    if (l2/l1)>=bl:
        return True
    else:
        return False

def geom_cons_3_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    bu,bl = tol(10*10**-3)
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    return (l2/l1)-bl

def geom_cons_4(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    bu,bl = tol(l2)
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    if (l_spring)<= bu:
        return True
    else:
        return False

def geom_cons_4_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    bu,bl = tol(10*10**-3)
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    return (l_spring) - bu

def geom_cons_5(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    bu,bl = tol(2,tolerence=10**1)
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    if (l_spring - (N_spring*d_spring))/dielectric>=bl:
        return True
    else:
        return False

def geom_cons_5_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    bu,bl = tol(2)
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    return (l_spring - (N_spring*d_spring)) - bl*dielectric

def geom_cons_6(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.05
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(2*eps_max*l_sma*l2/l1)
    if l_lever>=bl:
        return True
    else:
        return False

def geom_cons_6_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.04
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(2*eps_max*l_sma*l2/l1)
    return l_lever - bl

def geom_cons_7(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.05
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(eps_max*l_sma*l2/l1)
    if l_spring**2 - (l_spring-dielectric) <= 0:
        return False
    elif np.sqrt(l_spring**2 - (l_spring-dielectric)**2)>=bl:
        return True
    else:
        return False

def geom_cons_7_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    eps_max = 0.05
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(eps_max*l_sma*l2/l1)
    if l_spring**2 - (l_spring-dielectric) < 0:
        return 1
    return np.sqrt(l_spring**2 - (l_spring-dielectric)**2) - bl

def geom_cons_8(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)
    if l_lever < bu:
        return True
    else:
        return False

def geom_cons_8_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(10*10**-3)

    return l_lever - bu

def geom_cons_9(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(6)
    if D_spring/d_spring >= bl:
        return True
    else:
        return False

def geom_cons_9_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(6)
    return D_spring/d_spring - bl

def current(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F
    
    return (10**7)*d_sma**2 + 275.92*d_sma + 0.0301

def func_cons_1(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # Mat= [material,E,G,v]
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(current(individual),tolerence=1)
    if I < bu:
        return True
    else:
        return False

def func_cons_1_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(current(individual))

    return I - bu

def func_cons_2(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # Mat= [material,E,G,v]
    bu,bl = tol(0)

    if F[1]*l1 - (F[2]*l1 + 0.41*F[0]*l2) > bl:
        return True
    else:
        return False

def func_cons_2_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(0)
   
    return F[1]*l1 - F[2]*l1 + 0.41*F[0]*l2 -bl

def func_cons_3(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # Mat= [material,E,G,v]
    bu,bl = tol(0)

    if F[2]*l1 - (F[0]*l2*0.51 )< bu:
        return True
    else:
        return False

def func_cons_3_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    bu,bl = tol(0)
    return F[2]*l1 - F[0]*l2*0.51 - bu

def func_cons_4(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F
    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # Mat= [material,E,G,v]
    vblock = 500
    eps_air = 10**6
    bu,bl = tol(vblock/eps_air)
    if 2*dielectric >=bl:
        return True
    else:
        return False

def func_cons_4_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    vblock = 500
    eps_air = 10**6
    bu,bl = tol(vblock/eps_air)
    return 2*dielectric - bl

def func_cons_5(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    # F = [F_spring,F_heating_sma,F_rest_sma,I_supply,R,LT,HT]
    # Mat= [material,E,G,v]
    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    if k > 0:
        return True
    else:
        return False

def func_cons_5_dist(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    return k

def power(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat = materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    rho = 6.45*10**3
    cp = 836.8
    h = 65.5*np.exp(-d_sma/4)*(70-25)**(1/6.0)
    k = 1+(h*np.pi*d_sma*l_sma*(25-70))/(l_sma*F[4]*I**2)
    if k < 0:
        k = 0
    t = -((rho*d_sma*cp)/(4*h))*np.log(k)
    return l_sma*F[4]*t*I**2

def height(individual):
    d_sma,l1,l2,d_spring,l_spring,D_spring,N_spring,dielectric,l_lever,I,Mat,l_sma  = individual
    Mat= materialProp(Mat)
    F = smaProp(d_sma)
    F_spring = (Mat[2]*d_spring**4)/(8*N_spring*D_spring**2) * dielectric
    F = [F_spring] + F

    z = [D_spring,d_sma]
    return z[np.argmax(z)]

def J(individual):
    return (power(individual),height(individual))

def valid(individual):
    if geom_cons_1(individual) and geom_cons_2(individual) and geom_cons_3(individual) and geom_cons_4(individual) and geom_cons_5(individual) \
        and geom_cons_6(individual) and geom_cons_7(individual) and geom_cons_9(individual) and func_cons_1(individual) \
            and func_cons_2(individual) and func_cons_3(individual) and func_cons_4(individual):

        return True
    else:
        return False

exp_output = []

for exp in experiment:
    result = J(exp)
    feasible = valid(exp)
    z = exp + [result] + [feasible]
    exp_output.append(z)
    print(z)

columns = ["d_sma","l1","l2","d_spring","l_spring","D_spring","N_spring","Dielectric","l_lever","I","Material","l_sma","Solution","Feasibility"]
experiment = pd.DataFrame(exp_output,columns=columns)
experiment.to_csv("experiment.csv")