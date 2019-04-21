import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from bitstring import BitArray


def decode(individual):
    vblock = 500
    eps_air = 10**6

    
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

doe = pd.DataFrame()
d_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.1*10**-3, 0.13*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]
ds_l1 = pd.DataFrame()
doe["d_sma"] = d_sma
l1 = []
l1_max = (10*10**-3)/2 
l1_min = 0.5*10**-3
dl1 = ((10*10**-3)/2 - 0.5*10**-3)/2**12

def list_set(d,mx,mn):
    val = 0
    l =[]
    while True:
        if val == 0:
            val=val+dl1+mn
        
        else:
            val=val+dl1
        if val > mx:
            break
        l.append(val)
    
    return l

ds_l1["l1"] = list_set(dl1,l1_max,l1_min)
ds_l1["l2"] = list_set(dl1,l1_max,l1_min)

dd_spring = ((2*10**-3)/2 - 0.025*10**-3)/2**12
d_spring_max = 2*10**-3
d_spring_min = 0.025*10**-3
ds_d_spring = pd.DataFrame()
ds_d_spring["d_spring"] = list_set(dd_spring,d_spring_max,d_spring_min)


# ds_d_spring["d_spring"] = ld_spring
print(ds_d_spring.head)

# dl_spring = ((10*10**-3)/2 - 0.0001*10**-3)/2**12
# dD_spring = ((10*10**-3)/2 - 0.05*10**-3)/2**12
# dN_spring = 1
# d_dielectric = ((10*10**-3)/4 - vblock/eps_air)/2**12
# dl_lever = ((10*10**-3)/2 - 0)/2**12
# dI = (4000 * 10**-3 - 5*10**-3)/2**12

doe = pd.concat([doe, ds_l1,ds_d_spring], axis=1) 


print(doe.head)
