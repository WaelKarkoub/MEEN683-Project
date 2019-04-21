import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from bitstring import BitArray
vblock = 500
eps_air = 10**6

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

    d_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]

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
def list_set(d,mx,mn):
    val = 0
    l =[]
    counter = 0
    while True:
        if val == 0 and counter == 0:
            val=val+mn
        else:
            val=val+d

        if val > mx:
            break
        l.append(val)
        counter+=1
        print(counter)
    
    return l

doe = pd.DataFrame()
d_sma = [0.025*10**-3, 0.038*10**-3, 0.05*10**-3, 0.076*10**-3, 0.1*10**-3, 0.13*10**-3, 0.15*10**-3, 0.20*10**-3, 0.25*10**-3, 0.31*10**-3, 0.38*10**-3,0.51*10**-3]
ds_l1 = pd.DataFrame()
doe["d_sma"] = d_sma
l1 = []
l1_max = (10*10**-3)/2 
l1_min = 0.5*10**-3
dl1 = ((10*10**-3)/2 - 0.5*10**-3)/2**12

ds_l1["l1"] = list_set(dl1,l1_max,l1_min)
ds_l1["l2"] = list_set(dl1,l1_max,l1_min)

dd_spring = ((2*10**-3)/2 - 0.025*10**-3)/2**12
d_spring_max = 10**-3
d_spring_min = 0.025*10**-3
ds_d_spring = pd.DataFrame()
ds_d_spring["d_spring"] = list_set(dd_spring,d_spring_max,d_spring_min)

dl_spring = ((10*10**-3)/2 - 0.0001*10**-3)/2.**12
d_l_spring_max = 5*10**-3
d_l_spring_min = 0.0001*10**-3
ds_l_spring = pd.DataFrame()
ds_l_spring["l_spring"] = list_set(dl_spring,d_l_spring_max,d_l_spring_min)

dD_spring = ((10*10**-3)/2 - 0.05*10**-3)/2**12
d_D_spring_max = 5*10**-3
d_D_spring_min = 0.05*10**-3
ds_D_spring = pd.DataFrame()
ds_D_spring["D_spring"] = list_set(dD_spring,d_D_spring_max,d_D_spring_min)

dN_spring = 1
d_N_spring_max = 15
d_N_spring_min = 1
ds_N_spring = pd.DataFrame()
ds_N_spring["N_spring"] = list_set(dN_spring,d_N_spring_max,d_N_spring_min)

d_dielectric = ((10*10**-3)/4 - vblock/eps_air)/2**12
d_die_max = (10*10**-3)/4
d_die_min = vblock/eps_air
ds_die = pd.DataFrame()
ds_die["Dielectric"] = list_set(d_dielectric,d_die_max,d_die_min)

print("dlever")
dl_lever = ((10*10**-3)/2 - 0)/2**12
d_l_lever_max = 5*10**-3
d_l_lever_min = 0
ds_l_lever = pd.DataFrame()
ds_l_lever["l_lever"] = list_set(dl_lever,d_l_lever_max,d_l_lever_min)


dI = (4000 * 10**-3 - 5*10**-3)/2**12
d_I_max = 4000 * 10**-3
d_I_min = 5*10**-3
ds_I = pd.DataFrame()
ds_I["I"] = list_set(dI,d_I_max,d_I_min)

dMat = 1
d_Mat_max = 4
d_mat_min = 0
ds_Mat = pd.DataFrame()
ds_Mat["Material"] = list_set(dMat,d_Mat_max,d_mat_min)

doe = pd.concat([doe, ds_l1,ds_d_spring,ds_l_spring,ds_D_spring,ds_N_spring,ds_die,ds_l_lever,ds_I,ds_Mat], axis=1)
print("saving") 
doe.to_csv('full_doe.csv', index=False)

data = []
for i in doe["d_sma"]:
    if not np.isnan(i):
        l1 = doe["l1"].values.tolist()
        l2 = doe["l2"].values.tolist()
        d_spring = doe["d_spring"].values.tolist()
        l_spring = doe["l_spring"].values.tolist()
        D_spring = doe["D_spring"].values.tolist()
        N_spring = doe["N_spring"].values.tolist()
        Dielectric = doe["Dielectric"].values.tolist()
        l_lever = doe["l_lever"].values.tolist()
        I = doe["I"].values.tolist()
        Mat = doe["Material"].values.tolist()

        exp = [i,random.choice(l1),random.choice(l2),random.choice(d_spring),random.choice(l_spring),random.choice(D_spring),\
            random.choice(N_spring),random.choice(Dielectric),random.choice(l_lever),random.choice(I),random.choice(Mat)]
        
        data.append(exp)
        

columns = ["d_sma","l1","l2","d_spring","l_spring","D_spring","N_spring","Dielectric","l_lever","I","Material"]
random_doe = pd.DataFrame(data,columns = columns)
random_doe.to_csv('random_doe.csv', index=False)
        
    

