# Program to translate real-time electricity prices into signals that can be transmitted to occupants of buildings 
# to influence their participation in a social game
# Aim: load shaping to ensure demand curve is shaped like an inverted price curve
# Output: Continuously varying real number

# Price curves: https://openei.org/apps/USURDB/rate/view/5cbf78b25457a34e40671081#3__Energy
# Sample demand curves: https://github.com/buds-lab/the-building-data-genome-project
# (selected office building: Office_Elizabeth)
# Sample PV: generated using https://pvwatts.nrel.gov

# Net demand = building load (occupant controlled loads + fixed HVAC and other loads) - onsite PV generation

import csv  
import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt

pv = np.array([])
price = np.array([])
demand = np.array([])

with open('building_data.csv', encoding='utf8') as csvfile:
# with open('../../building_data.csv', encoding='utf8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader,None)
    rowcount = 0
    for row in csvreader:
        pv = np.append(pv, 0.001*float(row[3])) # Converting Wh to kWh
        price = np.append(price, float(row[2])) # Cost per kWh
        val = row[5]
        if val in (None,""): #How to treat missing values
            val = 0
        else:
            val = float(val) # kWh
        demand = np.append(demand, val)
        rowcount+=1
        # if rowcount>100:
        #     break

pvsize = 5 #Assumption

netdemand = demand.copy()
for i in range(len(demand)):
    netdemand[i] = demand[i] - pvsize*pv[i]

# Data starts at 5 am on Jan 1
day = 45
netdemand_24 = netdemand[24*day-5:24*day+19]
price_24 = price[24*day-5:24*day+19]
pv_24 = pv[24*day-5:24*day+19]
demand_24 = demand[24*day-5:24*day+19]

# Calculate optimal load scheduling. 90% of load is fixed, 10% is controllable.
def optimise_24h(netdemand_24, price_24):
    currentcost = netdemand_24*price_24
    
    fixed_load = 0.9*netdemand_24
    controllable_load = sum(0.1*netdemand_24)
    # fixed_load = 0*netdemand_24
    # controllable_load = sum(netdemand_24)
    
    def objective(x):
        load = fixed_load + x
        cost = np.multiply(price_24,load)
        # Negative demand means zero cost, not negative cost
        # Adding L1 regularisation to penalise shifting of occupant demand
        lambd = 0.005
        return sum(np.maximum(cost,0)) + lambd*sum(abs(x-0.1*netdemand_24))

    def constraint_sumofx(x):
        return sum(x) - controllable_load
    
    def constraint_x_positive(x):
        return x 

    x0 = np.zeros(24)
    cons = [
        {'type':'eq', 'fun': constraint_sumofx},
        {'type':'ineq', 'fun':constraint_x_positive}
    ]
    sol = minimize(objective, x0, constraints=cons)
    print(sol)
    return sol

sol = optimise_24h(netdemand_24,price_24)
x = sol['x']
plt.plot(netdemand_24, color='r')
plt.plot(x + 0.9*netdemand_24, color='b')
# plt.plot(x,color='b')
plt.plot(demand_24,color='m')
plt.plot(price_24*100, color='g')
plt.plot(pv_24*pvsize,color='y')
plt.show()

# Signal should be according to what the optimal shifted controllable load is (x)
signal = x / np.linalg.norm(x)
print(signal)
plt.plot(signal,'yellow')
plt.show()