'''
Lab 3 - Modeling and Simulation
Date: 07/2024

Implementation of a logistic growth model simulation using the Runge-Kutta method.
'''

import numpy as np
import matplotlib.pyplot as plt

P0 = 10
r = 0.1
K = 1000
h = 0.1
T = 20

'''
Define the differential equation for the population growth.

Parameters:
    P (float): Population value
    
Returns:
    float: The rate of change of the population
'''
def dPdt(P):
    return r * P * (1 - P / K)

'''
Runge-Kutta method to solve the differential equation.

Parameters:
    P (float): Population value
    
Returns:
    float: The new population value after time step h
'''
def runge_kutta(P):
    k1 = h * dPdt(P)
    k2 = h * dPdt(P + 0.5 * k1)
    k3 = h * dPdt(P + 0.5 * k2)
    k4 = h * dPdt(P + k3)
    return P + (k1 + 2 * k2 + 2 * k3 + k4) / 6

'''
Plot the population graph over time.

Parameters:
    t (np.array): Time values
    P (np.array): Population values
'''
def plot_graph(t, P):
    plt.plot(t, P, label='Population over Time')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Logistic Growth Model Simulation')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def main():
    t = np.arange(0, T + h, h)
    P = np.zeros(len(t))
    P[0] = P0

    for i in range(1, len(t)):
        P[i] = runge_kutta(P[i-1])

    print(f"Estimated Population at T=20: {round(P[-1])}")
    
    plot_graph(t, P)

if __name__ == "__main__":
    main()
