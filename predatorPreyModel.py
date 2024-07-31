import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
beta = 0.02
gamma = 0.1
delta = 0.01
h = 0.1
t = np.arange(0, 50, h)

'''
Define the differential equations for the predator-prey model.

Parameters:
    R (float): Prey population value
    P (float): Predator population value
    
Returns:
    float: The rate of change of the prey population
    float: The rate of change of the predator population
'''
def dRdt(R, P):
    return alpha * R - beta * R * P

def dPdt(R, P):
    return delta * R * P - gamma * P

'''
Runge-Kutta method to solve the differential equation.

Parameters:
    R (float): Prey population value
    P (float): Predator population value
    
Returns:
    float: The new prey population value after time step h
    float: The new predator population value after time step h
'''
def runge_kutta(R, P):
    k1_R = h * dRdt(R, P)
    k1_P = h * dPdt(R, P)
    k2_R = h * dRdt(R + 0.5 * k1_R, P + 0.5 * k1_P)
    k2_P = h * dPdt(R + 0.5 * k1_R, P + 0.5 * k1_P)
    k3_R = h * dRdt(R + 0.5 * k2_R, P + 0.5 * k2_P)
    k3_P = h * dPdt(R + 0.5 * k2_R, P + 0.5 * k2_P)
    k4_R = h * dRdt(R + k3_R, P + k3_P)
    k4_P = h * dPdt(R + k3_R, P + k3_P)
    
    R_new = R + (1/6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
    P_new = P + (1/6) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)
    
    return R_new, P_new

'''
Plot the population graph over time.

Parameters:
    t (np.array): Time values
    P (np.array): Population values
'''
def plot_graph(t, R, P):
    plt.plot(t, R, label='Prey')
    plt.plot(t, P, label='Predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
def main():
    R = np.zeros(len(t))
    P = np.zeros(len(t))

    R[0] = 40
    P[0] = 9

    for i in range(1, len(t)):
        R[i], P[i] = runge_kutta(R[i-1], P[i-1])

    print('Estimated prey population at T = 50:', round(R[-1]))
    print('Estimated predator population at T = 50:', round(P[-1]))
    
    plot_graph(t, R, P)

if __name__ == "__main__":
    main()
