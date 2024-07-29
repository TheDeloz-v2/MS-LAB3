import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
beta = 0.02
gamma = 0.1
delta = 0.01
h = 0.1
t = np.arange(0, 50, h)

def dRdt(R, P):
    return alpha * R - beta * R * P

def dPdt(R, P):
    return delta * R * P - gamma * P

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

R = np.zeros(len(t))
P = np.zeros(len(t))

R[0] = 40
P[0] = 9

for i in range(1, len(t)):
    R[i], P[i] = runge_kutta(R[i-1], P[i-1])

plt.plot(t, R, label='Prey')
plt.plot(t, P, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

print('Estimated prey population at T = 50:', round(R[-1]))
print('Estimated predator population at T = 50:', round(P[-1]))
