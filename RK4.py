import matplotlib.pyplot as plt
import numpy as np
import math

# Define the ODE functions
def f(x, y, t):
    return x*(1-x)*(x-0.04) - 0.2 * x * y

def g(x, y, t):
    return 0.6 * x * y - 0.45 * y

# Set initial conditions and integration parameters
t0 = 0
#y0 = 4
#x0 = 6
h = 0.01
x0 = 0.8
y0 = 0.3
num_steps = int(100/h)

# Solve the ODE using RK4
T = np.zeros(num_steps)
Y = np.zeros(num_steps)
X = np.zeros(num_steps)   

T[0] = t0
Y[0] = y0
X[0] = x0

for i in range(num_steps-1):
    k1u = h * f(X[i], Y[i],T[i])
    k1v = h * g(X[i], Y[i],T[i])

    k2u = h * f(X[i] + 0.5 * k1u, Y[i] + 0.5 * k1v, T[i] + h/2)
    k2v = h * g(X[i] + 0.5 * k1u, Y[i] + 0.5 * k1v, T[i] + h/2)

    k3u = h * f(X[i] + 0.5 * k2u, Y[i] + 0.5 * k2v, T[i] + h/2)
    k3v = h * g(X[i] + 0.5 * k2u, Y[i] + 0.5 * k2v, T[i] + h/2)

    k4u = h * f(X[i] + k3u, Y[i] + k3v, T[i] + h)
    k4v = h * g(X[i] + k3u, Y[i] + k3v, T[i] + h)

    X[i+1] = X[i] + (k1u + 2 * k2u + 2 * k3u + k4u) / 6
    Y[i+1] = Y[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    T[i+1] = T[i] + h

#Result output
print(X)
print(Y)

print (np.max(X))
print (np.min(X))
print (np.max(Y))
print (np.min(Y))


# Plotting the results
plt.plot(T, Y, color='red', label='Y - Correlation')
plt.xlabel('t')
plt.ylabel('Y')
plt.legend()
plt.show()

plt.plot(T, X, color='red', label='X - Correlation')
plt.xlabel('t')
plt.ylabel('X')
plt.legend()
plt.show()

plt.plot(X, Y, color='red', label='X-Y - Correlation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
