import numpy as np
import math
import matplotlib.pyplot as plt

# Initial values
u = 0.8
v = 0.3
h = 0.01
n = int((100 - 0) / h)

X = np.zeros(n + 1)
U = np.zeros(n + 1)
V = np.zeros(n + 1)
du = np.zeros(n + 1)
dv = np.zeros(n + 1)


U[0] = u
V[0] = v
X[0] = 0
#U[0] = 3
#V[0] = 3

def u_derivative_func(U,V,X,i):
    return (U[i])*(1-U[i])*(U[i]-0.04) - 0.2 * U[i] * V[i]
def v_derivative_func(U,V,X,i):
    return 0.6 * U[i] * V[i] - 0.45 * V[i]

# Input function here
du[0] = u_derivative_func(U,V,X,0)
#du[0] = v
dv[0] = v_derivative_func(U,V,X,0)

# Calculate initial values using Euler's method
for i in range(min(3, n)):
    U[i+1] = U[i] + h * du[i]
    V[i+1] = V[i] + h * dv[i]
    X[i+1] = X[i] + h
    
    du[i+1] = u_derivative_func(U,V,X,i+1)
    dv[i+1] = v_derivative_func(U,V,X,i+1)
# Modifying using RK1/Euler trapezoidal method
    for j in range(10):
        U[i+1] = U[i] + h/2 * (du[i+1]+du[i])
        V[i+1] = V[i] + h/2 * (dv[i+1]+dv[i])
        #nho viet output
print(U)
print(V)
    
    
# Adams explicit method
for i in range(3, n):
    U[i + 1] = U[i] + h * (23 * du[i] - 16 * du[i-1] + 5 * du[i - 2]) / 12
    V[i + 1] = V[i] + h * (23 * dv[i] - 16 * dv[i-1] + 5 * dv[i - 2]) / 12
    X[i + 1] = X[i] + h
    du[i+1] = u_derivative_func(U,V,X,i+1)
    dv[i+1] = v_derivative_func(U,V,X,i+1)
    #nho viet output
# Modify with Adams Implicit method
    for j in range(20):
        V[i + 1] = V[i] + h * (9 * dv[i+1] + 19 * dv[i] - 5 * dv[i-1] + 1 * dv[i - 2]) / 24
    #nho viet output
        U[i + 1] = U[i] + h * (9 * du[i+1] + 19 * du[i] - 5 * du[i-1] + 1 * du[i - 2]) / 24
    #nho viet output

print(U)
print(V)
print(np.max(U), np.min(U))
print(np.max(V), np.min(V))

# Plotting U values against X
plt.plot(X, U)
plt.xlabel('X')
plt.ylabel('Q')
plt.title('Plot of U values')
plt.grid(True)
plt.show()

plt.plot(X, V)
plt.xlabel('X')
plt.ylabel('I')
plt.title('Plot of V values')
plt.grid(True)
plt.show()