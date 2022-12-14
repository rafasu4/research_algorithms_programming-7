import numpy as np
import cvxpy as cp
import random
import time
import matplotlib.pyplot as plt

def create_matrix(total_equations):
    matrix = np.random.rand(total_equations, total_equations)
    return matrix

def create_constants(value):
    constants = np.random.rand(value)
    return constants


total_equations = random.randint(1, 10000)
coefficients = create_matrix(total_equations)
constants = create_constants(total_equations)


# Get the current time in seconds
start_time = time.time()

np.linalg.solve(coefficients, constants)

# Get the current time in seconds again
end_time = time.time()

# Calculate the difference between the start and end times
elapsed_time = end_time - start_time

# Print the matrix
print(elapsed_time)

x = total_equations
y = elapsed_time

fig, ax = plt.subplots()

# Use the plot() method on the axes object to draw a line graph
ax.plot(x, y)


# Show the plot
plt.show()

# Generate a random non-trivial linear program.
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()
