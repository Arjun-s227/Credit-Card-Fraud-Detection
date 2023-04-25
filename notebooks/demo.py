import numpy as np

# Initialization
np.random.seed(0)
x1 = np.array([1, 1, 1, 1]) # s1(p)
x2 = np.array([1, 1, -1, -1]) # s1(p)
xi = np.array([1, 1, 1, 1]) # input
print("The input vector is:")
print(xi)

# Hebbian learning rule
# w_ij = summation si(p) * sj(p) for P patterns
w = np.outer(x1, x1) + np.outer(x2, x2)
np.fill_diagonal(w, 0)

# Update loop
stop = True
y = xi.copy() # yi = xi
while stop:
    update = [0, 3, 2, 1] # update order
    for i in update:
        yin = xi[i] + np.dot(y, w[:, i]) # update of weights
        if yin > 0:
            y[i] = 1
        else:
            y[i] = -1

    # Check for convergence
    if np.array_equal(y, x1):
        print("Convergence has reached to pattern 1")
        stop = False
        print("The converged output is:")
        print(y)
        print("The weight matrix is:")
        print(w)
    elif np.array_equal(y, x2):
        print("Convergence has reached to pattern 2")
        stop = False
        print("The converged output is:")
        print(y)
        print("The weight matrix is:")
        print(w)
