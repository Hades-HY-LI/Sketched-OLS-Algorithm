import numpy as np
import pandas as pd
import timeit
from sympy.discrete.transforms import fwht
from numba import jit
from tabulate import tabulate

# Setup Walsh-Hadamard transform
@jit(nopython=True)
def fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

# Define Sketched_OLS function
def sketched_ols(X, y, epsilon= 0.1):
    n, d = X.shape
    r = int(d * np.log(n) / epsilon)  # Compute r
    # Generate S matrix
    t = np.random.randint(n, size=r)
    # Generate diagonal values of D matrix
    D_val = np.random.choice([1, -1], size=n).reshape(n, 1)
    # For time saving, we use element-wise product by diagonal value of D directly
    DX = D_val * X
    Dy = D_val * y
    # S.T @ HDX is to get the t-th value of corresponding column of HDX
    X_star = np.sqrt(n / r) * np.apply_along_axis(fwht, 0, DX)[t, :]
    y_star = np.sqrt(n / r) * np.apply_along_axis(fwht, 0, Dy)[t]
    start = timeit.default_timer()
    b_sketched = np.linalg.inv(X_star.T @ X_star) @ X_star.T @ y_star
    stop = timeit.default_timer()
    return b_sketched, stop - start

# Generate design matrix X and response y
X = np.random.uniform(size=(1048576*20)).reshape(1048576, 20)
y = np.random.uniform(size=(1048576)).reshape(1048576, 1)

# Calculate beta using the original form
start = timeit.default_timer()
b = np.linalg.inv(X.T @ X) @ X.T @ y
stop = timeit.default_timer()

# Calculate running time and squared error of beta given different level of epsilon
epsilon = np.array([.1, .05, .01, .001])
eps = epsilon.shape[0]
Bs = np.empty((20, 4))
Time_s = []
B_error = []
for i in np.arange(eps):
    bs, time_s = sketched_ols(X, y, epsilon[i])
    b_error = np.sum((b - bs) ** 2)
    Bs[:, i] = bs.reshape(20)
    Time_s = np.append(Time_s, time_s)
    B_error = np.append(B_error, b_error)

# Print the result in pd.Dataframe
Q6_data = np.array([np.append(stop - start, Time_s), np.append(0, B_error)])
Q6_df = pd.DataFrame(data=Q6_data,
                     index=['Running time', 'Squared error'],
                     columns=['Original', 'epsilon = .1', 'epsilon = .05', 'epsilon = .01', 'epsilon = .001'])
print(Q6_df)

# Print the result in table
Q6_table = [['Original', 'epsilon = .1', 'epsilon = .05', 'epsilon = .01', 'epsilon = .001'],
            np.append(stop - start, Time_s),
            np.append(0, B_error)]
print(tabulate(Q6_table))

