
from Ranmath import TimeSeriesMatrix

import builtins
import matplotlib.pyplot as plt, numpy as np

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

print("Importing")

matrix.from_CSV("data.csv")
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Loaded data [first 5 plots]")

plt.show(plot)

print("Generating")

matrix.generate.exponential_decay(5, 100, 3.5)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Exponential Decay [first 5 plots]")
plt.show(plot)

matrix.generate.inverse_wishart(50, 100, 0.3)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Inverse Wishart [first 5 plots]")
plt.show(plot)

print("Exporting")

matrix.to_CSV("exported.csv")
array = matrix.to_ndarray()
print(array)

print("Normalizing")

matrix.normalize.outlier(...)
matrix.normalize.standard(...)
matrix.normalize.winsorization(...)

print("Getting characteristics")

covariance = matrix.characteristics.covariance_cube(...)
eigenValues = matrix.characteristics.autocorrelation_eigenvalues(...)
eigenVectors = matrix.characteristics.autocorrelation_eigenvectors(...)

