
from Ranmath import TimeSeriesMatrix

import builtins
import matplotlib.pyplot as plt, numpy as np

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

print("Importing")

matrix.fromCSV("data.csv")
print(matrix.array)

plot = plt.plot(matrix.array), plt.xlabel("Loaded data [first 5 plots]")

plt.show(plot)

print("Generating")

matrix.generate.exponentialDecay(50, 100, 3.5)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Exponential Decay [first 5 plots]")
plt.show(plot)

matrix.generate.inverseWishart(50, 100, 0.3)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Inverse Wishart [first 5 plots]")
plt.show(plot)

print("Exporting")

matrix.toCSV("exported.csv")
array = matrix.toNDArray()
print(array)

print("Normalizing")

matrix.normalize.outlier(...)
matrix.normalize.standard(...)
matrix.normalize.winsorization(...)

print("Getting characteristics")

covariance = matrix.characteristics.covarianceCube(...)
eigenValues = matrix.characteristics.eigenValues(...)
eigenVectors = matrix.characteristics.eigenVectors(...)

