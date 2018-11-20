
from Ranmath import TimeSeriesMatrix
from Ranmath.MatrixReconstructors import SingleMatrixReconstructor

import builtins
import matplotlib.pyplot as plt, numpy as np

print = lambda x: builtins.print(x, end='\n\n')

print("Creation")

matrix = TimeSeriesMatrix()

print("Importing")

#matrix.from_CSV("data.csv")

#plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Loaded data [first 5 plots]")

#plt.show(plot)

matrix.generate.inverse_wishart(5, 100, 3.5)

#macierz kowariancji, pierwsze okno, sample
print(matrix.characteristics.rw_covariance_cubes(1, 0).sample_cube[0])

#print(matrix.array)

autocorr_eigvec = matrix.characteristics.rw_autocorrelation_eigenvectors(1, 0)

autocorr_eigval = matrix.characteristics.rw_autocorrelation_eigenvalues(1, 0)

print("SAMPLE")

#to powinno dac przy generatorach rzeczywista macierz bedaca macierza kowariancji z pierwszego okna, (sample)

reconstructor = SingleMatrixReconstructor(autocorr_eigvec.sample_eigenvectors[0],
                                          autocorr_eigval.sample_eigenvalues[0])

print(reconstructor.reconstruct())

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Inverse wishart for 5 assets")

plt.show(plot)

print("Generating")

matrix.generate.exponential_decay(50, 100, 3.5)
print(matrix.array)

plot = plt.plot(matrix.array[:5, :].T), plt.xlabel("Exponential Decay [first 5 plots]")
plt.show(plot)

matrix.generate.inverse_wishart(50, 100, 0.3)
print(matrix.array)

plot = plt.plot(matrix.array[:1, :].T), plt.xlabel("Inverse Wishart [first 5 plots]")
plt.show(plot)

print("Exporting")

matrix.to_CSV("exported.csv")
array = matrix.to_ndarray()
print(array)

print("Getting characteristics")

covariance = matrix.characteristics.rw_covariance_cubes(3, 4)
eigenValues = matrix.characteristics.rw_autocorrelation_eigenvalues(3, 4)
eigenVectors = matrix.characteristics.rw_autocorrelation_eigenvectors(3, 4)

print("Normalizing")

matrix.normalize.standard()

plot = plt.plot(matrix.array[:1, :].T), plt.xlabel("Inverse Wishart [first 5 plots] normalized")
plt.show(plot)

matrix.normalize.outlier()

matrix.normalize.winsorization()
