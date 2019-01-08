
from Ranmath import TimeSeriesMatrix
from Ranmath.MatrixReconstructors import SingleMatrixReconstructor
from Ranmath.CorrelationEstimators import LinearShrinkageEstimator, LedoitPecheRIEstimator, QuarticRIEstimator, OracleEstimator
from Ranmath.Norms import frobenius_eigenvalues_distance, frobenius_norm_squared
import matplotlib.pyplot as plt, numpy as np


def test_generating_inverse_wishart(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, iterations, normalise_covariance=True, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration "+str(iteration)+" [first 5 assets]")
        plt.show(plot)


def test_generating_exp_decay(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.exponential_decay(150, 200, 4.75, iterations, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration "+str(iteration)+" [first 5 assets]")
        plt.show(plot)


def test_saving_and_reading_from_csv():
    matrix = TimeSeriesMatrix()
    matrix.generate.exponential_decay(150, 200, 4.75, 30, verbose=True)
    generated = matrix.array
    matrix.export_matrix.to_CSV("exported.csv")
    matrix.import_matrix.from_CSV("exported.csv")
    imported = matrix.array
    if np.array_equiv(generated[:0, :, :], imported):
        print("PASSED, array exported and loaded correctly")
    else:
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
        axs[0].plot(generated[0, :3, :].T)
        axs[1].plot(imported[0, :3, :].T)
        plt.show()


def test_rolling_window():
    matrix = TimeSeriesMatrix()
    matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 10, verbose=True)
    print(matrix.characteristics.rw_autocorrelation_eigenvalues(10, 10, verbose=True).sample_eigenvalues.shape)
    print(matrix.characteristics.rw_autocorrelation_eigenvectors(50, 50, verbose=True).sample_eigenvectors.shape)
    print(matrix.characteristics.rw_sample_estimator_cubes(30, 30, verbose=True).sample_cube.shape)
    print(matrix.characteristics.rw_data_cubes(40, 40, verbose=True).sample_cube.shape)


def test_global_sampler():
    matrix = TimeSeriesMatrix()
    matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 10, verbose=True)
    print(matrix.characteristics.global_eigenvalues(verbose=True).shape)
    print(matrix.characteristics.global_eigenvectors(verbose=True).shape)
    print(matrix.characteristics.global_sample_estimator_cube(verbose=True).shape)


def test_lse_optimal_alphas():
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, 30, normalise_covariance=True, verbose=True)

    n_iter, N, T = matrix.array.shape

    sample_eigenvalues = matrix.characteristics.global_eigenvalues(verbose=True)
    sample_eigenvectors = matrix.characteristics.global_eigenvectors(verbose=True)

    ls_estimator = LinearShrinkageEstimator()

    alpha_oracle = ls_estimator.get_lse_alpha_oracle(sample_eigenvalues,
                                                     sample_eigenvectors,
                                                     matrix.generate.last_C)
    alpha_bonafide = ls_estimator.get_lse_alpha_bonafide(sample_eigenvalues,
                                                         sample_eigenvectors,
                                                         matrix.array)
    true_C = matrix.generate.last_C


    alpha_list = np.arange(0., 1.01, 0.01)
    to_plot = np.zeros(len(alpha_list))

    reconstructor = SingleMatrixReconstructor()

    for i in range(len(alpha_list)):
        frobenius_values = []
        estimated_eigenvalues = ls_estimator.estimate_eigenvalues(sample_eigenvalues,
                                                                  1. - alpha_list[i], alpha_list[i])
        for iteration in range(n_iter):
            estimated_C = reconstructor.reconstruct(sample_eigenvectors[iteration],
                                                    estimated_eigenvalues[iteration])
            frobenius_values.append(frobenius_norm_squared(estimated_C - true_C))

        to_plot[i] = np.array(frobenius_values).mean()

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
    axs.set_xlabel('α')
    axs.set_ylabel('frobenius_norm_squared(LSE estimated C - true C)')
    axs.scatter(alpha_list, to_plot, s=10)

    print("Oracle optimal α: "+str(alpha_oracle[1].real))
    print("Bonafide optimal α: "+str(alpha_bonafide[1].real))
    print("Simulation optimal α: "+str(alpha_list[np.argmin(to_plot)]))
    plt.show()


def test_ledoit_peche_rie():

    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, 30, normalise_covariance=True, verbose=True)

    n_iter, N, T = matrix.array.shape

    sample_eigenvalues = matrix.characteristics.global_eigenvalues(verbose=True)
    sample_eigenvectors = matrix.characteristics.global_eigenvectors(verbose=True)

    lp_rie_estimator = LedoitPecheRIEstimator()

    optimal_q = lp_rie_estimator.get_optimal_q(N, T)
    eta_scale = lp_rie_estimator.get_optimal_eta_scale(N)
    eta_factor_list = np.arange( -3. , 1.05 , 0.05 )

    to_plot = np.zeros(len(eta_factor_list))

    true_C = matrix.generate.last_C

    reconstructor = SingleMatrixReconstructor()

    for i in range(len(eta_factor_list)):

        frobenius_values = []
        eta = eta_scale * 10 ** eta_factor_list[i]

        estimated_eigenvalues = lp_rie_estimator.estimate_eigenvalues(sample_eigenvalues, eta, optimal_q)

        for iteration in range(n_iter):
            estimated_C = reconstructor.reconstruct(sample_eigenvectors[iteration],
                                                    estimated_eigenvalues[iteration])
            frobenius_values.append(frobenius_norm_squared(estimated_C - true_C))

        to_plot[i] = np.array(frobenius_values).mean()

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
    axs.set_xlabel('η factor')
    axs.set_ylabel('frobenius_norm_squared(LP RIE estimated C - true C)')
    axs.scatter(eta_factor_list, to_plot, s=10)

    print("Simulation optimal η factor: " + str(eta_factor_list[np.argmin(to_plot)]))
    plt.show()


if __name__ == '__main__':
    print("Started")

    # test_generating_exp_decay(5) <- works
    # test_saving_and_reading_from_csv() <- works
    # test_saving_and_reading_from_csv() <- works
    # test_rolling_window() <- works
    # test_global_sampler() <- works
    # test_lse_optimal_alphas() <- works
    # test_ledoit_peche_rie() <- works




