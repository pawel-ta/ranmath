from Ranmath import TimeSeriesMatrix
from Ranmath.MatrixReconstructors import SingleMatrixReconstructor
from Ranmath.CorrelationEstimators import LinearShrinkageEstimator, LedoitPecheRIEstimator, QuarticRIEstimator, \
    OracleEstimator, EigenvaluesClippingEstimator
from Ranmath.Norms import frobenius_eigenvalues_distance, frobenius_norm_squared
import scipy.linalg as la
import builtins
import matplotlib.pyplot as plt, numpy as np
from Ranmath.Resolvents.C1A1WishartResolvent import C1A1WishartResolvent
from Ranmath.Resolvents.ExponentialDecayResolvent import ExponentialDecayResolvent
from Ranmath.Resolvents.InverseWishartResolvent import InverseWishartResolvent
from Ranmath.Resolvents.InvWishExpDecayMixedResolvent import InvWishExpDecayMixedResolvent
from Ranmath.Resolvents.SimulatedEigenvaluesResolvent import SimulatedEigenvaluesResolvent

def test_generating_inverse_wishart(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, iterations, normalize_covariance=True, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration " + str(iteration) + " [first 5 assets]")
        plt.show(plot)


def test_generating_exp_decay(iterations):
    matrix = TimeSeriesMatrix()
    matrix.generate.exponential_decay(150, 200, 4.75, iterations, verbose=True)
    print(matrix.array.shape)
    for iteration in range(iterations):
        plot = plt.plot(matrix.array[iteration, :5, :].T)
        plt.xlabel("Inverse Wishart iteration " + str(iteration) + " [first 5 assets]")
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

def test_eigenvalues_clipping():
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, 1, normalize_covariance=True, verbose=True)
    # C = matrix.generate.last_C
    # matrix.generate.exponential_decay(200, 200, 4.5, 1)
    # A = matrix.generate.last_A
    # matrix.generate.multivariate_gaussian(C, A, 1)
    # matrix.generate.multivariate_gaussian(np.eye(400), np.eye(200), 1)

    plot_type = 'C=IW, A=ED'


    # C = np.eye(200)
    # for i in range(C.shape[0]):
    #    if i > C.shape[0]/2:
    #        C[i, i] = 2
    # matrix.generate.multivariate_gaussian(C, np.eye(200), 30)

    n_iter, N, T = matrix.array.shape

    sample_eigenvalues = matrix.characteristics.global_eigenvalues(verbose=True)
    sample_eigenvectors = matrix.characteristics.global_eigenvectors(verbose=True)

    cl_estimator = EigenvaluesClippingEstimator()

    true_C = matrix.generate.last_C

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)

    estimated_eigenvalues = cl_estimator.estimate_eigenvalues(sample_eigenvalues, cl_estimator.get_optimal_q(N, T)).mean(axis=0)


    true_C_eigenvalues = la.eigvals(true_C)
    eigenvales_numbers = np.array([i for i in range(true_C_eigenvalues.shape[0])])

    axs.scatter(eigenvales_numbers, true_C_eigenvalues, c='blue', label='Eigenvalues of true C', marker='.')
    axs.scatter(eigenvales_numbers, estimated_eigenvalues, c='green',
                      label='Eigenvalues after clipping procedure', marker='.')
    axs.set_xlabel('λk [kth eigenvalue]')
    axs.set_ylabel('value')
    axs.set_title('\nEigenvalues Clipping Estimator ['+plot_type+']\n'
                       )
    axs.legend(loc='upper right')

    plt.show()

def test_lse_optimal_alphas():
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(200, 200, 0.3, 30, normalize_covariance=True, verbose=True)
    C = matrix.generate.last_C
    matrix.generate.exponential_decay(200, 200, 4.5, 30)
    A = matrix.generate.last_A
    matrix.generate.multivariate_gaussian(C, A, 30)
    #matrix.generate.multivariate_gaussian(np.eye(200), np.eye(200), 30)

    plot_type = 'C=IW, A=ED'


    # C = np.eye(200)
    # for i in range(C.shape[0]):
    #    if i > C.shape[0]/2:
    #        C[i, i] = 2
    # matrix.generate.multivariate_gaussian(C, np.eye(200), 30)

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

    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)

    axs[0, 0].set_xlabel('α')
    axs[0, 0].set_ylabel('F(LSE estimated C - true C)')
    axs[0, 0].set_title('Ledoit-Wolf LSE Frobenius norm for varying α ['+plot_type+']\n'
                        'min(F)= '+' '+str(np.min(to_plot)))
    axs[0, 0].scatter(alpha_list, to_plot, s=10)

    estimated_eigenvalues_sim_alpha = ls_estimator.estimate_eigenvalues(sample_eigenvalues,
                                                                        1. - alpha_list[np.argmin(to_plot)],
                                                                        alpha_list[np.argmin(to_plot)]).mean(axis=0)
    estimated_eigenvalues_oracle_alpha = ls_estimator.estimate_eigenvalues(sample_eigenvalues,
                                                                           1. - alpha_oracle[1].real,
                                                                           alpha_oracle[1].real).mean(axis=0)
    estimated_eigenvalues_bonafide_alpha = ls_estimator.estimate_eigenvalues(sample_eigenvalues,
                                                                             1. - alpha_bonafide[1].real,
                                                                             alpha_bonafide[1].real).mean(axis=0)
    true_C_eigenvalues = la.eigvals(true_C)
    eigenvales_numbers = np.array([i for i in range(true_C_eigenvalues.shape[0])])

    axs[0, 1].scatter(eigenvales_numbers, true_C_eigenvalues, c='blue', label='Eigenvalues of true C', marker='.')
    axs[0, 1].scatter(eigenvales_numbers, estimated_eigenvalues_sim_alpha, c='green',
                   label='Ledoit-Wolf LSE for simulation optimal α', marker='.')
    axs[0, 1].set_xlabel('λk [kth eigenvalue]')
    axs[0, 1].set_ylabel('value')
    axs[0, 1].set_title('Ledoit-Wolf LSE for simulation α ['+plot_type+']\n'
                        'α =' + ' ' + str(alpha_list[np.argmin(to_plot)]))
    axs[0, 1].legend(loc='upper right')

    axs[1, 0].scatter(eigenvales_numbers, true_C_eigenvalues, c='blue', label='Eigenvalues of true C', marker='.')
    axs[1, 0].scatter(eigenvales_numbers, estimated_eigenvalues_bonafide_alpha, c='green',
                      label='Ledoit-Wolf LSE for bonafide optimal α', marker='.')
    axs[1, 0].set_xlabel('λk [kth eigenvalue]')
    axs[1, 0].set_ylabel('value')
    axs[1, 0].set_title('\nLedoit-Wolf LSE for bonafide α ['+plot_type+']\n'
                        'α =' + ' ' + str(alpha_bonafide[1].real))
    axs[1, 0].legend(loc='upper right')

    axs[1, 1].scatter(eigenvales_numbers, true_C_eigenvalues, c='blue', label='Eigenvalues of true C', marker='.')
    axs[1, 1].scatter(eigenvales_numbers, estimated_eigenvalues_oracle_alpha, c='green',
                      label='Ledoit-Wolf LSE for oracle optimal α', marker='.')
    axs[1, 1].set_xlabel('λk [kth eigenvalue]')
    axs[1, 1].set_ylabel('value')
    axs[1, 1].set_title('\nLedoit-Wolf LSE for oracle α ['+plot_type+']\n'
                        'α =' + ' ' + str(alpha_oracle[1].real))
    axs[1, 1].legend(loc='upper right')

    print("Oracle optimal α: " + str(alpha_oracle[1].real))
    print("Bonafide optimal α: " + str(alpha_bonafide[1].real))
    print("Simulation optimal α: " + str(alpha_list[np.argmin(to_plot)]))
    plt.show()


def test_ledoit_peche_rie():
    matrix = TimeSeriesMatrix()
    matrix.generate.inverse_wishart(150, 200, 0.3, 30, normalize_covariance=True, verbose=True)
    C = matrix.generate.last_C
    matrix.generate.exponential_decay(150, 200, 4.5, 30)
    A = matrix.generate.last_A
    matrix.generate.multivariate_gaussian(C, A, 30)
    #matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 30)

    plot_type = 'C=IW, A=ED'


    # C = np.eye(150)
    # for i in range(C.shape[0]):
    #  if i > C.shape[0]/2:
    #        C[i, i] = 2
    # matrix.generate.multivariate_gaussian(C, np.eye(200), 30)

    n_iter, N, T = matrix.array.shape

    sample_eigenvalues = matrix.characteristics.global_eigenvalues(verbose=True)
    sample_eigenvectors = matrix.characteristics.global_eigenvectors(verbose=True)

    lp_rie_estimator = LedoitPecheRIEstimator()

    optimal_q = lp_rie_estimator.get_optimal_q(N, T)
    eta_scale = lp_rie_estimator.get_optimal_eta_scale(N)
    eta_factor_list = np.arange(-3., 1.05, 0.05)

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

    bouchaud_eta_scale = lp_rie_estimator.get_optimal_eta_scale(N)
    simulation_eta = bouchaud_eta_scale * 10 ** eta_factor_list[np.argmin(to_plot)]

    print("Bouchaud optimal η scale: " + str(bouchaud_eta_scale))
    print("Simulation optimal η factor: " + str(simulation_eta))
    plt.show()

    estimated_eigenvalues_sim_eta = lp_rie_estimator.estimate_eigenvalues(sample_eigenvalues,
                                                                          simulation_eta,
                                                                          optimal_q).mean(axis=0)

    true_C_eigenvalues = la.eigvals(true_C)

    eigenvales_numbers = np.array([i for i in range(true_C_eigenvalues.shape[0])])

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)

    axs[0].set_xlabel('η factor')
    axs[0].set_ylabel('F(LP RIE estimated C - true C)')
    axs[0].set_title('Ledoit-Peche RIE Frobenius norm for varying η factor ['+plot_type+']\n'
                     'min(F) = '+str(np.min(to_plot)))
    axs[0].scatter(eta_factor_list, to_plot, s=10)

    axs[1].scatter(eigenvales_numbers, true_C_eigenvalues, c='blue', label='Eigenvalues of true C', marker='.')
    axs[1].scatter(eigenvales_numbers, estimated_eigenvalues_sim_eta, c='green',
                   label='Ledoit-Peche RIE for simulation optimal η factor', marker='.')
    axs[1].set_xlabel('λk [kth eigenvalue]')
    axs[1].set_ylabel('value')
    axs[1].set_title('Ledoit-Peche RIE for simulation η factor and Bouchauds η scale ['+plot_type+']\n'
                      'η = ' + str(simulation_eta))
    axs[1].legend(loc='upper right')

    plt.show()


def test_resolvents(x_min, x_max):

    kappa = 0.3
    tau = 4.75
    eta = 0.005
    q = LedoitPecheRIEstimator().get_optimal_q(150, 200)

    x_plot_arr = np.arange(x_min, x_max, 0.01)
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)

    matrix = TimeSeriesMatrix()
    matrix.generate.multivariate_gaussian(np.eye(150), np.eye(200), 30)
    C_1_eigvals = matrix.characteristics.global_eigenvalues()
    matrix.generate.inverse_wishart(150, 200, kappa, 30, normalize_covariance=True)
    C_inverse_wishart_eigvals = matrix.characteristics.global_eigenvalues()
    matrix.generate.exponential_decay(150, 200, tau, 30)
    C_exp_decay_eigvals = matrix.characteristics.global_eigenvalues()
    matrix.generate.inverse_wishart(150, 200, kappa, 30, normalize_covariance=True)
    C_inverse_wishart = matrix.generate.last_C
    matrix.generate.exponential_decay(150, 200, tau, 30)
    A_exp_decay = matrix.generate.last_A
    matrix.generate.multivariate_gaussian(C_inverse_wishart, A_exp_decay, 30)
    mixed_eigvals = matrix.characteristics.global_eigenvalues()

    axs[0, 0].plot(x_plot_arr, C1A1WishartResolvent.compute(q, x_plot_arr, eta)[0], c='red',
                   label='theory (resolvent)')
    axs[0, 0].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_1_eigvals, x_plot_arr, eta)[0], marker='.',
                      label='simulation (resolvent)')
    axs[0, 0].set_title('C=1, A=1')
    axs[0, 0].set_ylabel('Re(resolvent)')
    axs[0, 0].legend(loc='lower right')

    axs[0, 1].plot(x_plot_arr, C1A1WishartResolvent.compute(q, x_plot_arr, eta)[1] / np.pi, c='red',
                   label='theory (resolvent)')
    axs[0, 1].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_1_eigvals, x_plot_arr, eta)[1] / np.pi, marker='.',
                      label='simulation (resolvent)')
    axs[0, 1].hist([item for sublist in C_1_eigvals for item in sublist], density=True, bins=100,
                   range=(x_min, x_max), histtype='step', color='green', label='simulation (histogram)')
    axs[0, 1].set_title('C=1, A=1')
    axs[0, 1].set_ylabel('density via Im(resolvent)')
    axs[0, 1].legend(loc='upper right')

    axs[1, 0].plot(x_plot_arr, InverseWishartResolvent.compute(q, kappa, x_plot_arr, eta)[0],
                   c='red', label='theory (resolvent)')
    axs[1, 0].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_inverse_wishart_eigvals, x_plot_arr, eta)[0], marker='.',
                      label='simulation (resolvent)')
    axs[1, 0].set_title('C=IW, A=1')
    axs[1, 0].set_ylabel('Re(resolvent)')
    axs[1, 0].legend(loc='lower right')

    axs[1, 1].plot(x_plot_arr,
                   InverseWishartResolvent.compute(q, kappa, x_plot_arr, eta)[1] / np.pi, c='red',
                   label='theory (resolvent)')
    axs[1, 1].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_inverse_wishart_eigvals, x_plot_arr, eta)[1] / np.pi, marker='.',
                      label='simulation (resolvent)')
    axs[1, 1].hist([item for sublist in C_inverse_wishart_eigvals for item in sublist], density=True, bins=100,
                   range=(x_min, x_max), histtype='step', color='green', label='simulation (histogram)')
    axs[1, 1].set_title('C=IW, A=1')
    axs[1, 1].set_ylabel('density via Im(resolvent)')
    axs[1, 1].legend(loc='upper right')

    plt.show()

    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)

    axs[0, 0].plot(x_plot_arr[:ExponentialDecayResolvent.compute(q, tau, x_plot_arr)[0].shape[0]],
                   ExponentialDecayResolvent.compute(q, tau, x_plot_arr)[0], c='red',
                   label='theory (resolvent)')
    axs[0, 0].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_exp_decay_eigvals, x_plot_arr, eta)[0], marker='.',
                      label='simulation (resolvent)')
    axs[0, 0].set_title('C=1, A=ED')
    axs[0, 0].set_ylabel('Re(resolvent)')
    axs[0, 0].legend(loc='lower right')

    axs[0, 1].plot(x_plot_arr[:ExponentialDecayResolvent.compute(q, tau, x_plot_arr)[1].shape[0]],
                   ExponentialDecayResolvent.compute(q, tau, x_plot_arr)[1] / np.pi,
                   c='red', label='theory (resolvent)')
    axs[0, 1].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(C_exp_decay_eigvals, x_plot_arr, eta)[1] / np.pi, marker='.',
                      label='simulation (resolvent)')
    axs[0, 1].hist([item for sublist in C_exp_decay_eigvals for item in sublist], density=True, bins=100,
                   range=(x_min, x_max), histtype='step', color='green', label='simulation (histogram)')
    axs[0, 1].set_title('C=1, A=ED')
    axs[0, 1].set_ylabel('density via Im(resolvent)')
    axs[0, 1].legend(loc='upper right')

    axs[1, 0].plot(x_plot_arr[:InvWishExpDecayMixedResolvent.compute(q, kappa, tau, x_plot_arr)[0].shape[0]],
                   InvWishExpDecayMixedResolvent.compute(q, kappa, tau, x_plot_arr)[0],
                   c='red', label='theory (resolvent)')
    axs[1, 0].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(mixed_eigvals, x_plot_arr, eta)[0], marker='.',
                      label='simulation (resolvent)')
    axs[1, 0].set_title('C=IW, A=ED')
    axs[1, 0].set_ylabel('Re(resolvent)')
    axs[1, 0].legend(loc='lower right')

    axs[1, 1].plot(x_plot_arr[:InvWishExpDecayMixedResolvent.compute(q, kappa, tau, x_plot_arr)[1].shape[0]],
                   InvWishExpDecayMixedResolvent.compute(q, kappa, tau, x_plot_arr)[
                       1] / np.pi, c='red', label='theory (resolvent)')
    axs[1, 1].scatter(x_plot_arr, SimulatedEigenvaluesResolvent.compute_array(mixed_eigvals, x_plot_arr, eta)[1] / np.pi, marker='.',
                      label='simulation (resolvent)')
    axs[1, 1].hist([item for sublist in mixed_eigvals for item in sublist], density=True, bins=100,
                   range=(x_min, x_max), histtype='step', color='green', label='simulation (histogram)')
    axs[1, 1].set_title('C=IW, A=ED')
    axs[1, 1].set_ylabel('density via Im(resolvent)')
    axs[1, 1].legend(loc='upper right')

    plt.show()


def test_how_bad_is_sample_estimator():

    series=TimeSeriesMatrix()
    series.generate.multivariate_gaussian(np.eye(200), np.eye(200), 10)
    C = series.generate.last_C
    eigvals_est = series.characteristics.global_eigenvalues()
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)

    x_plot_arr = np.arange(0.01, 4.0, 0.01)
    y_plot_arr = np.array([1/(2*np.pi)*np.sqrt((4-x)/x) for x in x_plot_arr])

    axs.set_title('Sample estimator eigenvalues distribution for matrix with all true covariance eigenvalues equal to 1')
    axs.plot(x_plot_arr, y_plot_arr, c='red', label='theory (Marchenko-Pastur)', linewidth=0.5)
    axs.hist([item for sublist in eigvals_est for item in sublist], density=True, stacked=True, bins=100,
             range=(0.0, 4.0), histtype='step', color='green', label='simulation (histogram)')
    axs.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    print("Started")

    # test_generating_inverse_wishart(5)
    # test_generating_exp_decay(5)
    # test_saving_and_reading_from_csv() <- works
    # test_saving_and_reading_from_csv() <- works
    # test_rolling_window() <- works
    # test_global_sampler() <- works
    # test_lse_optimal_alphas()
    # test_ledoit_peche_rie()
    # test_resolvents(0.0, 3.5)
    # test_how_bad_is_sample_estimator()
    # test_eigenvalues_clipping()
    matrix = TimeSeriesMatrix()
    matrix.normalize.standard()

