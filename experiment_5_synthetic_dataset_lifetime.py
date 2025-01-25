import matplotlib.pylab as plt
import numpy as np
import scipy.spatial.transform.rotation as rotation

from config import *
from mondrian_kernel import evaluate_all_lifetimes
from utils import initialize_plotting, remove_chartjunk, tableau20

def construct_data(N_train, N_test):
    """ Construct the Mondrian line dataset with the given number of datapoints.
    """
    
    # sample data
    N = N_train + N_test
    epsilon = 0.01
    X = np.array([[np.random.random(), (2 * np.random.random() - 1) * epsilon, (2 * np.random.random() - 1) * epsilon] for _ in range(0, N)])
    
    # label the data
    y = np.array([1 if ((X[i][1] > 0 and X[i][2] > 0) or (X[i][1] < 0 and X[i][2] < 0)) else 0 for i in range(0, N)])
    
    # format the data
    rot = np.array([[1 / np.sqrt(3), -1 / np.sqrt(6), -1 / np.sqrt(2)], [1 / np.sqrt(3), 2 / np.sqrt(6), 0], [1 / np.sqrt(3), -1 / np.sqrt(6), 1 / np.sqrt(2)]])
    X = np.array([rot @ X[i] for i in range(0, N)])

    # pick training indices sequentially
    indices_train = range(0, N_train)
    indices_test  = range(N_train, N)

    # split the data into train and test
    X_train = X[indices_train]
    X_test  = X[indices_test ]
    y_train = y[indices_train]
    y_test  = y[indices_test ]

    return X_train, y_train, X_test, y_test

def experiment_5_synthetic_dataset_lifetime():
    """ Compares approximation error of Mondrian kernel and uniformly rotated Mondrian kernel
        on the non-adversarial, synthetic Mondrian line dataset versus the lifetime. 
    """

    # fix random seed
    np.random.seed(seed)

    # synthetize data
    N_train = 500
    N_test = 500
    X, y, X_test, y_test = construct_data(N_train, N_test)

    # Mondrian kernel lifetime sweep parameters
    M = 500
    lifetime_max = 4e1
    delta = 0 

    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True, uniformly_rotated=False, validation=False)
    lifetimes = res['times']
    error_train = res['kernel_train'] 
    error_test = res['kernel_test'] 
    error_train = [num / 100.0 for num  in error_train]
    error_test = [num / 100.0 for num  in error_test]

    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True, uniformly_rotated=True, validation=False)
    lifetimes_unif = res['times']
    error_train_unif = res['kernel_train'] 
    error_test_unif = res['kernel_test'] 
    error_train_unif = [num / 100.0 for num  in error_train_unif]
    error_test_unif = [num / 100.0 for num  in error_test_unif]

    # set up plot
    fig = plt.figure(num=2, figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)

    ax.set_title('$M = %d$, $\\mathcal{D}$ = synthetic ($D = 3$, $N_{train} =%d$, $N_{test}=%d$)' % (M, N_train, N_test))
    #ax.set_xscale('log')
    ax.yaxis.grid(which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('lifetime $\\lambda$')
    ax.set_ylabel('relative error [$\\%$]')

    ax.plot(lifetimes, error_train, drawstyle="steps-post", ls='-', color=tableau20(4), label='train Mondrian kernel')
    ax.plot(lifetimes, error_test, drawstyle="steps-post", ls='-', color=tableau20(5), label='test Mondrian kernel')
    ax.plot(lifetimes_unif, error_train_unif, drawstyle="steps-post", ls='-', color=tableau20(2), label='train uniformly rotated')
    ax.plot(lifetimes_unif, error_test_unif, drawstyle="steps-post", ls='-', color=tableau20(3), label='test uniformly rotated')
    ax.legend(frameon=False)
    
    # plot ground truth and estimate
    ax.set_xticks([1e0, 1e1, 2e1, 3e1, 4e1], labels=["$0$", "$10$", "$20$", "$30$", "$40$"])
    ax.set_xlim((1e0, 4e1))
    
def main():
    initialize_plotting()
    experiment_5()
    plt.show()

if __name__ == "__main__":
    main()