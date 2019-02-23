
from fake_data import generate_fake_data, get_params, get_small_params
from hmmc_tf import HMMC
import numpy as np

#Example of how to use a hidden markov model to infer paramaters from a mixture of gaussians.

def main():
    #Generates data from a mixture of gaussians and train a HMM to learn
    #the distribution of the hidden variables.
    np.random.seed(128)
    X = generate_fake_data(100, 100, get_params)
    model = HMMC(5, 3)
    model.fit(X, 10)
    likelihood = model.get_individual_log_likelihood(X[0])
    print("Log likelihood of X[0] with fitted model:", likelihood)

    #Now compare the results with a model with the true parameters.
    M, D, K, pi, A, R, mu, sigma = get_params()
    sigma2 = np.zeros((sigma.shape[0], sigma.shape[1], sigma.shape[2]))
    for m in range(M):
        for k in range(K):
            sigma2[m, k] = np.diag(sigma[m, k])

    model.set_params(M, D, K, pi, A, R, mu, sigma2)
    likelihood = model.get_individual_log_likelihood(X[0])
    print("Log likelihood of X[0] with real parameters:", likelihood)

if __name__ == "__main__":
    main();