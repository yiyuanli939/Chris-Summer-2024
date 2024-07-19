import numpy as np
print("NumPy version:", np.__version__)

# Test a basic NumPy operation
arr = np.array([1, 2, 3, 4, 5])
print("NumPy array mean:", arr.mean())

import scipy as sp
print("SciPy version:", sp.__version__)

# Test a basic SciPy operation
from scipy import stats
print("Normal distribution CDF at 0:", stats.norm.cdf(0))

import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Data generation function
def generate_data(n_clusters, n_per_cluster, beta, sigma, tau):
    cluster_id = np.repeat(np.arange(1, n_clusters + 1), n_per_cluster)
    
    X = np.random.normal(0, 1, (n_clusters * n_per_cluster, len(beta) - 1))
    
    u = np.random.normal(0, sigma, n_clusters)
    u_repeated = np.repeat(u, n_per_cluster)
    
    epsilon = np.random.normal(0, 1, n_clusters * n_per_cluster)
    
    star_y = beta[0] + np.dot(X, beta[1:]) + u_repeated + epsilon
    
    y_list = [np.where(star_y > t, 1, 0) for t in tau]
    
    data = {
        'cluster': cluster_id,
        'X': X,
        'true_u': u_repeated,
        'star_y': star_y
    }
    
    for i, y in enumerate(y_list):
        data[f'y_{i+1}'] = y
    
    for i, t in enumerate(tau):
        data[f'true_tau_{i+1}'] = np.full(n_clusters * n_per_cluster, t)
    
    return data

# Define the g function
def g(v_i, X_i, beta, tau, d_i, sigma_v):
    A = len(d_i)
    normal_density = np.exp(-v_i**2 / (2 * sigma_v**2)) / np.sqrt(2 * np.pi * sigma_v**2)
    X_i_beta = np.dot(X_i, beta)
    product_term = 1
    for a in range(A):
        phi_term = d_i[a] * stats.norm.cdf(-tau[a] + X_i_beta + v_i) + \
                   (1 - d_i[a]) * stats.norm.cdf(tau[a] - X_i_beta - v_i)
        product_term *= phi_term
    return normal_density * product_term

# Define the compute_log_prob function
def compute_log_prob(X_i, d_i, beta, tau, sigma_v, rule):
    def integrand(v_i):
        return g(v_i, X_i, beta, tau, d_i, sigma_v)
    
    # Use numpy's quadrature function instead of aghQuad
    prob, _ = np.polynomial.hermite.hermgauss(len(rule['x']))
    prob = np.sum(prob * integrand(rule['x']))
    return np.log(prob)

# Define the log_likelihood function
def log_likelihood(params, X, y, rule):
    n_beta = X.shape[1]
    n_tau = y.shape[1]
    
    beta = params[:n_beta]
    tau = params[n_beta:n_beta+n_tau]
    sigma_v = np.exp(params[n_beta+n_tau])
    
    log_probs = np.array([compute_log_prob(X[i,:], y[i,:], beta, tau, sigma_v, rule) for i in range(X.shape[0])])
    return -np.sum(log_probs)

# Generate data
np.random.seed(42)
n_clusters = 500
n_per_cluster = 1
beta_true = np.array([3, -4, 5, -6, 2, 5])
sigma_true = 1
tau_true = np.array([-3, 3, 4])

data = generate_data(n_clusters, n_per_cluster, beta_true, sigma_true, tau_true)

# Prepare data for optimization
X = np.column_stack((np.ones(n_clusters * n_per_cluster), data['X']))
y = np.column_stack([data[f'y_{i+1}'] for i in range(len(tau_true))])

# Set up Gauss-Hermite quadrature rule
rule = {'x': np.polynomial.hermite.hermgauss(200)[0]}

# Initial parameter values
initial_beta = np.zeros(X.shape[1])
initial_tau = np.ones(y.shape[1])
initial_sigma = 1
initial_params = np.concatenate([initial_beta, initial_tau, [np.log(initial_sigma)]])

# Perform optimization with Nelder-Mead
result_nm = minimize(log_likelihood, initial_params, args=(X, y, rule), 
                     method='Nelder-Mead', options={'maxiter': 1000})

# Perform optimization with BFGS
result_bfgs = minimize(log_likelihood, initial_params, args=(X, y, rule), 
                       method='BFGS', options={'maxiter': 1000})

# Print the results
print("True parameters:")
print("Beta:", beta_true)
print("Tau:", tau_true)
print("Sigma:", sigma_true)

print("\nOptimized parameters (Nelder-Mead):")
print("Beta:", result_nm.x[:X.shape[1]])
print("Tau:", result_nm.x[X.shape[1]:X.shape[1]+y.shape[1]])
print("Sigma:", np.exp(result_nm.x[-1]))
print("Convergence:", result_nm.success)
print("Function evaluations:", result_nm.nfev)

print("\nOptimized parameters (BFGS):")
print("Beta:", result_bfgs.x[:X.shape[1]])
print("Tau:", result_bfgs.x[X.shape[1]:X.shape[1]+y.shape[1]])
print("Sigma:", np.exp(result_bfgs.x[-1]))
print("Convergence:", result_bfgs.success)
print("Function evaluations:", result_bfgs.nfev)
