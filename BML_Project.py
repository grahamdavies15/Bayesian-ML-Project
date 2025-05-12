# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import hmc_Lab as hmc

# Read Data
train = pd.read_csv('ee-train.csv')
test = pd.read_csv('ee-test.csv')

train = pd.DataFrame(train)

print(f'Train data: \n',train.head(5))
#print(f'Test data \n',test.head(2))

# Standardise and view values:
from sklearn.preprocessing import StandardScaler

X_train_raw = train.iloc[:, 1:9].values  # inputs (x_1 to x_8)
y_train = train.iloc[:, 9].values

X_test_raw = test.iloc[:, 1:9].values  # inputs (x_1 to x_8)
y_test = test.iloc[:, 9].values

columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

# create the scaler
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train_raw)
scaled_test = scaler.transform(X_test_raw)

# combine
X_train = np.column_stack((train.iloc[:, 0].values, scaled_train))
X_test = np.column_stack((test.iloc[:, 0].values, scaled_test))


columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
df_train =  pd.DataFrame(np.column_stack((X_train, y_train)), columns = columns)
df_test =  pd.DataFrame(np.column_stack((X_test, y_test)), columns = columns)

print(f'Train data: \n',df_train.head(5))
print(f'Summary Statistics: \n', df_train.describe())
print(f'Empty values: \n', df_train.isnull().sum())

print(y_test.shape)


# Remove x_0
df_without_x0 = df_train.drop('x0', axis=1)
corr = df_without_x0.corr()

# plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title("Correlation Heatmap of 'Energy Efficiency' Variables")
plt.show()

# subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i in range(X_train.shape[1]):
    # Scatter plot of x_i vs Heating Load
    axes[i].scatter(X_train[:, i], y_train, alpha=0.5)
    axes[i].set_title(f'x_{i} vs Heating Load')
    axes[i].set_xlabel(f'x_{i}')
    axes[i].set_ylabel('Heating Load')

plt.tight_layout()
plt.show()







# Least-squares linear model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

model_train_41 = LinearRegression()
model_train_41.fit(X_train, y_train)

y_train_pred_41 = model_train_41.predict(X_train)
y_test_pred_41 = model_train_41.predict(X_test)

mae_train_41 = mean_absolute_error(y_train, y_train_pred_41)
mae_test_41 = mean_absolute_error(y_test, y_test_pred_41)

print(f'MAE Train: {mae_train_41:.4f}')
print(f'MAE Test: {mae_test_41:.4f}')

rmse_train_41 = mean_squared_error(y_train, y_train_pred_41, squared=False)
rmse_test_41 = mean_squared_error(y_test, y_test_pred_41, squared=False)

print(f'RMSE Train: {rmse_train_41:.4f}')
print(f'RMSE Test: {rmse_test_41:.4f}')

# training
plt.figure(figsize=(10, 8))
plt.scatter(y_train, y_train_pred_41, label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train_pred_41), max(y_train_pred_41)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Training Data)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_41, label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test_pred_41), max(y_test_pred_41)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Test Data)')
plt.legend()
plt.show()

print(f' X_train shape: ', X_train.shape)
print(f' y_train shape: ', y_train.shape)

print(f' X_test shape: ', X_test.shape)
print(f' y_test shape: ', y_test.shape)












# Type-2 maximum likelihood
# Compute log-likelihood
def compute_log_marginal(X, y, alph, beta):
    N, M = X.shape
    PHI = X
    C = (1 / beta) * np.eye(N) + PHI @ PHI.T / alph

    lgp = -N / 2 * np.log(2 * np.pi)
    _, log_det = np.linalg.slogdet(C)
    lgp -= log_det / 2
    lgp -= y.T @ np.linalg.inv(C) @ y / 2

    return lgp

# Create a grid of log_alpha and log_beta values
log_alpha_values = np.linspace(-5, 0, 100)
log_beta_values = np.linspace(-5, 0, 100)

log_prob_y = np.zeros((len(log_alpha_values), len(log_beta_values)))

for i, log_alpha in enumerate(log_alpha_values):
    alpha = np.exp(log_alpha)
    for j, log_beta in enumerate(log_beta_values):
        beta = np.exp(log_beta)
        log_prob_y[i, j] = compute_log_marginal(X_train, y_train, alpha, beta)

max_indices = np.unravel_index(np.argmax(log_prob_y), log_prob_y.shape)
max_log_alpha = log_alpha_values[max_indices[0]]
max_log_beta = log_beta_values[max_indices[1]]

likelihood = compute_log_marginal(X_train, y_train, np.exp(max_log_alpha), np.exp(max_log_beta))

print(f'Most probable alpha: {np.exp(max_log_alpha):.4f}')
print(f'Most probable beta: {np.exp(max_log_beta):.4f}')
print(f'Max log alpha: {max_log_alpha:.4f}')
print(f'Max log beta: {max_log_beta:.4f}')
print(f'Likelihood: {likelihood:.4f}')

plt.figure(figsize=(10, 8))
plt.contourf(log_alpha_values, log_beta_values, log_prob_y.T, levels=60, cmap='viridis')
plt.plot(max_log_alpha, max_log_beta, 'ro', markersize=10, label='Maximum')
plt.colorbar(label='Log-Evidence')
plt.xlabel('Log Alpha')
plt.ylabel('Log Beta')
plt.title('Log-Posterior Distribution')

label_text = f"({max_log_alpha:.2f}, {max_log_beta:.2f})"
plt.annotate(label_text, xy=(max_log_alpha, max_log_beta), xytext=(10, 10), textcoords="offset points", fontsize=12)

plt.legend()
plt.show()


def compute_posterior(X, y, alph, beta):
    PHI = X

    inverse = np.linalg.inv(PHI.T @ PHI + (1 / beta) * alph * np.eye(PHI.shape[1]))
    Mu = inverse @ PHI.T @ y

    SIGMA = (1 / beta) * inverse

    return Mu, SIGMA

# Compute the posterior distribution
Mu, SIGMA = compute_posterior(X_train, y_train, max_log_alpha, max_log_beta)

# Compute the predicted values for the training set
y_train_pred_42 = X_train @ Mu
y_test_pred_42 = X_test @ Mu

mae_train_42 = mean_absolute_error(y_train, y_train_pred_42)
mae_test_42 = mean_absolute_error(y_test, y_test_pred_42)

print(f'MAE Train: {mae_train_42:.4f}')
print(f'MAE Test: {mae_test_42:.4f}')


# RMSE
from sklearn.metrics import mean_squared_error

rmse_train_42 = mean_squared_error(y_train, y_train_pred_42, squared=False)
rmse_test_42 = mean_squared_error(y_test, y_test_pred_42, squared=False)

print(f'RMSE Train: {rmse_train_42:.4f}')
print(f'RMSE Test: {rmse_test_42:.4f}')

# training
plt.figure(figsize=(10,8))
plt.scatter(y_train, y_train_pred_42, label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train_pred_42), max(y_train_pred_42)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Training Data)')
plt.legend()
plt.show()

# test
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_42, label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test_pred_42), max(y_test_pred_42)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Test Data)')
plt.legend()
plt.show()










# HMC to linear regression model
# energy functions
def energy_func_lr(hps, x, y):
    alpha = hps[0]  # weight precision (1 / weight variance) (input as logged)
    beta = hps[1]  # noise precision (1 / noise variance) (input as logged)
    w = hps[2:]  # Weights

    N = x.shape[0]  # Number of training examples
    M = w.shape[0]  # Number of weights

    f_x = x @ w  # Predicted values

    log_likelihood = (N / 2) * (beta - np.log(2 * np.pi)) - (np.exp(beta) / 2) * np.sum((y - f_x) ** 2)
    log_prior = (M / 2) * (alpha - np.log(2 * np.pi)) - (np.exp(alpha) / 2) * np.sum(w ** 2)

    # get the negative log likelihood
    neglgp = -(log_likelihood + log_prior)

    return neglgp


def energy_grad_lr(hps, x, y):
    alpha = hps[0]  # weight precision
    beta = hps[1]  # noise precision
    w = hps[2:]  # Weights

    N = x.shape[0]  # Number of training examples
    M = w.shape[0]  # Number of weights

    f_x = x @ w

    # gradient alpha
    d_alpha = M / 2 - (np.exp(alpha) / 2) * np.sum(w ** 2)

    # gradient beta
    d_beta = N / 2 - (np.exp(beta) / 2) * np.sum((y - f_x) ** 2)

    # gradient weights
    d_w = np.exp(beta) * x.T @ (y - f_x) - np.exp(alpha) * w

    # Combine the gradients into a single array
    g = np.concatenate(([-d_alpha], [-d_beta], -d_w))

    return g

np.random.seed(seed=1)

# Set the HMC parameters
R = 10000
L = 20
eps = 0.0519

burn = int(R/10)

Y_train = y_train

# Initialize the hyperparameters

alpha0 = -5
beta0 = -3
w0 = np.zeros(X_train.shape[1])
x0 = np.concatenate(([alpha0], [beta0], w0))


S, *_ = hmc.sample(x0, energy_func_lr, energy_grad_lr, R, L, eps, burn=burn, checkgrad=True, args=[X_train, Y_train])

# get samples
alpha_samples = S[:, 0]
beta_samples = S[:, 1]
weight_samples = S[:, 2:]

fig, axs = plt.subplots(4, 3, figsize=(12, 16))

# Plot alpha trace
axs[0, 0].plot(alpha_samples)
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Log Alpha')
axs[0, 0].set_title('Alpha')

# Plot beta trace
axs[0, 1].plot(beta_samples)
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Log Beta')
axs[0, 1].set_title('Beta')

# Plot weight traces
for i in range(weight_samples.shape[1]):
    row = (i + 2) // 3
    col = (i + 2) % 3
    axs[row, col].plot(weight_samples[:, i])
    axs[row, col].set_xlabel('Iteration')
    axs[row, col].set_ylabel(f'Weight {i+1}')
    axs[row, col].set_title(f'Weight {i+1}')

# remove others
for i in range(weight_samples.shape[1] + 2, 12):
    row = i // 3
    col = i % 3
    fig.delaxes(axs[row, col])

plt.tight_layout()
plt.show()

avg_4 = np.mean(S, axis=0)

#print(avg_4)
#print(X_train.shape)

# get predictions
y_train_pred_44 = X_train @ avg_4[2:]
y_test_pred_44 = X_test @ avg_4[2:]

mae_train_44 = mean_absolute_error(y_train, y_train_pred_44)
mae_test_44 = mean_absolute_error(y_test, y_test_pred_44)

print(f'MAE Train: {mae_train_44:.4f}')
print(f'MAE Test: {mae_test_44:.4f}')

rmse_train_44 = mean_squared_error(y_train, y_train_pred_44, squared=False)
rmse_test_44 = mean_squared_error(y_test, y_test_pred_44, squared=False)

print(f'RMSE Train: {rmse_train_44:.4f}')
print(f'RMSE Test: {rmse_test_44:.4f}')

likelihood = compute_log_marginal(X_train, y_train, np.exp(avg_4[0]), np.exp(avg_4[1]))

print(f'Most probable alpha: {np.exp(avg_4[0]):.4f}')
print(f'Most probable beta: {np.exp(avg_4[1]):.4f}')
print(f'Max log alpha: {avg_4[0]:.4f}')
print(f'Max log beta: {avg_4[1]:.4f}')
print(f'Likelihood: {likelihood:.4f}')

plt.figure(figsize=(10,8))
plt.scatter(alpha_samples, beta_samples, label='Train')
plt.xlabel('Log Alpha')
plt.ylabel('Log Beta')
plt.title('Alpha vs Beta samples.')
plt.legend()
plt.show()

# training
plt.figure(figsize=(10,8))
plt.scatter(y_train, y_train_pred_44, label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train_pred_42), max(y_train_pred_42)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Training Data)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_44, label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test_pred_42), max(y_test_pred_42)], 'k--', label='Ideal')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title('Predicted vs. Actual Heating Load (Test Data)')
plt.legend()
plt.show()









# Apply HMC as a classifier
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def energy_func_logi(hps, x, y):
    alpha = hps[0]  # weight precision (1 / weight variance) (input as logged)
    w = hps[1:]  # Weights

    f_x = sigmoid(x @ w)  # Predicted probabilities using sigmoid link function

    log_likelihood = np.sum(y * np.log(f_x) + (1 - y) * np.log(1 - f_x))

    M = w.shape[0]  # Number of weights
    log_prior = (M / 2) * (alpha - np.log(2 * np.pi)) - (np.exp(alpha) / 2) * np.sum(w ** 2)

    # Sum and get the negative log likelihood
    neglgp = -(log_likelihood + log_prior)

    return neglgp


def energy_grad_logi(hps, x, y):
    alpha = hps[0]  # weight precision (1 / weight variance) (input as logged)
    w = hps[1:]  # Weights

    M = w.shape[0]  # Number of weights
    f_x = sigmoid(x @ w)  # Predicted probabilities using sigmoid link function

    # gradient alpha
    d_alpha = M / 2 - (np.exp(alpha) / 2) * np.sum(w ** 2)

    # gradient weights
    d_w = x.T @ (y - f_x) - np.exp(alpha) * w

    # Combine the gradients into a single array
    g = np.concatenate(([-d_alpha], -d_w))

    return g

np.random.seed(seed=1)

# Convert target variable to binary labels
y_train_binary = (y_train > 23.0).astype(int)
y_test_binary = (y_test > 23.0).astype(int)

# initialise HMC parameters
R = 10000  # Number of samples
L = 20    # Number of leapfrog steps
eps = 0.025
burn = int(R /10)  # Burn-in period

# starting points
alpha0 = -4
w0 = np.zeros(X_train.shape[1])
x0 = np.concatenate(([alpha0], w0))

# Run HMC sampling
S, *_ = hmc.sample(x0, energy_func_logi, energy_grad_logi, R, L, eps, burn=burn, checkgrad=True, args=[X_train, y_train_binary])


from sklearn.metrics import accuracy_score

avg_5 = np.mean(S, axis=0)

print(avg_5)

# get predictions
y_train_pred_45 = sigmoid(X_train @ avg_5[1:])
y_test_pred_45 = sigmoid(X_test @ avg_5[1:])

# convert probabilities to binary predictions
y_train_pred_45_binary = (y_train_pred_45 > 0.5).astype(int)
y_test_pred_45_binary = (y_test_pred_45 > 0.5).astype(int)

# calculate misclassification rate
misclassification_rate_train = np.sum(y_train_binary != y_train_pred_45_binary) / len(y_train_binary)
misclassification_rate_test = np.sum(y_test_binary != y_test_pred_45_binary) / len(y_test_binary)

print(f'Misclassification Rate Train: {misclassification_rate_train:.4f}')
print(f'Misclassification Rate Test: {misclassification_rate_test:.4f}')

# get samples
alpha_samples = S[:, 0]
weight_samples = S[:, 1:]

fig, axs = plt.subplots(4, 3, figsize=(12, 16))

# Plot alpha trace
axs[0, 0].plot(alpha_samples)
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Alpha')
axs[0, 0].set_title('Trace Plot - Alpha')

# Plot weight traces
for i in range(weight_samples.shape[1]):
    row = (i + 1) // 3
    col = (i + 1) % 3
    axs[row, col].plot(weight_samples[:, i])
    axs[row, col].set_xlabel('Iteration')
    axs[row, col].set_ylabel(f'Weight {i+1}')
    axs[row, col].set_title(f'Trace Plot - Weight {i+1}')

# remove others
for i in range(weight_samples.shape[1] + 1, 12):
    row = i // 3
    col = i % 3
    fig.delaxes(axs[row, col])

plt.tight_layout()
plt.show()