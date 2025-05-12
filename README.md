# Bayesian Linear Models with HMC

This project explores and compares four predictive modelling techniques within the context of Bayesian machine learning:

- **Least Squares Linear Regression**
- **Bayesian Linear Regression (Type-II Maximum Likelihood)**
- **Bayesian Linear Regression using Hamiltonian Monte Carlo (HMC)**
- **Bayesian Logistic Regression using HMC (Classification Task)**

## Dataset

The models are applied to the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency), which contains 768 examples of architectural building parameters with the goal of predicting **Heating Load**.

## Techniques

1. **Bayesian Linear Regression**
   - Implements Type-II Maximum Likelihood
   - Optimises the evidence with respect to hyperparameters α and β

2. **Hamiltonian Monte Carlo**
   - HMC is first verified on a standard 2D Gaussian
   - Applied to both regression and classification tasks using energy and gradient-based sampling

3. **Baseline Linear Model**
   - Standard least-squares regression for performance comparison

## Results Summary

| Method                          | MAE (Test) | RMSE (Test) |
|---------------------------------|------------|-------------|
| Least Squares                   | 2.0922     | 2.8539      |
| Bayesian Linear (Type-II)       | 2.0628     | 2.8573      |
| Bayesian Linear (HMC)           | 2.0683     | 2.8433      |
| HMC Classifier (Binary Target)  | 0.0078 (Misclassification Rate) |

Contour plot demonstrating the optimal α and β found.

![image](https://github.com/grahamdavies15/Bayesian-ML-Project/blob/main/images/2_contour.png)

Example of the HMC samples.

![image](https://github.com/grahamdavies15/Bayesian-ML-Project/blob/main/images/3_act_100.png)


## Key Takeaways

- All methods achieve similar predictive accuracy.
- **HMC-based classification** performs extremely well due to the nature of the binary split and the strength of predictive features.

## Requirements

- Python 3.9+
- Libraries: `numpy`, `scipy`, `matplotlib`, `pandas`

## References

- Coursework and lecture material by M. Tipping, University of Bath (2024)
- UCI Machine Learning Repository
