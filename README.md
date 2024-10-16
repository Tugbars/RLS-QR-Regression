# RLS Polynomial Regression with QR Decomposition

This project provides an optimized implementation of Recursive Least Squares (RLS) for polynomial regression, utilizing QR decomposition with Givens rotations for efficient real-time data processing. The implementation is designed to handle both adding and removing data points dynamically while maintaining numerical stability, making it highly suitable for applications where continuous data streams need to be analyzed in real-time.

## Overview

Recursive Least Squares (RLS) is a powerful method for fitting models to data by continuously updating the model parameters as new data points are added. In this project, we extend the basic RLS algorithm to handle polynomial regression by fitting polynomial curves to data points. Furthermore, the integration of QR decomposition with Givens rotations ensures that the algorithm remains efficient and numerically stable, even as the dataset grows or shrinks.

## Mathematical Foundations and Implementation Choices

### Recursive Least Squares (RLS)

The RLS algorithm is widely used for solving least squares problems in a way that allows the model to be updated efficiently as new data is received. It is especially useful in real-time applications where recalculating the model from scratch every time new data is added would be computationally prohibitive.

In this implementation, we apply RLS to polynomial regression, meaning that we are fitting a polynomial function to the data rather than a simple linear function. The degree of the polynomial is adjustable, and the algorithm is designed to fit models ranging from simple linear relationships to higher-order curves.

### QR Decomposition with Givens Rotations

To ensure the numerical stability of the RLS process, we use QR decomposition. QR decomposition breaks a matrix into an orthogonal matrix (Q) and an upper triangular matrix (R), which allows us to solve least squares problems more accurately than direct methods like matrix inversion.

In this implementation, we chose to use Givens rotations for updating the QR decomposition incrementally. Givens rotations are particularly well-suited to this task because they allow the decomposition to be updated efficiently when data points are added or removed, without needing to recalculate the entire decomposition from scratch. This makes the algorithm both faster and more stable in real-time applications where data is constantly being updated.

### Polynomial Basis Optimization

Another key aspect of the implementation is the optimization of polynomial basis computation. Instead of relying on computationally expensive power functions (e.g., `pow(x, n)`), the basis functions are calculated using incremental multiplication. This not only speeds up the computation but also reduces rounding errors, contributing to better numerical stability, especially when dealing with high-order polynomials.

### Condition Number Estimation

To ensure the numerical stability of the QR decomposition, we monitor the condition number of the upper triangular matrix `R`. The condition number is a measure of how sensitive the matrix is to numerical errors. If the condition number exceeds a certain threshold, it indicates that the system is becoming unstable, and a full recomputation of the QR decomposition is triggered.

By periodically recalculating the QR decomposition when necessary, we maintain the stability and accuracy of the regression model over time.

## Key Features

1. **Incremental QR Decomposition**  
   The implementation uses Givens rotations to update the QR decomposition incrementally. This means that when a new data point is added or an old one is removed, the decomposition is adjusted without the need to recompute it from scratch. This significantly reduces computational overhead, particularly in applications where data is continuously streamed or updated.

2. **Optimized Polynomial Regression**  
   The code is designed to handle polynomial regression of various degrees. The polynomial terms are computed efficiently using incremental multiplication, which reduces computational costs and improves accuracy compared to traditional power-based methods.

3. **Numerical Stability Monitoring**  
   To ensure that the regression remains accurate over time, the condition number of the `R` matrix is continuously monitored. When the condition number exceeds a predefined threshold, the system triggers a reorthogonalization of the QR decomposition to maintain numerical stability.

4. **Real-Time Data Processing**  
   Thanks to the incremental QR updates and efficient polynomial basis computation, this implementation is highly suitable for real-time data processing. It can handle streams of data and continuously update the model with minimal latency.

5. **Adjustable Polynomial Degree**  
   The implementation supports polynomial regression of various degrees. By adjusting the degree parameter, users can fit anything from linear models to more complex curves, depending on the nature of their data.

6. **Regularization**  
   A regularization parameter is included in the implementation to prevent overfitting. This ensures that the model generalizes well to unseen data, especially in situations where the dataset is small or noisy.
