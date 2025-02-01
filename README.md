# RLS Polynomial Regression with Chebyshev Basis and QR Decomposition

This project provides an optimized implementation of Recursive Least Squares (RLS) for polynomial regression that leverages a Chebyshev polynomial basis for improved numerical stability. The algorithm employs an incremental QR decomposition update using Givens rotations for efficient real-time data processing, and it optionally applies a forgetting factor to discount older measurements. An enhanced condition number estimation routine ensures that the QR decomposition is fully reorthogonalized when the system becomes too ill-conditioned.

## Overview

Recursive Least Squares (RLS) is a powerful method for dynamically fitting models to data by continuously updating the model parameters as new data is received. In this project, RLS is extended to polynomial regression by fitting a polynomial curve—represented in a Chebyshev basis—to streaming data. This approach is particularly well-suited for applications where data is continuously updated and real-time responsiveness is essential.

## Mathematical Foundations and Implementation Choices

### Recursive Least Squares (RLS)

The RLS algorithm is designed to solve least squares problems efficiently by updating the model parameters with each new data point. Instead of recalculating the solution from scratch, the algorithm updates the existing solution incrementally, which is crucial for real-time applications. In this implementation, the algorithm is tailored for polynomial regression, allowing the fitting of polynomial curves to data that may evolve over time.

### Chebyshev Basis Functions

**Why Chebyshev?**  
Traditional polynomial regression often uses monomials (i.e., `1, x, x², …`), which can suffer from severe numerical instability as the degree increases. Chebyshev polynomials, on the other hand, are known for their superior numerical properties and lower condition numbers when operating in the interval `[-1, 1]`.

**Normalization:**  
Since Chebyshev polynomials are defined on `[-1,1]`, the independent variable `x` is first normalized as follows:

$$
r = 2 \cdot \frac{x - \min{x}}{\max{x} - \min{x}} - 1
$$


This normalization not only improves numerical stability but also ensures that the basis functions are computed in an optimal range.

**Recurrence Relation:**  
The Chebyshev basis is computed using the recurrence:

$$
T_0(x) = 1
$$

$$
T_1(x) = x
$$

$$
T_n(x) = 2 \cdot x \cdot T_{n-1}(x) - T_{n-2}(x), \quad \text{for } n \geq 2
$$


This approach is more efficient and less error-prone than using the power function for each term.

### QR Decomposition with Givens Rotations

**Incremental Updates:**  
To maintain numerical stability, the algorithm performs a QR decomposition of the design matrix. Givens rotations are used to incrementally update the QR factors as new data points are added or removed. This in-place update avoids recomputing the entire decomposition at every step, making the approach computationally efficient for streaming data.

**Hybrid Approach:**  
While the majority of updates use efficient Givens rotations, the implementation also monitors the condition number of the `R` matrix. If the condition number exceeds a specified threshold, a full reorthogonalization is triggered via a Householder-based QR recomputation with column pivoting. This hybrid approach combines speed (through incremental updates) with robustness (via periodic full recomputations).

### Forgetting Factor

**Optional Adaptivity:**  
An optional forgetting factor is integrated into the implementation. When enabled, the existing QR factors are scaled by `sqrt(lambda)` before a new data point is incorporated. This effectively discounts older measurements, allowing the model to adapt more rapidly to recent changes in the data.

**Mathematical Rationale:**  
A forgetting factor, typically denoted as `λ` with `0 < λ ≤ 1`, modifies the cost function so that older data points contribute less to the parameter update. This is particularly beneficial in non-stationary environments where the underlying process may change over time.

### Condition Number Estimation

**Enhanced Stability Monitoring:**  
The condition number of the upper-triangular matrix `R` is continuously monitored using an improved estimation routine based on the 1‑norm. The algorithm computes `‖R‖₁` and iteratively estimates `‖R⁻¹‖₁`, and their product provides an estimate of the condition number. If the condition number exceeds a defined threshold, the algorithm triggers a full recomputation of the QR decomposition, ensuring that the model remains numerically stable.

### Performance and Memory Optimizations

**In-Place Computation and Global Scratch Space:**  
To reduce memory overhead and improve cache performance, the implementation performs many operations in-place. Global scratch space is optionally used for temporary matrices during the reorthogonalization process, minimizing dynamic memory allocations and enhancing overall speed.

**Loop Merging and Minimizing Copies:**  
Many loops have been merged or streamlined to avoid redundant computations and excessive copying of data, which contributes to improved real-time performance.

### Gradient Computation

**Derivatives of the Fitted Function:**  
Once the regression model is updated, the first- and second-order derivatives of the fitted function are computed. Using the Chebyshev basis, the first derivative is given by:

$$
\frac{df}{dx} = f'(r) \cdot \left(\frac{2}{\max{x} - \min{x}}\right)
$$


where `f'(r)` is computed using the known derivatives of the Chebyshev polynomials:
- `T₁'(r) = 1`
- `T₂'(r) = 4r`
- `T₃'(r) = 12r² - 3`

The second derivative is computed in a similar manner using the second derivatives of the Chebyshev polynomials and applying the chain rule.

---

## Key Features

1. **Incremental QR Decomposition**  
   - *Efficient Updates:* Uses Givens rotations to update the QR factors in-place when new data points are added or removed.
   - *Hybrid Reorthogonalization:* Monitors the condition number to trigger a full recomputation when needed.

2. **Chebyshev Polynomial Basis**  
   - *Improved Numerical Stability:* Chebyshev polynomials, combined with proper normalization, reduce the ill-conditioning typical of high-degree polynomial regression.
   - *Efficient Computation:* The recurrence relation minimizes overhead compared to traditional power computations.

3. **Optional Forgetting Factor**  
   - *Adaptive Weighting:* Applies a forgetting factor to discount older data, allowing the regression model to adapt rapidly to changes.
   - *Mathematically Sound:* Scales QR factors by `sqrt(lambda)` to incorporate exponential forgetting directly into the update process.

4. **Enhanced Condition Number Estimation**  
   - *Robust Stability Checks:* Uses an improved 1-norm estimation method to monitor the condition number and trigger reorthogonalization when necessary.
   - *Maintains Accuracy:* Prevents the model from becoming unstable due to numerical errors over long data streams.

5. **Optimized Performance and Memory Usage**  
   - *In-Place Operations:* Minimizes memory copying and maximizes cache efficiency.
   - *Global Scratch Space:* Optionally uses global scratch space to reduce stack allocation overhead during reorthogonalization.

6. **Robust Gradient Computation**  
   - *First- and Second-Order Derivatives:* Computes both slope and curvature of the fitted function using the chain rule applied to the Chebyshev basis.
   - *Direct Use in Adaptive Systems:* Provides essential information for real-time control or further analysis of the fitted model.

---

This documentation describes the design and implementation of the Chebyshev-based RLS regression algorithm with QR decomposition and an optional forgetting factor. It highlights the key mathematical and computational decisions made to achieve high numerical stability, efficient incremental updates, and robust real-time data processing.
