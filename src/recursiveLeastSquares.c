/**
 * @file main.c
 * @brief Chebyshev-Based Recursive Least Squares (RLS) Regression with Optional Forgetting Factor
 *
 * This implementation uses a Chebyshev polynomial basis to represent the regression model.
 * The algorithm employs an incremental QR decomposition update (using Givens rotations)
 * to efficiently update the solution as new data arrives, and optionally applies a forgetting
 * factor to discount older measurements. An improved condition number estimation routine is
 * included (using 1-norm estimates) to trigger a full recomputation (reorthogonalization) of the
 * QR decomposition if the matrix becomes too ill-conditioned.
 *
 * The Chebyshev basis is computed using the recurrence:
 *   T0(x) = 1,
 *   T1(x) = x,
 *   Tn(x) = 2 * x * Tn-1(x) - Tn-2(x) for n ≥ 2.
 *
 * Since the regression is performed on a normalized variable r ∈ [−1, 1], the original
 * variable x is normalized as:
 *
 *   r = 2*(x - min_x)/(max_x - min_x) - 1.
 *
 * The fitted function (for a polynomial of degree 3, for example) is:
 *
 *   f(r) = a0*T0(r) + a1*T1(r) + a2*T2(r) + a3*T3(r).
 *
 * Its derivative with respect to r is:
 *
 *   f'(r) = a1*T1'(r) + a2*T2'(r) + a3*T3'(r),
 *
 * where for degree 3:
 *   T1'(r) = 1,
 *   T2'(r) = 4r,
 *   T3'(r) = 12r² - 3.
 *
 * Finally, by the chain rule:
 *
 *   df/dx = f'(r) * (dr/dx),  where dr/dx = 2/(max_x - min_x).
 *
 * A forgetting factor (if enabled) is applied to the accumulated QR factors (R and Qᵀ*b) by
 * scaling them with sqrt(lambda) before incorporating the new data, thus gradually discounting
 * older measurements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdarg.h>
#include "recursiveLeastSquares.h"


/* ********************************************************************
 * Configuration macros
 * ********************************************************************/

/** Regularization parameter (used to improve conditioning, similar to Ridge regression) */
#define REGULARIZATION_PARAMETER 1e-4

/** Threshold for the condition number above which the QR decomposition is recomputed */
#define CONDITION_NUMBER_THRESHOLD 1e4

/** Tolerance value for near-zero comparisons in the algorithm */
#define TOL (DBL_EPSILON * 3)

/** Uncomment or define USE_FORGETTING_FACTOR to enable the forgetting factor update */
#define USE_FORGETTING_FACTOR

#ifdef USE_FORGETTING_FACTOR
  /**
   * @brief Forgetting factor (lambda) used to discount older data.
   *
   * The forgetting factor should be in (0,1]. A value < 1 indicates that older data is
   * progressively discounted. In this implementation, the QR factors are scaled by sqrt(lambda)
   * before adding new data.
   */
  #define FORGETTING_FACTOR 0.55
#endif

/** Use global scratch space for temporary matrices to avoid repeated allocation on the stack */
#define USE_GLOBAL_SCRATCH_SPACE
#ifdef USE_GLOBAL_SCRATCH_SPACE
  /** Maximum total rows for the augmented matrix: data points + regularization rows */
  #define MAX_TOTAL_ROWS (RLS_WINDOW + MAX_POLYNOMIAL_DEGREE + 1)
  static double augmented_matrix_A[MAX_TOTAL_ROWS][MAX_POLYNOMIAL_DEGREE + 1];
  static double augmented_vector_b[MAX_TOTAL_ROWS];
#endif

/* ********************************************************************
 * Function definitions
 * ********************************************************************/

/**
 * @brief Initializes the regression state structure.
 *
 * This function resets all counters and zeroes the QR factors and coefficient vector.
 *
 * @param state Pointer to the RegressionState to be initialized.
 * @param degree Degree of the Chebyshev polynomial basis.
 * @param max_num_points Maximum number of data points (sliding window size).
 */
void initialize_regression_state(RegressionState *state, int degree, unsigned short max_num_points) {
    state->polynomial_degree = degree;
    state->current_num_points = 0;
    state->max_num_points = max_num_points;
    state->oldest_data_index = 0;
    state->total_data_points_added = 0;
    memset(state->coefficients, 0, sizeof(state->coefficients));
    memset(state->upper_triangular_R, 0, sizeof(state->upper_triangular_R));
    memset(state->Q_transpose_b, 0, sizeof(state->Q_transpose_b));
    for (int i = 0; i <= degree; ++i) {
        state->col_permutations[i] = i;
    }
}

/**
 * @brief Computes the Chebyshev basis functions for a normalized value.
 *
 * Given a normalized value (ranging in [-1, 1]), this function computes the Chebyshev
 * polynomials T0, T1, ... T_degree using the recurrence relation:
 *   T0(x) = 1,
 *   T1(x) = x,
 *   Tn(x) = 2*x*Tn-1(x) - Tn-2(x) for n ≥ 2.
 *
 * @param normalized_x The normalized input value.
 * @param degree The highest polynomial degree to compute.
 * @param basis Pointer to an array to store the computed basis values.
 */
static void compute_chebyshev_basis(double normalized_x, int degree, double *basis) {
    basis[0] = 1.0;
    if (degree >= 1) {
        basis[1] = normalized_x;
    }
    for (int i = 2; i <= degree; i++) {
        // Recurrence relation: T[i] = 2 * normalized_x * T[i-1] - T[i-2]
        basis[i] = 2.0 * normalized_x * basis[i - 1] - basis[i - 2];
    }
}

/**
 * @brief Incrementally updates the QR decomposition when adding a new row.
 *
 * This function uses Givens rotations to update the current upper-triangular matrix R
 * and the Qᵀ*b vector to incorporate a new data point (represented as a row in the design matrix).
 *
 * @param state Pointer to the current RegressionState.
 * @param new_row The new row (Chebyshev basis vector) to be added.
 * @param new_b The new measurement corresponding to the new row.
 */
static void updateQR_AddRow(RegressionState *state, const double *new_row, double new_b) {
    int num_coeff = state->polynomial_degree + 1;
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    // Copy the new row to a temporary array (for in-place rotations)
    memcpy(r_row, new_row, sizeof(double) * num_coeff);
    double Q_tb_new = new_b;
    // Apply Givens rotations to incorporate the new row into R
    for (int i = 0; i < num_coeff; ++i) {
        double a = state->upper_triangular_R[i][i];
        double b = r_row[i];
        if (fabs(b) > TOL) {
            // Compute rotation parameters: r = sqrt(a^2 + b^2), c = a/r, s = b/r
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;
            state->upper_triangular_R[i][i] = r;
            // Update remaining entries in the row
            for (int j = i + 1; j < num_coeff; ++j) {
                double temp = state->upper_triangular_R[i][j];
                state->upper_triangular_R[i][j] = c * temp + s * r_row[j];
                r_row[j] = -s * temp + c * r_row[j];
            }
            // Update the Qᵀ*b vector
            double temp_b = state->Q_transpose_b[i];
            state->Q_transpose_b[i] = c * temp_b + s * Q_tb_new;
            Q_tb_new = -s * temp_b + c * Q_tb_new;
        }
    }
}

/**
 * @brief Incrementally downdates the QR decomposition when removing an old row.
 *
 * When the sliding window is full, the oldest data point must be removed. This function
 * uses Givens rotations in a reverse manner to remove the influence of an old row from
 * the current QR decomposition.
 *
 * @param state Pointer to the current RegressionState.
 * @param old_row The row (Chebyshev basis vector) corresponding to the old data point.
 * @param old_b The measurement of the old data point.
 */
static void updateQR_RemoveRow(RegressionState *state, const double *old_row, double old_b) {
    int num_coeff = state->polynomial_degree + 1;
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    memcpy(r_row, old_row, sizeof(double) * num_coeff);
    double Q_tb_old = old_b;
    // Reverse Givens rotations to downdate the QR factors
    for (int i = 0; i < num_coeff; ++i) {
        double a = state->upper_triangular_R[i][i];
        double b = r_row[i];
        if (fabs(b) > TOL) {
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;
            state->upper_triangular_R[i][i] = r;
            for (int j = i + 1; j < num_coeff; ++j) {
                double temp = state->upper_triangular_R[i][j];
                state->upper_triangular_R[i][j] = c * temp - s * r_row[j];
                r_row[j] = s * temp + c * r_row[j];
            }
            double temp_b = state->Q_transpose_b[i];
            state->Q_transpose_b[i] = c * temp_b - s * Q_tb_old;
            Q_tb_old = s * temp_b + c * Q_tb_old;
        }
    }
}

/**
 * @brief Recomputes the full QR decomposition (using Householder reflections with pivoting).
 *
 * For numerical stability, especially after many incremental updates, the QR decomposition
 * is recomputed from scratch using the current data window (augmented with regularization rows).
 *
 * The augmented matrix is built using the Chebyshev basis for each data point and then
 * regularization rows (scaled by sqrt(REGULARIZATION_PARAMETER)) are appended. Column pivoting
 * is applied for additional numerical stability.
 *
 * @param state Pointer to the current RegressionState.
 * @param data Pointer to the array of raw data points.
 */
static void recompute_qr_decomposition(RegressionState *state, const MqsRawDataPoint_t *data) {
    int num_data = state->current_num_points;
    int num_coeff = state->polynomial_degree + 1;
    int total_rows = num_data + num_coeff;
#ifdef USE_GLOBAL_SCRATCH_SPACE
    if (total_rows > MAX_TOTAL_ROWS) return;
#else
    double augmented_matrix_A[total_rows][MAX_POLYNOMIAL_DEGREE + 1];
    double augmented_vector_b[total_rows];
#endif
    // Initialize the augmented matrix and vector to zero
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
#ifdef USE_GLOBAL_SCRATCH_SPACE
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#else
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#endif

    // Setup column indices for pivoting
    int col_indices[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coeff; ++j)
        col_indices[j] = j;

    // Determine the normalization range from the data window
    double min_x = (double)(state->total_data_points_added - num_data);
    double max_x = (double)(state->total_data_points_added - 1);

    // Build the design matrix using the Chebyshev basis
    for (int i = 0; i < num_data; ++i) {
        unsigned short data_index = state->oldest_data_index + i;
        double x = (double)(state->total_data_points_added - num_data + i);
        double normalized_x = (fabs(max_x - min_x) < TOL) ? 0.0 :
                                (2.0 * (x - min_x) / (max_x - min_x) - 1.0);
        augmented_matrix_A[i][0] = 1.0;
        if (num_coeff > 1)
            augmented_matrix_A[i][1] = normalized_x;
        for (int j = 2; j < num_coeff; ++j) {
            // Use recurrence to fill in the Chebyshev basis terms
            augmented_matrix_A[i][j] = 2.0 * normalized_x * augmented_matrix_A[i][j - 1] - augmented_matrix_A[i][j - 2];
        }
        // The dependent variable is the phase angle measurement.
        augmented_vector_b[i] = data[data_index].phaseAngle;
    }
    // Append regularization rows to the bottom of the augmented matrix to prevent overfitting.
    for (int i = 0; i < num_coeff; ++i) {
        int row = num_data + i;
        augmented_matrix_A[row][i] = sqrt(REGULARIZATION_PARAMETER);
#ifdef USE_GLOBAL_SCRATCH_SPACE
        augmented_vector_b[row] = 0.0;
#else
        augmented_vector_b[row] = 0.0;
#endif
    }
    // Reset the R and Q_transpose_b in the state
    memset(state->upper_triangular_R, 0, sizeof(state->upper_triangular_R));
    memset(state->Q_transpose_b, 0, sizeof(state->Q_transpose_b));

    // Compute column norms for pivoting (only for the upper triangular part)
    double col_norms[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coeff; ++j) {
        double sum = 0.0;
        for (int i = 0; i < total_rows; ++i)
            sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
        col_norms[j] = sqrt(sum);
    }
    // Householder QR with column pivoting
    for (int k = 0; k < num_coeff; ++k) {
        int max_col = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < num_coeff; ++j) {
            if (col_norms[j] > max_norm) {
                max_norm = col_norms[j];
                max_col = j;
            }
        }
        // Swap columns if needed
        if (max_col != k) {
            for (int i = 0; i < total_rows; ++i) {
                double temp = augmented_matrix_A[i][k];
                augmented_matrix_A[i][k] = augmented_matrix_A[i][max_col];
                augmented_matrix_A[i][max_col] = temp;
            }
            int temp_idx = col_indices[k];
            col_indices[k] = col_indices[max_col];
            col_indices[max_col] = temp_idx;
            double temp_norm = col_norms[k];
            col_norms[k] = col_norms[max_col];
            col_norms[max_col] = temp_norm;
        }
        // Compute the norm of column k from row k to end
        double sigma = 0.0;
        for (int i = k; i < total_rows; ++i) {
            double val = augmented_matrix_A[i][k];
            sigma += val * val;
        }
        sigma = sqrt(sigma);
        if (sigma < TOL) continue;
        double vk = augmented_matrix_A[k][k] + ((augmented_matrix_A[k][k] >= 0.0) ? sigma : -sigma);
        if (fabs(vk) < TOL) continue;
        double beta = 1.0 / (sigma * vk);
        for (int i = k + 1; i < total_rows; ++i)
            augmented_matrix_A[i][k] /= vk;
        augmented_matrix_A[k][k] = -sigma;
        // Apply the Householder transformation to remaining columns
        for (int j = k + 1; j < num_coeff; ++j) {
            double s = 0.0;
            for (int i = k; i < total_rows; ++i)
                s += augmented_matrix_A[i][k] * augmented_matrix_A[i][j];
            s *= beta;
            for (int i = k; i < total_rows; ++i)
                augmented_matrix_A[i][j] -= augmented_matrix_A[i][k] * s;
        }
        // Apply transformation to the right-hand side vector
        double s = 0.0;
        for (int i = k; i < total_rows; ++i)
            s += augmented_matrix_A[i][k] * augmented_vector_b[i];
        s *= beta;
        for (int i = k; i < total_rows; ++i)
            augmented_vector_b[i] -= augmented_matrix_A[i][k] * s;
        // Zero out the subdiagonal elements explicitly
        for (int i = k + 1; i < total_rows; ++i)
            augmented_matrix_A[i][k] = 0.0;
        // Update the column norms for remaining columns
        for (int j = k + 1; j < num_coeff; ++j) {
            double sum = 0.0;
            for (int i = k + 1; i < total_rows; ++i)
                sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
            col_norms[j] = sqrt(sum);
        }
    }
    // Extract the upper-triangular matrix R and the transformed right-hand side vector Q_transpose_b
    for (int i = 0; i < num_coeff; ++i) {
        for (int j = i; j < num_coeff; ++j)
            state->upper_triangular_R[i][j] = augmented_matrix_A[i][j];
        state->Q_transpose_b[i] = augmented_vector_b[i];
    }
    memcpy(state->col_permutations, col_indices, sizeof(int) * num_coeff);
}

/**
 * @brief Solves the upper-triangular system R*x = Qᵀ*b by back substitution.
 *
 * This function computes the regression coefficients by performing back substitution
 * on the upper-triangular matrix R. The coefficients are then rearranged according to the
 * column permutation indices.
 *
 * @param state Pointer to the current RegressionState.
 */
static inline void solve_for_coefficients(RegressionState *state) {
    int num_coeff = state->polynomial_degree + 1;
    double temp_coeff[MAX_POLYNOMIAL_DEGREE + 1] = {0};
    // Back substitution loop: solve from bottom row upward.
    for (int i = num_coeff - 1; i >= 0; --i) {
        double sum = state->Q_transpose_b[i];
        for (int j = i + 1; j < num_coeff; ++j)
            sum -= state->upper_triangular_R[i][j] * temp_coeff[j];
        if (fabs(state->upper_triangular_R[i][i]) < TOL)
            temp_coeff[i] = 0.0;
        else
            temp_coeff[i] = sum / state->upper_triangular_R[i][i];
    }
    // Rearrange coefficients according to the column permutation
    for (int i = 0; i < num_coeff; ++i) {
        int permuted_index = state->col_permutations[i];
        state->coefficients[permuted_index] = temp_coeff[i];
    }
}

/**
 * @brief Computes the 1-norm of an upper-triangular matrix R.
 *
 * The 1-norm is defined as the maximum absolute column sum. Since R is upper-triangular,
 * only the entries where i <= j are nonzero.
 *
 * @param R The upper-triangular matrix.
 * @param size The dimension of the matrix.
 * @return The computed 1-norm of R.
 */
static double compute_R_norm_1(const double R[MAX_POLYNOMIAL_DEGREE+1][MAX_POLYNOMIAL_DEGREE+1], int size) {
    double norm = 0.0;
    for (int j = 0; j < size; j++) {
        double colSum = 0.0;
        for (int i = 0; i <= j; i++) {
            colSum += fabs(R[i][j]);
        }
        if (colSum > norm)
            norm = colSum;
    }
    return norm;
}

/**
 * @brief Estimates the 1-norm of the inverse of R (‖R⁻¹‖₁) using an iterative algorithm.
 *
 * The algorithm starts with a normalized vector x and solves R*z = x by back substitution.
 * It then computes the 1-norm of z and normalizes z, iterating until convergence.
 *
 * @param R The upper-triangular matrix.
 * @param size The dimension of R.
 * @return An estimate of ‖R⁻¹‖₁.
 */
static double estimate_R_inverse_norm_1(const double R[MAX_POLYNOMIAL_DEGREE+1][MAX_POLYNOMIAL_DEGREE+1], int size) {
    double x[MAX_POLYNOMIAL_DEGREE+1], z[MAX_POLYNOMIAL_DEGREE+1];
    for (int i = 0; i < size; i++) {
        x[i] = 1.0 / size;
    }
    int maxIter = 5;
    double prevEst = 0.0, est = 0.0;
    const double tolIter = 1e-1;
    for (int iter = 0; iter < maxIter; iter++) {
        // Solve R*z = x via back substitution
        for (int i = size - 1; i >= 0; i--) {
            double sum = x[i];
            for (int j = i + 1; j < size; j++) {
                sum -= R[i][j] * z[j];
            }
            z[i] = sum / R[i][i];
        }
        // Compute the 1-norm of z
        est = 0.0;
        for (int i = 0; i < size; i++) {
            est += fabs(z[i]);
        }
        // Normalize z
        for (int i = 0; i < size; i++) {
            z[i] /= est;
        }
        if (fabs(est - prevEst) < tolIter * est)
            break;
        memcpy(x, z, sizeof(double)*size);
        prevEst = est;
    }
    return est;
}

/**
 * @brief Computes an improved estimate of the condition number of R.
 *
 * The condition number is estimated as:
 *   κ(R) = ‖R‖₁ * ‖R⁻¹‖₁,
 * where ‖R‖₁ is computed directly and ‖R⁻¹‖₁ is estimated using an iterative procedure.
 *
 * @param R The upper-triangular matrix.
 * @param size The dimension of R.
 * @return The estimated condition number.
 */
static double compute_condition_number_improved(const double R[MAX_POLYNOMIAL_DEGREE+1][MAX_POLYNOMIAL_DEGREE+1], int size) {
    double norm_R = compute_R_norm_1(R, size);
    double norm_R_inv = estimate_R_inverse_norm_1(R, size);
    return norm_R * norm_R_inv;
}

/**
 * @brief Computes the first-order derivative of the fitted function.
 *
 * The fitted function is represented in the Chebyshev basis:
 *   f(r) = a0*T0(r) + a1*T1(r) + a2*T2(r) + a3*T3(r),
 * with r = 2*(x - min_x)/(max_x - min_x) - 1.
 *
 * The derivative with respect to r is computed using:
 *   f'(r) = a1*T1'(r) + a2*T2'(r) + a3*T3'(r),
 * where for degree 3:
 *   T1'(r) = 1, T2'(r) = 4r, T3'(r) = 12r² - 3.
 *
 * Finally, by chain rule, df/dx = f'(r) * (dr/dx), with dr/dx = 2/(max_x - min_x).
 *
 * @param state Pointer to the current RegressionState.
 * @param x The current x value (used for normalization).
 * @return The first-order derivative (gradient) df/dx.
 */
double calculate_first_order_gradient_chebyshev(const RegressionState *state, double x) {
    double min_x, max_x;
    if (state->current_num_points < state->max_num_points) {
        min_x = 0.0;
        max_x = (double)state->total_data_points_added - 1;
    } else {
        min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)state->total_data_points_added - 1;
    }
    // Normalize x to obtain r in [-1, 1]
    double r = (fabs(max_x - min_x) < TOL) ? 0.0 : (2.0 * (x - min_x) / (max_x - min_x) - 1.0);
    double dfd_r = 0.0;
    // Compute derivative with respect to r using known Chebyshev derivatives
    if (state->polynomial_degree >= 1)
        dfd_r += state->coefficients[1] * 1.0;         // T1'(r)
    if (state->polynomial_degree >= 2)
        dfd_r += state->coefficients[2] * 4.0 * r;       // T2'(r)=4r
    if (state->polynomial_degree >= 3)
        dfd_r += state->coefficients[3] * (12.0 * r * r - 3.0); // T3'(r)=12r^2 - 3
    // Compute derivative of r with respect to x
    double drdx = (fabs(max_x - min_x) < TOL) ? 0.0 : 2.0 / (max_x - min_x);
    return dfd_r * drdx;
}

/**
 * @brief Computes the second-order derivative (curvature) of the fitted function.
 *
 * For the Chebyshev-based representation, the second derivative with respect to r is:
 *   f''(r) = a2*T2''(r) + a3*T3''(r),
 * with:
 *   T2''(r) = 4,
 *   T3''(r) = 24r.
 *
 * Then, by the chain rule (and noting that d²r/dx² = 0 for a linear normalization),
 * the second derivative with respect to x is:
 *
 *   f''(x) = f''(r) * (dr/dx)².
 *
 * @param state Pointer to the current RegressionState.
 * @param x The current x value (used for normalization).
 * @return The second-order derivative (curvature) f''(x).
 */
double calculate_second_order_gradient_chebyshev(const RegressionState *state, double x) {
    double min_x, max_x;
    if (state->current_num_points < state->max_num_points) {
        min_x = 0.0;
        max_x = (double)state->total_data_points_added - 1;
    } else {
        min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)state->total_data_points_added - 1;
    }
    double r = (fabs(max_x - min_x) < TOL) ? 0.0 : (2.0 * (x - min_x) / (max_x - min_x) - 1.0);
    double d2fd_r2 = 0.0;
    // Compute second derivative contributions from T2 and T3 for degree 3
    if (state->polynomial_degree >= 2)
        d2fd_r2 += state->coefficients[2] * 4.0;       // T2''(r)=4
    if (state->polynomial_degree >= 3)
        d2fd_r2 += state->coefficients[3] * (24.0 * r);  // T3''(r)=24r
    double drdx = (fabs(max_x - min_x) < TOL) ? 0.0 : 2.0 / (max_x - min_x);
    return d2fd_r2 * drdx * drdx;
}

/**
 * @brief Adds a new data point to the regression model.
 *
 * This function performs the following steps:
 *   1. Normalizes the new data point's x value to the Chebyshev domain [-1,1].
 *   2. (Optionally) Applies a forgetting factor by scaling the current QR factors.
 *   3. Updates the QR decomposition with the new row using Givens rotations.
 *   4. If the sliding window is full, removes the oldest data point.
 *   5. Updates the regression coefficients by solving R*x = Qᵀ*b.
 *   6. Estimates the condition number; if too high, recomputes the full QR decomposition.
 *
 * @param state Pointer to the current RegressionState.
 * @param data Pointer to the array of data points.
 * @param data_index Index of the new data point to add.
 */
void add_data_point_to_regression(RegressionState *state, const MqsRawDataPoint_t *data, unsigned short data_index) {
    int num_coeff = state->polynomial_degree + 1;
    double measurement = data[data_index].phaseAngle;
    double x_value = (double)state->total_data_points_added;
    double new_row[MAX_POLYNOMIAL_DEGREE + 1];
    unsigned short new_window_size = (state->current_num_points < state->max_num_points) ?
                                       (state->current_num_points + 1) : state->max_num_points;
    double min_x, max_x, normalized_x;
    // If only one data point, normalization is set to zero by definition.
    if (new_window_size == 1) {
        normalized_x = 0.0;
    } else {
        // Determine the normalization range based on the current window.
        if (state->current_num_points < state->max_num_points)
            min_x = 0.0;
        else
            min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)state->total_data_points_added;
        normalized_x = (fabs(max_x - min_x) < TOL) ? 0.0 :
                         (2.0 * (x_value - min_x) / (max_x - min_x) - 1.0);
    }
    
#ifdef USE_FORGETTING_FACTOR
    {
        /* 
         * Apply the forgetting factor: scale the existing QR factors by sqrt(lambda).
         * This discounts older data so that the algorithm adapts more quickly to recent changes.
         */
        double sqrt_lambda = sqrt(FORGETTING_FACTOR);
        for (int i = 0; i < num_coeff; i++) {
            for (int j = i; j < num_coeff; j++) {
                state->upper_triangular_R[i][j] *= sqrt_lambda;
            }
            state->Q_transpose_b[i] *= sqrt_lambda;
        }
    }
#endif

    // Compute the Chebyshev basis vector for the new data point.
    compute_chebyshev_basis(normalized_x, state->polynomial_degree, new_row);
    // Update the QR decomposition by adding the new row.
    updateQR_AddRow(state, new_row, measurement);
    // If the sliding window is full, remove the oldest data point.
    if (state->current_num_points >= state->max_num_points) {
        double old_x_value = (double)(state->total_data_points_added - state->max_num_points);
        double old_row[MAX_POLYNOMIAL_DEGREE + 1];
        min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)(state->total_data_points_added - 1);
        double normalized_old = (fabs(max_x - min_x) < TOL) ? 0.0 :
                                (2.0 * (old_x_value - min_x) / (max_x - min_x) - 1.0);
        compute_chebyshev_basis(normalized_old, state->polynomial_degree, old_row);
        double old_measurement = data[state->oldest_data_index].phaseAngle;
        // Downdate the QR factors to remove the contribution of the oldest data point.
        updateQR_RemoveRow(state, old_row, old_measurement);
        state->oldest_data_index++;
    } else {
        state->current_num_points++;
    }
    state->total_data_points_added++;
    // Solve for the regression coefficients from the current QR factors.
    solve_for_coefficients(state);
    // Estimate the condition number using the improved 1-norm method.
    double cond = compute_condition_number_improved(state->upper_triangular_R, num_coeff);
    // If the condition number is too high, recompute the full QR decomposition.
    if (cond > CONDITION_NUMBER_THRESHOLD) {
        recompute_qr_decomposition(state, data);
        solve_for_coefficients(state);
    }
}

/**
 * @brief Processes a sequence of data points and computes gradients.
 *
 * This function iterates over a specified segment of the data array, updating the regression
 * state and computing the gradient (first-order derivative) at each step using the provided
 * gradient calculation function.
 *
 * @param measurements Pointer to the array of data points.
 * @param length Number of data points to process.
 * @param start_index Index in the array from which to start processing.
 * @param degree The degree of the Chebyshev polynomial basis to use.
 * @param calculate_gradient Function pointer to the gradient calculation function.
 * @param result Pointer to the structure where computed gradients are stored.
 */
void trackGradients(const MqsRawDataPoint_t *measurements, unsigned short length, unsigned short start_index, unsigned char degree,
                    double (*calculate_gradient)(const RegressionState *, double), GradientCalculationResult *result) {
    result->size = 0;
    RegressionState state;
    initialize_regression_state(&state, degree, RLS_WINDOW);
    // Process each data point from start_index up to start_index + length
    for (unsigned short i = 0; i < length; ++i) {
        unsigned short current_index = start_index + i;
        add_data_point_to_regression(&state, measurements, current_index);
        double x_value = (double)(state.total_data_points_added - 1);
        // Only compute a gradient if we have enough data points (at least degree+1)
        if (state.current_num_points >= (state.polynomial_degree + 1)) {
            double grad = calculate_gradient(&state, x_value);
            if (result->size < RLS_WINDOW) {
                result->gradients[result->size++] = grad;
            } else {
                break;
            }
        }
    }
}


