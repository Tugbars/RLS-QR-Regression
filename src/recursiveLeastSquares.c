/**
 * @file main.c
 * @brief Chebyshev-Based Recursive Least Squares (RLS) Regression with Optional Forgetting Factor,
 * Adaptive Tolerances, Robust Pivoting Checks, and Occasional SVD-based Condition Number Verification.
 *
 * This implementation uses a Chebyshev polynomial basis to represent the regression model.
 * Incremental QR updates are performed using Givens rotations (with adaptive tolerances) so that
 * new data points can be added and old ones removed. A forgetting factor is optionally applied to
 * discount older measurements. In addition, the algorithm includes:
 *   - A robust pivoting check that triggers full reorthogonalization if any pivot element becomes too small.
 *   - An occasional SVD-based condition number estimation (via eigenvalue estimation of R^T R) to
 *     further ensure numerical stability.
 *
 * The Chebyshev basis is computed via the recurrence:
 *   T₀(x) = 1, T₁(x) = x, Tₙ(x) = 2 * x * Tₙ₋₁(x) - Tₙ₋₂(x) for n ≥ 2.
 *
 * The independent variable is normalized to r ∈ [−1,1] via:
 *
 *   r = 2*(x - min_x)/(max_x - min_x) - 1.
 *
 * The fitted function is:
 *
 *   f(r) = a₀*T₀(r) + a₁*T₁(r) + a₂*T₂(r) + a₃*T₃(r),
 *
 * and its derivative is computed via the chain rule.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdarg.h>
#include "recursiveLeastSquares.h"

/* ********************************************************************
 * Debug Macros (2 levels)
 * ********************************************************************/

/**
 * @def DEBUG_LEVEL
 * @brief Set the desired debug level:
 *        0: No debug output,
 *        1: Math output only,
 *        2: Transition output in addition to math output.
 */
#define DEBUG_LEVEL 2  // Adjust as needed

#if DEBUG_LEVEL >= 1
  /**
   * @brief Debug output for mathematical details.
   */
  #define MATH_DEBUG(fmt, ...) printf("[MATH] " fmt, ##__VA_ARGS__)
#else
  #define MATH_DEBUG(fmt, ...) do {} while(0)
#endif

#if DEBUG_LEVEL >= 2
  /**
   * @brief Debug output for transition and step information.
   */
  #define TRANSITION_DEBUG(fmt, ...) printf("[TRANSITION] " fmt, ##__VA_ARGS__)
#else
  #define TRANSITION_DEBUG(fmt, ...) do {} while(0)
#endif

/* ********************************************************************
 * Configuration macros (in implementation file only)
 * ********************************************************************/

#define MAX_POLYNOMIAL_DEGREE 4           /**< Maximum degree of the Chebyshev basis */
#define RLS_WINDOW 30                     /**< Sliding window size */
#define REGULARIZATION_PARAMETER 1e-4     /**< Regularization parameter */
#define CONDITION_NUMBER_THRESHOLD 1e4    /**< Condition number threshold for reorthogonalization */
#define TOL (DBL_EPSILON * 3)             /**< Base tolerance for near-zero comparisons */

#define USE_FORGETTING_FACTOR             /**< Enable exponential forgetting */
#ifdef USE_FORGETTING_FACTOR
  #define FORGETTING_FACTOR 0.55         /**< Forgetting factor λ: 0 < λ ≤ 1 */
#endif

#define USE_GLOBAL_SCRATCH_SPACE          /**< Use global scratch space for temporary matrices */
#ifdef USE_GLOBAL_SCRATCH_SPACE
  #define MAX_TOTAL_ROWS (RLS_WINDOW + MAX_POLYNOMIAL_DEGREE + 1)
  static double augmented_matrix_A[MAX_TOTAL_ROWS][MAX_POLYNOMIAL_DEGREE + 1];
  static double augmented_vector_b[MAX_TOTAL_ROWS];
#endif

#define USE_SVD_CONDITION_CHECK           /**< Enable SVD-based condition number checks */
#define SVD_CHECK_INTERVAL 15             /**< Perform SVD check every 15 updates */

/* ********************************************************************
 * Forward declarations of internal functions
 * ********************************************************************/

// Adaptive tolerance: scales the base TOL by the magnitude of two values.
static double tol_for_values(double a, double b);

// Checks if any pivot element in R is too small relative to its column norm.
static int needs_reorthogonalization(const RegressionState *state, int num_coeff);

// SVD-based condition number estimation function.
static double svd_condition_number(const double R[][MAX_POLYNOMIAL_DEGREE+1], int size);

// Compute Chebyshev basis for a normalized value.
static void compute_chebyshev_basis(double normalized_x, int degree, double *basis);

// Incrementally update the QR decomposition by adding a new row.
static void updateQR_AddRow(RegressionState *state, const double *new_row, double new_b);

// Incrementally downdate the QR decomposition by removing an old row.
static void updateQR_RemoveRow(RegressionState *state, const double *old_row, double old_b);

// Recompute the full QR decomposition using Householder reflections with column pivoting.
static void recompute_qr_decomposition(RegressionState *state, const MqsRawDataPoint_t *data);

// Solve for the regression coefficients via back substitution.
static inline void solve_for_coefficients(RegressionState *state);

// Compute the 1-norm of the upper-triangular matrix R.
static double compute_R_norm_1(const double R[][MAX_POLYNOMIAL_DEGREE+1], int size);

// Iteratively estimate the 1-norm of the inverse of R.
static double estimate_R_inverse_norm_1(const double R[][MAX_POLYNOMIAL_DEGREE+1], int size);

// Compute an improved condition number using the 1-norm estimates.
static double compute_condition_number_improved(const double R[][MAX_POLYNOMIAL_DEGREE+1], int size);

/* ********************************************************************
 * SVD-based condition number estimation
 * ********************************************************************/

static double svd_condition_number(const double R[][MAX_POLYNOMIAL_DEGREE+1], int size) {
    int i, j, k;
    double A[MAX_POLYNOMIAL_DEGREE+1][MAX_POLYNOMIAL_DEGREE+1] = {0};

    // Compute A = Rᵀ * R (R is upper-triangular).
    for (i = 0; i < size; i++) {
        for (j = i; j < size; j++) {
            double sum = 0.0;
            for (k = 0; k < size; k++) {
                double rki = (k < i) ? 0.0 : R[i][k];
                double rkj = (k < j) ? 0.0 : R[j][k];
                sum += rki * rkj;
            }
            A[i][j] = sum;
            A[j][i] = sum;
        }
    }

    // Power iteration to estimate the maximum eigenvalue of A.
    double v[MAX_POLYNOMIAL_DEGREE+1];
    double v_new[MAX_POLYNOMIAL_DEGREE+1];
    for (i = 0; i < size; i++) {
        v[i] = 1.0;
    }
    double lambda_max = 0.0;
    for (int iter = 0; iter < 100; iter++) {
        for (i = 0; i < size; i++) {
            double sum = 0.0;
            for (j = 0; j < size; j++) {
                sum += A[i][j] * v[j];
            }
            v_new[i] = sum;
        }
        double norm = 0.0;
        for (i = 0; i < size; i++) {
            norm += v_new[i] * v_new[i];
        }
        norm = sqrt(norm);
        for (i = 0; i < size; i++) {
            v_new[i] /= norm;
        }
        double lambda_est = 0.0;
        for (i = 0; i < size; i++) {
            double sum = 0.0;
            for (j = 0; j < size; j++) {
                sum += A[i][j] * v_new[j];
            }
            lambda_est += v_new[i] * sum;
        }
        if (fabs(lambda_est - lambda_max) < 1e-6 * fabs(lambda_est)) {
            lambda_max = lambda_est;
            break;
        }
        lambda_max = lambda_est;
        memcpy(v, v_new, sizeof(double)*size);
    }

    // Inverse power iteration for the smallest eigenvalue of A.
    double w[MAX_POLYNOMIAL_DEGREE+1];
    for (i = 0; i < size; i++) {
        v[i] = 1.0;
    }
    double lambda_min_inv = 0.0;
    for (int iter = 0; iter < 100; iter++) {
        double tempA[MAX_POLYNOMIAL_DEGREE+1][MAX_POLYNOMIAL_DEGREE+1];
        double b[MAX_POLYNOMIAL_DEGREE+1];
        for (i = 0; i < size; i++) {
            b[i] = v[i];
            for (j = 0; j < size; j++) {
                tempA[i][j] = A[i][j];
            }
        }
        // Gaussian elimination (forward elimination)
        for (i = 0; i < size; i++) {
            double pivot = tempA[i][i];
            for (j = i; j < size; j++) {
                tempA[i][j] /= pivot;
            }
            b[i] /= pivot;
            for (k = i + 1; k < size; k++) {
                double factor = tempA[k][i];
                for (j = i; j < size; j++) {
                    tempA[k][j] -= factor * tempA[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
        // Back substitution
        for (i = size - 1; i >= 0; i--) {
            w[i] = b[i];
            for (j = i + 1; j < size; j++) {
                w[i] -= tempA[i][j] * w[j];
            }
        }
        memcpy(v_new, w, sizeof(double)*size);
        double norm = 0.0;
        for (i = 0; i < size; i++) {
            norm += v_new[i] * v_new[i];
        }
        norm = sqrt(norm);
        for (i = 0; i < size; i++) {
            v_new[i] /= norm;
        }
        double lambda_est = 0.0;
        for (i = 0; i < size; i++) {
            double sum = 0.0;
            for (j = 0; j < size; j++) {
                sum += A[i][j] * v_new[j];
            }
            lambda_est += v_new[i] * sum;
        }
        if (fabs(lambda_est - lambda_min_inv) < 1e-6 * fabs(lambda_est))
            break;
        lambda_min_inv = lambda_est;
        memcpy(v, v_new, sizeof(double)*size);
    }
    double lambda_min = 1.0 / lambda_min_inv;
    double cond = sqrt(lambda_max / lambda_min);
    return cond;
}

/* ********************************************************************
 * Internal Utility Functions
 * ********************************************************************/

static double tol_for_values(double a, double b) {
    return TOL * (fabs(a) + fabs(b) + 1.0);
}

static int needs_reorthogonalization(const RegressionState *state, int num_coeff) {
    for (int i = 0; i < num_coeff; i++) {
        double diag = fabs(state->upper_triangular_R[i][i]);
        double col_norm = 0.0;
        for (int j = i; j < num_coeff; j++) {
            col_norm += fabs(state->upper_triangular_R[i][j]);
        }
        if (col_norm > 1e-12 && (diag / col_norm) < 1e-2)
            return 1;
    }
    return 0;
}

/* ********************************************************************
 * Public Functions
 * ********************************************************************/

/**
 * @brief Initializes the regression state structure.
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
 * @param normalized_x The normalized input value (should be in [-1, 1]).
 * @param degree The highest polynomial degree to compute.
 * @param basis Pointer to an array to store the computed basis values.
 */
static void compute_chebyshev_basis(double normalized_x, int degree, double *basis) {
    basis[0] = 1.0;
    if (degree >= 1) {
        basis[1] = normalized_x;
    }
    for (int i = 2; i <= degree; i++) {
        basis[i] = 2.0 * normalized_x * basis[i - 1] - basis[i - 2];
    }
}

/**
 * @brief Incrementally updates the QR decomposition when adding a new row.
 *
 * @param state Pointer to the current RegressionState.
 * @param new_row The new row (Chebyshev basis vector) to be added.
 * @param new_b The new measurement corresponding to the new row.
 */
static void updateQR_AddRow(RegressionState *state, const double *new_row, double new_b) {
    int num_coeff = state->polynomial_degree + 1;
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    memcpy(r_row, new_row, sizeof(double) * num_coeff);
    double Q_tb_new = new_b;
    for (int i = 0; i < num_coeff; ++i) {
        double a = state->upper_triangular_R[i][i];
        double b = r_row[i];
        double tol_dynamic = tol_for_values(a, b);
        if (fabs(b) > tol_dynamic) {
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;
            state->upper_triangular_R[i][i] = r;
            for (int j = i + 1; j < num_coeff; ++j) {
                double temp = state->upper_triangular_R[i][j];
                state->upper_triangular_R[i][j] = c * temp + s * r_row[j];
                r_row[j] = -s * temp + c * r_row[j];
            }
            double temp_b = state->Q_transpose_b[i];
            state->Q_transpose_b[i] = c * temp_b + s * Q_tb_new;
            Q_tb_new = -s * temp_b + c * Q_tb_new;
        }
    }
}

/**
 * @brief Incrementally downdates the QR decomposition when removing an old row.
 *
 * @param state Pointer to the current RegressionState.
 * @param old_row The Chebyshev basis row corresponding to the old data point.
 * @param old_b The measurement of the old data point.
 */
static void updateQR_RemoveRow(RegressionState *state, const double *old_row, double old_b) {
    int num_coeff = state->polynomial_degree + 1;
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    memcpy(r_row, old_row, sizeof(double) * num_coeff);
    double Q_tb_old = old_b;
    for (int i = 0; i < num_coeff; ++i) {
        double a = state->upper_triangular_R[i][i];
        double b = r_row[i];
        double tol_dynamic = tol_for_values(a, b);
        if (fabs(b) > tol_dynamic) {
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
 * @brief Recomputes the full QR decomposition using Householder reflections with column pivoting.
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
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
#ifdef USE_GLOBAL_SCRATCH_SPACE
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#else
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#endif

    int col_indices[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coeff; ++j)
        col_indices[j] = j;

    double min_x = (double)(state->total_data_points_added - num_data);
    double max_x = (double)(state->total_data_points_added - 1);

    for (int i = 0; i < num_data; ++i) {
        unsigned short data_index = state->oldest_data_index + i;
        double x = (double)(state->total_data_points_added - num_data + i);
        double normalized_x = (fabs(max_x - min_x) < TOL) ? 0.0 :
                                (2.0 * (x - min_x) / (max_x - min_x) - 1.0);
        augmented_matrix_A[i][0] = 1.0;
        if (num_coeff > 1)
            augmented_matrix_A[i][1] = normalized_x;
        for (int j = 2; j < num_coeff; ++j) {
            augmented_matrix_A[i][j] = 2.0 * normalized_x * augmented_matrix_A[i][j - 1] - augmented_matrix_A[i][j - 2];
        }
        augmented_vector_b[i] = data[data_index].phaseAngle;
    }

    for (int i = 0; i < num_coeff; ++i) {
        int row = num_data + i;
        augmented_matrix_A[row][i] = sqrt(REGULARIZATION_PARAMETER);
#ifdef USE_GLOBAL_SCRATCH_SPACE
        augmented_vector_b[row] = 0.0;
#else
        augmented_vector_b[row] = 0.0;
#endif
    }

    memset(state->upper_triangular_R, 0, sizeof(state->upper_triangular_R));
    memset(state->Q_transpose_b, 0, sizeof(state->Q_transpose_b));

    double col_norms[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coeff; ++j) {
        double sum = 0.0;
        for (int i = 0; i < total_rows; ++i)
            sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
        col_norms[j] = sqrt(sum);
    }

    for (int k = 0; k < num_coeff; ++k) {
        int max_col = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < num_coeff; ++j) {
            if (col_norms[j] > max_norm) {
                max_norm = col_norms[j];
                max_col = j;
            }
        }
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
        for (int j = k + 1; j < num_coeff; ++j) {
            double s = 0.0;
            for (int i = k; i < total_rows; ++i)
                s += augmented_matrix_A[i][k] * augmented_matrix_A[i][j];
            s *= beta;
            for (int i = k; i < total_rows; ++i)
                augmented_matrix_A[i][j] -= augmented_matrix_A[i][k] * s;
        }
        double s = 0.0;
        for (int i = k; i < total_rows; ++i)
            s += augmented_matrix_A[i][k] * augmented_vector_b[i];
        s *= beta;
        for (int i = k; i < total_rows; ++i)
            augmented_vector_b[i] -= augmented_matrix_A[i][k] * s;
        for (int i = k + 1; i < total_rows; ++i)
            augmented_matrix_A[i][k] = 0.0;
        for (int j = k + 1; j < num_coeff; ++j) {
            double sum = 0.0;
            for (int i = k + 1; i < total_rows; ++i)
                sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
            col_norms[j] = sqrt(sum);
        }
    }
    for (int i = 0; i < num_coeff; ++i) {
        for (int j = i; j < num_coeff; ++j)
            state->upper_triangular_R[i][j] = augmented_matrix_A[i][j];
        state->Q_transpose_b[i] = augmented_vector_b[i];
    }
    memcpy(state->col_permutations, col_indices, sizeof(int) * num_coeff);
}

/**
 * @brief Solves the upper-triangular system R*x = Qᵀ*b using back substitution.
 *
 * The solution is rearranged according to the column permutation.
 *
 * @param state Pointer to the RegressionState.
 */
static inline void solve_for_coefficients(RegressionState *state) {
    int num_coeff = state->polynomial_degree + 1;
    double temp_coeff[MAX_POLYNOMIAL_DEGREE + 1] = {0};
    for (int i = num_coeff - 1; i >= 0; --i) {
        double sum = state->Q_transpose_b[i];
        for (int j = i + 1; j < num_coeff; ++j)
            sum -= state->upper_triangular_R[i][j] * temp_coeff[j];
        if (fabs(state->upper_triangular_R[i][i]) < 1e-12)
            temp_coeff[i] = 0.0;
        else
            temp_coeff[i] = sum / state->upper_triangular_R[i][i];
    }
    for (int i = 0; i < num_coeff; ++i) {
        int permuted_index = state->col_permutations[i];
        state->coefficients[permuted_index] = temp_coeff[i];
    }
}

/**
 * @brief Computes the 1-norm of an upper-triangular matrix R.
 *
 * Only the upper triangular part (i ≤ j) is considered.
 *
 * @param R The upper-triangular matrix.
 * @param size The dimension of R.
 * @return The computed 1-norm.
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
 * @brief Iteratively estimates the 1-norm of the inverse of R.
 *
 * Uses back substitution and normalization to approximate ‖R⁻¹‖₁.
 *
 * @param R The upper-triangular matrix.
 * @param size The dimension of R.
 * @return Estimated 1-norm of R⁻¹.
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
        for (int i = size - 1; i >= 0; i--) {
            double sum = x[i];
            for (int j = i + 1; j < size; j++) {
                sum -= R[i][j] * z[j];
            }
            z[i] = sum / R[i][i];
        }
        est = 0.0;
        for (int i = 0; i < size; i++) {
            est += fabs(z[i]);
        }
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
 * @brief Computes an improved condition number of R using 1-norm estimates.
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
 * @brief Computes the first-order derivative (gradient) of the fitted function with respect to x.
 *
 * The fitted function is represented in the Chebyshev basis:
 *   f(r) = a₀*T₀(r) + a₁*T₁(r) + a₂*T₂(r) + a₃*T₃(r),
 * where r = 2*(x - min_x)/(max_x - min_x) - 1.
 *
 * The derivative with respect to r is:
 *   f'(r) = a₁*T₁'(r) + a₂*T₂'(r) + a₃*T₃'(r),
 * where for degree 3:
 *   T₁'(r) = 1, T₂'(r) = 4r, T₃'(r) = 12r² - 3.
 *
 * Then, by the chain rule,
 *
 *   df/dx = f'(r) * (dr/dx)  with dr/dx = 2/(max_x - min_x).
 *
 * @param state Pointer to the current RegressionState.
 * @param x The current x value (for normalization).
 * @return The computed first-order derivative.
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
    // Normalize x into the Chebyshev domain [-1,1]
    double r = (fabs(max_x - min_x) < TOL) ? 0.0 : (2.0 * (x - min_x) / (max_x - min_x) - 1.0);
    double dfd_r = 0.0;
    if (state->polynomial_degree >= 1)
        dfd_r += state->coefficients[1] * 1.0;         // T₁'(r)
    if (state->polynomial_degree >= 2)
        dfd_r += state->coefficients[2] * 4.0 * r;       // T₂'(r)=4r
    if (state->polynomial_degree >= 3)
        dfd_r += state->coefficients[3] * (12.0 * r * r - 3.0); // T₃'(r)=12r²-3
    double drdx = (fabs(max_x - min_x) < TOL) ? 0.0 : 2.0 / (max_x - min_x);
    MATH_DEBUG("x=%.4f, r=%.4f, f'(r)=%.4f, dr/dx=%.4f, df/dx=%.4f\n", x, r, dfd_r, drdx, dfd_r * drdx);
    return dfd_r * drdx;
}

/**
 * @brief Computes the second-order derivative (curvature) of the fitted function with respect to x.
 *
 * For the Chebyshev-based representation:
 *   f''(r) = a₂*T₂''(r) + a₃*T₃''(r),
 * where:
 *   T₂''(r) = 4, and T₃''(r) = 24r.
 *
 * Then, by the chain rule (with d²r/dx² = 0):
 *
 *   f''(x) = f''(r) * (dr/dx)², where dr/dx = 2/(max_x - min_x).
 *
 * @param state Pointer to the current RegressionState.
 * @param x The current x value (for normalization).
 * @return The computed second-order derivative.
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
    if (state->polynomial_degree >= 2)
        d2fd_r2 += state->coefficients[2] * 4.0;       // T₂''(r)=4
    if (state->polynomial_degree >= 3)
        d2fd_r2 += state->coefficients[3] * (24.0 * r);  // T₃''(r)=24r
    double drdx = (fabs(max_x - min_x) < TOL) ? 0.0 : 2.0 / (max_x - min_x);
    MATH_DEBUG("x=%.4f, r=%.4f, f''(r)=%.4f, (dr/dx)²=%.4f, f''(x)=%.4f\n", x, r, d2fd_r2, drdx*drdx, d2fd_r2*drdx*drdx);
    return d2fd_r2 * drdx * drdx;
}

/**
 * @brief Adds a new data point to the regression model and updates the QR factors.
 *
 * This function performs the following major steps:
 *  1. **Normalization:** The new data point's x value is normalized to the Chebyshev domain [-1,1]
 *     using the formula: r = 2*(x - min_x)/(max_x - min_x) - 1.
 *  2. **Forgetting Factor:** If enabled, the current QR factors (R and Qᵀ*b) are scaled by √(λ)
 *     to discount older data.
 *  3. **Incremental Update:** The Chebyshev basis vector for the new data point is computed, and the QR
 *     decomposition is updated using Givens rotations (with adaptive tolerances).
 *  4. **Downdating:** If the sliding window is full, the contribution of the oldest data point is removed
 *     using reverse Givens rotations.
 *  5. **Coefficient Update:** The regression coefficients are updated by solving the system R*x = Qᵀ*b.
 *  6. **Stability Checks:** The algorithm performs a robust pivoting check (via needs_reorthogonalization())
 *     and an SVD-based condition number check every SVD_CHECK_INTERVAL updates. If either check fails,
 *     the full QR decomposition is recomputed.
 *
 * @param state Pointer to the RegressionState.
 * @param data Pointer to the array of raw data points.
 * @param data_index Index of the new data point.
 */
void add_data_point_to_regression(RegressionState *state, const MqsRawDataPoint_t *data, unsigned short data_index) {
    int num_coeff = state->polynomial_degree + 1;
    double measurement = data[data_index].phaseAngle;
    double x_value = (double)state->total_data_points_added;
    double new_row[MAX_POLYNOMIAL_DEGREE + 1];
    unsigned short new_window_size = (state->current_num_points < state->max_num_points) ?
                                       (state->current_num_points + 1) : state->max_num_points;
    double min_x, max_x, normalized_x;
    
    /* Step 1: Normalize x to the Chebyshev domain */
    if (new_window_size == 1) {
        normalized_x = 0.0;  // When only one point, use 0 as normalized value.
    } else {
        if (state->current_num_points < state->max_num_points)
            min_x = 0.0;
        else
            min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)state->total_data_points_added;
        normalized_x = (fabs(max_x - min_x) < TOL) ? 0.0 :
                         (2.0 * (x_value - min_x) / (max_x - min_x) - 1.0);
    }
    TRANSITION_DEBUG("Normalized x=%.4f to r=%.4f\n", x_value, normalized_x);
    
#ifdef USE_FORGETTING_FACTOR
    /* Step 2: Apply the forgetting factor by scaling the current QR factors */
    {
        double sqrt_lambda = sqrt(FORGETTING_FACTOR);
        for (int i = 0; i < num_coeff; i++) {
            for (int j = i; j < num_coeff; j++) {
                state->upper_triangular_R[i][j] *= sqrt_lambda;
            }
            state->Q_transpose_b[i] *= sqrt_lambda;
        }
        TRANSITION_DEBUG("Applied forgetting factor (λ=%.2f)\n", FORGETTING_FACTOR);
    }
#endif

    /* Step 3: Compute the Chebyshev basis for the new data point and update QR via Givens rotations */
    compute_chebyshev_basis(normalized_x, state->polynomial_degree, new_row);
    MATH_DEBUG("Computed Chebyshev basis for new point: ");
    for (int i = 0; i < num_coeff; i++) {
        MATH_DEBUG("%.4f ", new_row[i]);
    }
    MATH_DEBUG("\n");
    updateQR_AddRow(state, new_row, measurement);
    TRANSITION_DEBUG("Updated QR decomposition with new data point (measurement=%.4f)\n", measurement);

    /* Step 4: Downdate if sliding window is full */
    if (state->current_num_points >= state->max_num_points) {
        double old_x_value = (double)(state->total_data_points_added - state->max_num_points);
        double old_row[MAX_POLYNOMIAL_DEGREE + 1];
        min_x = (double)(state->total_data_points_added - state->max_num_points);
        max_x = (double)(state->total_data_points_added - 1);
        double normalized_old = (fabs(max_x - min_x) < TOL) ? 0.0 :
                                (2.0 * (old_x_value - min_x) / (max_x - min_x) - 1.0);
        compute_chebyshev_basis(normalized_old, state->polynomial_degree, old_row);
        MATH_DEBUG("Computed Chebyshev basis for removal: ");
        for (int i = 0; i < num_coeff; i++) {
            MATH_DEBUG("%.4f ", old_row[i]);
        }
        MATH_DEBUG("\n");
        double old_measurement = data[state->oldest_data_index].phaseAngle;
        updateQR_RemoveRow(state, old_row, old_measurement);
        TRANSITION_DEBUG("Removed oldest data point (measurement=%.4f) at index %u\n", old_measurement, state->oldest_data_index);
        state->oldest_data_index++;
    } else {
        state->current_num_points++;
        TRANSITION_DEBUG("Increased current window size to %u\n", state->current_num_points);
    }
    state->total_data_points_added++;
    TRANSITION_DEBUG("Total data points added: %u\n", state->total_data_points_added);

    /* Step 5: Update the regression coefficients via back substitution */
    solve_for_coefficients(state);
    MATH_DEBUG("Updated coefficients: ");
    for (int i = 0; i < num_coeff; i++) {
        MATH_DEBUG("%.4f ", state->coefficients[i]);
    }
    MATH_DEBUG("\n");

    /* Step 6: Robust Stability Checks */
    if (needs_reorthogonalization(state, num_coeff)) {
        TRANSITION_DEBUG("Pivot check failed: reorthogonalizing...\n");
        recompute_qr_decomposition(state, data);
        solve_for_coefficients(state);
        TRANSITION_DEBUG("Reorthogonalization complete.\n");
    }
#ifdef USE_SVD_CONDITION_CHECK
    if (state->total_data_points_added % SVD_CHECK_INTERVAL == 0) {
        double cond = svd_condition_number(state->upper_triangular_R, num_coeff);
        MATH_DEBUG("SVD-based condition number: %.4e\n", cond);
        if (cond > CONDITION_NUMBER_THRESHOLD) {
            TRANSITION_DEBUG("SVD condition check failed: reorthogonalizing...\n");
            recompute_qr_decomposition(state, data);
            solve_for_coefficients(state);
            TRANSITION_DEBUG("Reorthogonalization complete after SVD check.\n");
        }
    } else {
        double cond = compute_condition_number_improved(state->upper_triangular_R, num_coeff);
        MATH_DEBUG("Improved condition number: %.4e\n", cond);
        if (cond > CONDITION_NUMBER_THRESHOLD) {
            TRANSITION_DEBUG("Condition number exceeded threshold: reorthogonalizing...\n");
            recompute_qr_decomposition(state, data);
            solve_for_coefficients(state);
            TRANSITION_DEBUG("Reorthogonalization complete after condition check.\n");
        }
    }
#else
    {
        double cond = compute_condition_number_improved(state->upper_triangular_R, num_coeff);
        if (cond > CONDITION_NUMBER_THRESHOLD) {
            TRANSITION_DEBUG("Condition number exceeded threshold: reorthogonalizing...\n");
            recompute_qr_decomposition(state, data);
            solve_for_coefficients(state);
        }
    }
#endif
}

/**
 * @brief Processes a sequence of data points and computes gradients.
 *
 * Iterates over a specified segment of the measurements array. For each data point,
 * it updates the regression model incrementally and, once there are enough points (≥ degree+1),
 * computes the gradient using the provided gradient function. Computed gradients are stored in the result.
 *
 * @param measurements Pointer to the array of raw data points.
 * @param length Number of data points to process.
 * @param start_index Starting index in the measurements array.
 * @param degree Degree of the Chebyshev polynomial basis.
 * @param calculate_gradient Function pointer for computing the gradient.
 * @param result Pointer to the structure where computed gradients are stored.
 */
void trackGradients(const MqsRawDataPoint_t *measurements, unsigned short length, unsigned short start_index, unsigned char degree,
                    double (*calculate_gradient)(const RegressionState *, double), GradientCalculationResult *result) {
    result->size = 0;
    result->valid = 0;
    RegressionState state;
    initialize_regression_state(&state, degree, RLS_WINDOW);
    TRANSITION_DEBUG("Starting gradient tracking from index %u for %u points.\n", start_index, length);
    for (unsigned short i = 0; i < length; ++i) {
        unsigned short current_index = start_index + i;
        add_data_point_to_regression(&state, measurements, current_index);
        double x_value = (double)(state.total_data_points_added - 1);
        if (state.current_num_points >= (state.polynomial_degree + 1)) {
            double grad = calculate_gradient(&state, x_value);
            MATH_DEBUG("At x=%.4f, computed gradient=%.4f\n", x_value, grad);
            if (result->size < RLS_WINDOW) {
                result->gradients[result->size++] = grad;
            } else {
                TRANSITION_DEBUG("Gradient array is full; stopping tracking.\n");
                break;
            }
        } else {
            TRANSITION_DEBUG("Not enough points to compute gradient (current=%u).\n", state.current_num_points);
        }
    }
    if (result->size > 0)
        result->valid = 1;
    TRANSITION_DEBUG("Gradient tracking completed with %u gradients computed.\n", result->size);
}


