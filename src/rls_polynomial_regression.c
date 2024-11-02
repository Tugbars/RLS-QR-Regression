/**
 * @file rls_polynomial_regression.c
 * @brief Implementation of Recursive Least Squares (RLS) for polynomial regression using QR decomposition.
 *
 * This file contains the implementation of an RLS algorithm for polynomial regression,
 * optimized for numerical stability using QR decomposition with Givens rotations.
 * The implementation incorporates regularization, periodic reorthogonalization,
 * and condition number monitoring.
 *
 * Upgrades implemented:
 * - Incremental QR Decomposition using Givens rotations for efficiency.
 * - Optimized polynomial basis computation using incremental multiplication.
 * - Optimized Householder transformations in recompute_qr_decomposition.
 * - Improved condition number estimation.
 */

#include "rls_polynomial_regression.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdarg.h>  // For va_list, va_start, va_end

// Uncomment the following line to use global scratch space
// #define USE_GLOBAL_SCRATCH_SPACE

/** Regularization parameter for Ridge regression */
#define REGULARIZATION_PARAMETER 1e-4  /**< Regularization parameter lambda */

/** Reorthogonalization interval */
#define REORTHOGONALIZATION_INTERVAL 50  /**< Interval for reorthogonalization */

/** Condition number threshold */
#define CONDITION_NUMBER_THRESHOLD 1e8  /**< Threshold for condition number */

/** 
 * @def DEBUG_LEVEL
 * @brief Set the desired debug level (0: No debug, 1: Critical, 2: Detailed, 3: Verbose).
 */
#define DEBUG_LEVEL 0  // Adjust as needed

// Define specific debug print macros
#if DEBUG_LEVEL >= 1
    #define DEBUG_PRINT_1(fmt, ...) debug_print(1, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT_1(fmt, ...) do {} while(0)
#endif

#if DEBUG_LEVEL >= 2
    #define DEBUG_PRINT_2(fmt, ...) debug_print(2, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT_2(fmt, ...) do {} while(0)
#endif

#if DEBUG_LEVEL >= 3
    #define DEBUG_PRINT_3(fmt, ...) debug_print(3, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT_3(fmt, ...) do {} while(0)
#endif

/**
 * @brief Prints debug messages based on the debug level.
 *
 * This function handles the actual printing of debug messages. It's separated from the macros to allow
 * for more complex debug handling in the future, such as logging to files or other outputs.
 *
 * @param level The debug level of the message.
 * @param fmt The format string.
 * @param ... The variable arguments corresponding to the format string.
 */
static void debug_print(int level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    // Optionally, prepend debug messages with level information
    // printf("[DEBUG LEVEL %d] ", level);
    vprintf(fmt, args);
    va_end(args);
}

static void recompute_qr_decomposition(RegressionState *regression_state, const double *measurements);
static inline void solve_for_coefficients(RegressionState *regression_state);
static double compute_condition_number(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size);
static void updateQR_AddRow(RegressionState *regression_state, const double *new_row, double new_b);
static void updateQR_RemoveRow(RegressionState *regression_state, const double *old_row, double old_b);
void trackFirstOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree);
void trackSecondOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree);

#ifdef USE_GLOBAL_SCRATCH_SPACE
// Maximum possible total rows (define this appropriately)
#define MAX_TOTAL_ROWS (RLS_WINDOW + MAX_POLYNOMIAL_DEGREE + 1)

// Declare the arrays as static global variables
static double augmented_matrix_A[MAX_TOTAL_ROWS][MAX_POLYNOMIAL_DEGREE + 1];
static double augmented_vector_b[MAX_TOTAL_ROWS];
#endif


/**
 * @brief Initializes the RegressionState structure.
 *
 * @param regression_state Pointer to the RegressionState structure to initialize.
 * @param degree The degree of the polynomial (e.g., 2 for quadratic, 3 for cubic).
 */
void initialize_regression_state(RegressionState *regression_state, uint8_t degree, uint16_t max_num_points) {
    DEBUG_PRINT_3("Initializing RegressionState with degree=%u\n", degree);

    regression_state->current_num_points = 0;
    regression_state->max_num_points = max_num_points;
    regression_state->oldest_data_index = 0;
    regression_state->total_data_points_added = 0;
    regression_state->reorthogonalization_counter = 0;
    regression_state->polynomial_degree = degree;

    // Initialize regression coefficients to zero
    memset(regression_state->coefficients, 0, sizeof(regression_state->coefficients));
    DEBUG_PRINT_2("Regression coefficients initialized to zero.\n");

    // Initialize the upper triangular matrix R and Q^T * b vector to zero
    memset(regression_state->upper_triangular_R, 0, sizeof(regression_state->upper_triangular_R));
    memset(regression_state->Q_transpose_b, 0, sizeof(regression_state->Q_transpose_b));
    DEBUG_PRINT_2("Upper triangular matrix R and Q^T * b initialized to zero.\n");

    // Initialize column permutations to default ordering
    for (int i = 0; i <= degree; ++i) {
        regression_state->col_permutations[i] = i;
    }
    DEBUG_PRINT_2("Column permutations initialized to default ordering.\n");
}


/**
 * @brief Adds a new data point to the regression model and updates it.
 *
 * This function adds a new measurement to the regression model. It updates the QR decomposition
 * incrementally using Givens rotations when a new data point is added. If the buffer is full,
 * it removes the oldest data point using Givens rotations for downdating. It also handles
 * reorthogonalization and condition number monitoring.
 *
 * @param regression_state Pointer to the RegressionState structure that holds the model state.
 * @param measurement The new measurement value to add.
 */
void add_data_point_to_regression(RegressionState *regression_state, const double *measurements, uint16_t data_index) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    double measurement = measurements[data_index];
    DEBUG_PRINT_3("Adding new data point: measurement=%.6f at data_index=%u\n", measurement, data_index);

    // Generate the index (independent variable) internally
    double x_value = (double)(regression_state->total_data_points_added);
    DEBUG_PRINT_2("Current x_value for new data point: %.2f\n", x_value);

    // Prepare the new row (polynomial basis vector) using incremental multiplication
    double new_row[MAX_POLYNOMIAL_DEGREE + 1];
    double x_power = 1.0;  // Start with x^0
    for (int i = 0; i < num_coefficients; ++i) {
        new_row[i] = x_power;
        x_power *= x_value;
    }
    DEBUG_PRINT_3("New polynomial basis vector computed.\n");

    // Update QR decomposition with the new data point
    updateQR_AddRow(regression_state, new_row, measurement);
    DEBUG_PRINT_3("QR decomposition updated with new data point.\n");

    // If buffer is full, remove the oldest data point
    if (regression_state->current_num_points >= regression_state->max_num_points) {
        uint16_t oldest_data_index = regression_state->oldest_data_index;
        double old_measurement = measurements[oldest_data_index];
        DEBUG_PRINT_3("Buffer full. Removing oldest data point at index %u with measurement=%.6f.\n", oldest_data_index, old_measurement);

        // Generate the old_row (polynomial basis vector for the oldest data point)
        double old_x_value = (double)(regression_state->total_data_points_added - regression_state->max_num_points);
        x_power = 1.0;
        double old_row[MAX_POLYNOMIAL_DEGREE + 1];
        for (int i = 0; i < num_coefficients; ++i) {
            old_row[i] = x_power;
            x_power *= old_x_value;
        }
        DEBUG_PRINT_3("Old polynomial basis vector computed for removal.\n");

        // Update QR decomposition to remove the oldest data point
        updateQR_RemoveRow(regression_state, old_row, old_measurement);
        DEBUG_PRINT_3("QR decomposition updated by removing oldest data point.\n");

        // Update oldest_data_index
        regression_state->oldest_data_index = oldest_data_index + 1;
        DEBUG_PRINT_3("Oldest data index updated to %u.\n", regression_state->oldest_data_index);
    } else {
        // Buffer is not yet full
        regression_state->current_num_points++;
        DEBUG_PRINT_3("Buffer not full yet. Current number of points: %u.\n", regression_state->current_num_points);
    }

    // Increment the total number of data points added
    regression_state->total_data_points_added++;
    DEBUG_PRINT_3("Total data points added incremented to %u.\n", regression_state->total_data_points_added);

    // Recompute QR decomposition if necessary
    regression_state->reorthogonalization_counter++;
    DEBUG_PRINT_3("Reorthogonalization counter incremented to %u.\n", regression_state->reorthogonalization_counter);
    if (regression_state->reorthogonalization_counter >= REORTHOGONALIZATION_INTERVAL) {
        DEBUG_PRINT_3("Reorthogonalization interval reached. Recomputing QR decomposition.\n");
        recompute_qr_decomposition(regression_state, measurements);
        regression_state->reorthogonalization_counter = 0;  // Reset counter
        DEBUG_PRINT_3("QR decomposition recomputed and counter reset.\n");
    }

    // Solve for the new regression coefficients
    solve_for_coefficients(regression_state);
    DEBUG_PRINT_2("Regression coefficients solved and updated.\n");

    // Compute condition number to check numerical stability
    double condition_number = compute_condition_number(regression_state->upper_triangular_R, num_coefficients);
    DEBUG_PRINT_2("Computed condition number: %.6e\n", condition_number);
    if (condition_number > CONDITION_NUMBER_THRESHOLD) {
        DEBUG_PRINT_1("Condition number %.6e exceeds threshold %.6e. Recomputing QR decomposition.\n", condition_number, CONDITION_NUMBER_THRESHOLD);
        recompute_qr_decomposition(regression_state, measurements);
        // Solve for the new coefficients again
        solve_for_coefficients(regression_state);
        DEBUG_PRINT_2("Regression coefficients re-solved after recomputing QR decomposition.\n");
    }
}

/**
 * @brief Updates the QR decomposition by adding a new row using Givens rotations.
 *
 * This function incrementally updates the existing QR decomposition when a new data point is added
 * to the regression model. It uses Givens rotations to maintain the upper triangular
 * structure of the R matrix without recomputing the entire decomposition.
 *
 * **Changes Introduced:**
 * - **Use of Givens Rotations:**
 *   - Previously, the entire QR decomposition was recomputed when new data was added.
 *   - Now, Givens rotations are employed to update the QR factors incrementally.
 *   - This change improves efficiency by reducing computational load and enables real-time updates.
 * - **Efficiency and Numerical Stability:**
 *   - Givens rotations are numerically stable for incremental updates, especially when dealing
 *     with streaming data or large datasets.
 *
 * **Why Givens Rotations Were Introduced:**
 * - To allow efficient incremental updates to the QR decomposition without full recomputation.
 * - To enhance numerical stability during the addition of new data points.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param new_row The new row to add (polynomial basis vector).
 * @param new_b The new measurement value.
 */
static void updateQR_AddRow(RegressionState *regression_state, const double *new_row, double new_b) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Updating QR decomposition by adding a new row.\n");

    // Copy the new row into a temporary array
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    memcpy(r_row, new_row, sizeof(double) * num_coefficients);
    DEBUG_PRINT_2("New row copied for QR update.\n");

    double Q_tb_new = new_b;
    DEBUG_PRINT_2("Initial Q^T * b for new row: %.6f\n", Q_tb_new);

    // Apply Givens rotations to zero out sub-diagonal elements
    for (int i = 0; i < num_coefficients; ++i) {
        double a = regression_state->upper_triangular_R[i][i];
        double b = r_row[i];

        if (fabs(b) > 1e-10) {  // Avoid division by zero
            // Compute Givens rotation parameters
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;

            DEBUG_PRINT_2("Applying Givens rotation at column %d: c=%.6f, s=%.6f\n", i, c, s);

            // Update R
            regression_state->upper_triangular_R[i][i] = r;
            for (int j = i + 1; j < num_coefficients; ++j) {
                double temp = regression_state->upper_triangular_R[i][j];
                regression_state->upper_triangular_R[i][j] = c * temp + s * r_row[j];
                r_row[j] = -s * temp + c * r_row[j];
                DEBUG_PRINT_3("Updated R[%d][%d]=%.6f and r_row[%d]=%.6f\n", i, j, regression_state->upper_triangular_R[i][j], j, r_row[j]);
            }

            // Update Q^T * b
            double temp_b = regression_state->Q_transpose_b[i];
            regression_state->Q_transpose_b[i] = c * temp_b + s * Q_tb_new;
            Q_tb_new = -s * temp_b + c * Q_tb_new;
            DEBUG_PRINT_3("Updated Q_transpose_b[%d]=%.6f and Q_tb_new=%.6f\n", i, regression_state->Q_transpose_b[i], Q_tb_new);
        } else {
            // No rotation needed; R[i][i] remains the same
            DEBUG_PRINT_3("No rotation needed for column %d.\n", i);
            // Update Q^T * b with the new data
            regression_state->Q_transpose_b[i] += regression_state->upper_triangular_R[i][i] * 0.0;
        }
    }

    DEBUG_PRINT_2("QR decomposition updated with new row.\n");
}

/**
 * @brief Updates the QR decomposition by removing an old row using Givens rotations.
 *
 * This function incrementally downdates the existing QR decomposition when an old data point
 * is removed from the regression model. It uses Givens rotations to adjust the R matrix,
 * maintaining its upper triangular structure.
 *
 * **Changes Introduced:**
 * - **Use of Givens Rotations for Downdating:**
 *   - Previously, the QR decomposition was recomputed entirely when old data was removed.
 *   - Now, Givens rotations are used to efficiently remove the influence of the oldest data point.
 * - **Efficiency Improvement:**
 *   - This change reduces computational overhead, making the algorithm suitable for real-time applications.
 * - **Numerical Stability:**
 *   - Givens rotations provide a stable method for downdating, minimizing numerical errors.
 *
 * **Why Givens Rotations Were Introduced:**
 * - To enable efficient and stable downdating of the QR decomposition when data points are removed.
 * - To maintain the numerical integrity of the regression model over time.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param old_row The old row to remove (polynomial basis vector).
 * @param old_b The old measurement value.
 */
static void updateQR_RemoveRow(RegressionState *regression_state, const double *old_row, double old_b) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Updating QR decomposition by removing an old row.\n");

    // Subtract the influence of the old data point
    double r_row[MAX_POLYNOMIAL_DEGREE + 1];
    memcpy(r_row, old_row, sizeof(double) * num_coefficients);
    DEBUG_PRINT_2("Old row copied for QR downdate.\n");

    double Q_tb_old = old_b;
    DEBUG_PRINT_2("Initial Q^T * b for old row: %.6f\n", Q_tb_old);

    // Apply reverse Givens rotations to remove the old data point
    for (int i = 0; i < num_coefficients; ++i) {
        double a = regression_state->upper_triangular_R[i][i];
        double b = r_row[i];

        if (fabs(b) > 1e-10) {  // Avoid division by zero
            // Compute Givens rotation parameters
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;

            DEBUG_PRINT_2("Applying Givens rotation at column %d: c=%.6f, s=%.6f\n", i, c, s);

            // Update R
            regression_state->upper_triangular_R[i][i] = r;
            for (int j = i + 1; j < num_coefficients; ++j) {
                double temp = regression_state->upper_triangular_R[i][j];
                regression_state->upper_triangular_R[i][j] = c * temp - s * r_row[j];
                r_row[j] = s * temp + c * r_row[j];
                DEBUG_PRINT_3("Updated R[%d][%d]=%.6f and r_row[%d]=%.6f\n", i, j, regression_state->upper_triangular_R[i][j], j, r_row[j]);
            }

            // Update Q^T * b
            double temp_b = regression_state->Q_transpose_b[i];
            regression_state->Q_transpose_b[i] = c * temp_b - s * Q_tb_old;
            Q_tb_old = s * temp_b + c * Q_tb_old;
            DEBUG_PRINT_3("Updated Q_transpose_b[%d]=%.6f and Q_tb_old=%.6f\n", i, regression_state->Q_transpose_b[i], Q_tb_old);
        } else {
            // No rotation needed
            DEBUG_PRINT_3("No rotation needed for column %d.\n", i);
            // Update Q^T * b
            // regression_state->Q_transpose_b[i] -= regression_state->upper_triangular_R[i][i] * 0.0;
        }
    }

    DEBUG_PRINT_2("QR decomposition updated by removing old row.\n");
}

/**
 * @brief Recomputes the QR decomposition using optimized Householder reflections with column pivoting.
 *
 * This function performs a full recomputation of the QR decomposition using Householder reflections
 * with column pivoting for enhanced numerical stability. Column pivoting rearranges the columns
 * based on their norms to position the largest possible pivot elements on the diagonal, reducing
 * potential numerical issues.
 *
 * **Changes Introduced:**
 * - **Access to External Measurements:**
 *   - Measurements are accessed directly from the external buffer `measurements`.
 *   - The function signature now includes `const double *measurements`.
 * - **Column Pivoting:**
 *   - Implements column pivoting by tracking and swapping columns based on their norms.
 *   - Updates the column indices array to keep track of permutations.
 * - **Memory Optimization:**
 *   - Uses global scratch space for large arrays to minimize stack usage.
 *
 * **Why These Changes Were Introduced:**
 * - To eliminate redundant storage of measurements within `RegressionState`.
 * - To enhance numerical stability during QR decomposition.
 * - To minimize stack usage by avoiding large local arrays.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param measurements External buffer containing measurement data.
 */
static void recompute_qr_decomposition(RegressionState *regression_state, const double *measurements) {
    int num_data_points = regression_state->current_num_points;
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Recomputing QR decomposition with column pivoting for RegressionState.\n");

    // Regularization parameter to prevent overfitting
    double regularization_lambda = REGULARIZATION_PARAMETER;
    DEBUG_PRINT_2("Regularization parameter lambda=%.6f\n", regularization_lambda);

    // Total number of rows after adding regularization terms
    int total_rows = num_data_points + num_coefficients;

    // Maximum possible total rows
    int max_total_rows = regression_state->max_num_points + num_coefficients;

    // Use the global scratch space for large arrays
#ifdef USE_GLOBAL_SCRATCH_SPACE
    // Ensure that total_rows does not exceed MAX_TOTAL_ROWS
    if (total_rows > MAX_TOTAL_ROWS) {
        DEBUG_PRINT_1("Error: total_rows (%d) exceeds MAX_TOTAL_ROWS (%d).\n", total_rows, MAX_TOTAL_ROWS);
        return;
    }
#else
    // Declare the arrays as local variables (on the stack)
    double augmented_matrix_A[total_rows][MAX_POLYNOMIAL_DEGREE + 1];
    double augmented_vector_b[total_rows];
#endif

    // Initialize arrays to zero
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
    DEBUG_PRINT_2("Augmented matrices initialized to zero.\n");

    // Initialize column indices for pivoting
    int col_indices[MAX_POLYNOMIAL_DEGREE + 1];
    for (int i = 0; i < num_coefficients; ++i) {
        col_indices[i] = i;
    }

    // Populate the augmented matrix with original data
    for (int i = 0; i < num_data_points; ++i) {
        uint16_t data_index = regression_state->oldest_data_index + i;
        double x = (double)(regression_state->total_data_points_added - num_data_points + i);
        double measurement = measurements[data_index];
        DEBUG_PRINT_3("Processing data point %d: x=%.2f, y=%.6f\n", i, x, measurement);

        // Construct the design matrix A with polynomial terms using incremental multiplication
        double x_power = 1.0;  // Start with x^0
        for (int j = 0; j < num_coefficients; ++j) {
            augmented_matrix_A[i][j] = x_power;
            x_power *= x;  // Multiply by x to get the next power
        }

        // Measurement vector b (dependent variable)
        augmented_vector_b[i] = measurement;
    }
    DEBUG_PRINT_2("Augmented matrix populated with original data.\n");

    // Add regularization rows to the augmented matrix and vector
    for (int i = 0; i < num_coefficients; ++i) {
        int row = num_data_points + i;
        augmented_matrix_A[row][i] = sqrt(regularization_lambda);
        augmented_vector_b[row] = 0.0;
        DEBUG_PRINT_3("Added regularization row %d: A[%d][%d]=%.6f, b[%d]=%.6f\n", i, row, i, augmented_matrix_A[row][i], row, augmented_vector_b[row]);
    }
    DEBUG_PRINT_2("Regularization rows added to augmented matrix.\n");

    // Initialize R and Q^T * b in the regression state
    memset(regression_state->upper_triangular_R, 0, sizeof(regression_state->upper_triangular_R));
    memset(regression_state->Q_transpose_b, 0, sizeof(regression_state->Q_transpose_b));
    DEBUG_PRINT_2("Upper triangular matrix R and Q^T * b reset to zero.\n");

    // Initialize column norms for pivoting
    double col_norms[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coefficients; ++j) {
        double sum = 0.0;
        for (int i = 0; i < total_rows; ++i) {
            sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
        }
        col_norms[j] = sqrt(sum);
    }

    // Perform Householder QR Decomposition with column pivoting
    for (int k = 0; k < num_coefficients; ++k) {
        // Find the column with the maximum norm
        int max_col = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < num_coefficients; ++j) {
            if (col_norms[j] > max_norm) {
                max_norm = col_norms[j];
                max_col = j;
            }
        }

        // Swap columns in A, col_indices, and col_norms if necessary
        if (max_col != k) {
            // Swap columns in A
            for (int i = 0; i < total_rows; ++i) {
                double temp = augmented_matrix_A[i][k];
                augmented_matrix_A[i][k] = augmented_matrix_A[i][max_col];
                augmented_matrix_A[i][max_col] = temp;
            }

            // Swap entries in col_indices
            int temp_idx = col_indices[k];
            col_indices[k] = col_indices[max_col];
            col_indices[max_col] = temp_idx;

            // Swap norms
            double temp_norm = col_norms[k];
            col_norms[k] = col_norms[max_col];
            col_norms[max_col] = temp_norm;

            DEBUG_PRINT_2("Swapped columns %d and %d for pivoting.\n", k, max_col);
        }

        // Compute the norm of the k-th column below the diagonal
        double sigma = 0.0;
        for (int i = k; i < total_rows; ++i) {
            double val = augmented_matrix_A[i][k];
            sigma += val * val;
        }
        sigma = sqrt(sigma);

        DEBUG_PRINT_3("Householder transformation for column %d: sigma=%.6f\n", k, sigma);

        if (sigma == 0.0) {
            DEBUG_PRINT_3("Sigma is zero for column %d. Skipping transformation.\n", k);
            continue;  // Skip if the column norm is zero
        }

        // Compute Householder vector
        double vk = augmented_matrix_A[k][k] + ((augmented_matrix_A[k][k] >= 0) ? sigma : -sigma);

        // Avoid division by zero
        if (fabs(vk) < 1e-12) {
            DEBUG_PRINT_3("vk too small for column %d. Skipping transformation.\n", k);
            continue;
        }

        double beta = 1.0 / (sigma * vk);

        // Apply the Householder transformation to A
        for (int i = k + 1; i < total_rows; ++i) {
            augmented_matrix_A[i][k] /= vk;
        }
        augmented_matrix_A[k][k] = -sigma;

        // Apply transformation to the remaining columns
        for (int j = k + 1; j < num_coefficients; ++j) {
            double s = 0.0;
            for (int i = k; i < total_rows; ++i) {
                s += augmented_matrix_A[i][k] * augmented_matrix_A[i][j];
            }
            s *= beta;

            for (int i = k; i < total_rows; ++i) {
                augmented_matrix_A[i][j] -= augmented_matrix_A[i][k] * s;
            }
            DEBUG_PRINT_3("Applied Householder transformation to column %d: s=%.6f\n", j, s);
        }

        // Apply transformation to the vector b
        double s = 0.0;
        for (int i = k; i < total_rows; ++i) {
            s += augmented_matrix_A[i][k] * augmented_vector_b[i];
        }
        s *= beta;

        for (int i = k; i < total_rows; ++i) {
            augmented_vector_b[i] -= augmented_matrix_A[i][k] * s;
        }
        DEBUG_PRINT_3("Applied Householder transformation to vector b: s=%.6f\n", s);

        // Zero out below-diagonal elements
        for (int i = k + 1; i < total_rows; ++i) {
            augmented_matrix_A[i][k] = 0.0;
        }
        DEBUG_PRINT_3("Zeroed out below-diagonal elements for column %d.\n", k);

        // Update the norms of the remaining columns
        for (int j = k + 1; j < num_coefficients; ++j) {
            double sum = 0.0;
            for (int i = k + 1; i < total_rows; ++i) {
                sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
            }
            col_norms[j] = sqrt(sum);
        }
    }

    // Extract R and Q^T * b with column permutations
    for (int i = 0; i < num_coefficients; ++i) {
        for (int j = i; j < num_coefficients; ++j) {
            regression_state->upper_triangular_R[i][j] = augmented_matrix_A[i][j];
        }
        regression_state->Q_transpose_b[i] = augmented_vector_b[i];
    }

    // Store the column permutation indices in the regression state
    memcpy(regression_state->col_permutations, col_indices, sizeof(int) * num_coefficients);
    DEBUG_PRINT_3("QR decomposition recomputed with column pivoting.\n");
}

/**
 * @brief Solves the upper triangular system to find the regression coefficients with column permutations.
 *
 * This function performs back substitution on the upper triangular matrix R to solve for
 * the regression coefficients, taking into account any column permutations due to pivoting.
 * It ensures that the latest coefficients are used for gradient calculations and predictions.
 *
 * **Changes Introduced:**
 * - **Handling Column Permutations:**
 *   - Adjusts the back substitution process to account for the column permutations.
 *   - Rearranges the coefficients to match the original variable ordering.
 *
 * **Why These Changes Were Introduced:**
 * - To correctly solve for coefficients when column pivoting is used.
 * - To ensure that the coefficients correspond to the correct polynomial terms.
 *
 * @param regression_state Pointer to the RegressionState structure.
 */
static inline void solve_for_coefficients(RegressionState *regression_state) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Solving for regression coefficients using back substitution with column permutations.\n");

    double temp_coefficients[MAX_POLYNOMIAL_DEGREE + 1];

    // Perform back substitution to solve R * x = Q^T * b
    for (int i = num_coefficients - 1; i >= 0; --i) {
        double sum = regression_state->Q_transpose_b[i];
        for (int j = i + 1; j < num_coefficients; ++j) {
            sum -= regression_state->upper_triangular_R[i][j] * temp_coefficients[j];
        }

        if (fabs(regression_state->upper_triangular_R[i][i]) < 1e-12) {
            DEBUG_PRINT_1("Warning: Small diagonal element R[%d][%d]=%.6e during back substitution.\n", i, i, regression_state->upper_triangular_R[i][i]);
            temp_coefficients[i] = 0.0;  // Assign zero to avoid division by zero
        } else {
            temp_coefficients[i] = sum / regression_state->upper_triangular_R[i][i];
            DEBUG_PRINT_3("Temporary Coefficient[%d] solved: %.6f\n", i, temp_coefficients[i]);
        }
    }

    // Rearrange the solution according to the original column order
    for (int i = 0; i < num_coefficients; ++i) {
        int permuted_index = regression_state->col_permutations[i];
        regression_state->coefficients[permuted_index] = temp_coefficients[i];
        DEBUG_PRINT_3("Coefficient[%d] set to %.6f (permuted from position %d)\n", permuted_index, temp_coefficients[i], i);
    }

    DEBUG_PRINT_2("Regression coefficients updated with column permutations.\n");
}


#ifdef USE_IMPROVED_CONDITION_NUMBER
/**
 * @brief Computes the 1-norm of the upper triangular matrix R.
 *
 * This function calculates the maximum absolute column sum of the upper triangular matrix R,
 * which is useful for estimating the condition number of R.
 *
 * **Why This Function Was Introduced:**
 * - To provide a more accurate estimation of the matrix norm, which is essential for computing the condition number.
 * - Utilizing the 1-norm leverages the structure of the upper triangular matrix for efficient computation.
 *
 * @param R The upper triangular matrix R.
 * @param size The size of the matrix (number of coefficients).
 * @return The 1-norm of matrix R.
 */
static double compute_R_norm_1(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size) {
    DEBUG_PRINT_3("Computing the 1-norm of matrix R.\n");
    double norm = 0.0;
    for (int j = 0; j < size; ++j) {
        double sum = 0.0;
        for (int i = 0; i <= j; ++i) {  // Only upper triangular part
            sum += fabs(R[i][j]);
            DEBUG_PRINT_3("Adding |R[%d][%d]|=%.6f to column sum.\n", i, j, fabs(R[i][j]));
        }
        DEBUG_PRINT_3("Column %d sum: %.6f\n", j, sum);
        if (sum > norm) {
            norm = sum;
            DEBUG_PRINT_3("Updated norm to %.6f\n", norm);
        }
    }
    DEBUG_PRINT_2("Computed 1-norm of R: %.6f\n", norm);
    return norm;
}

/**
 * @brief Estimates the 1-norm of the inverse of the upper triangular matrix R.
 *
 * This function estimates the maximum absolute column sum of the inverse of R without explicitly computing R^{-1}.
 * It uses an iterative method suitable for upper triangular matrices.
 *
 * **Why This Function Was Introduced:**
 * - To enable accurate estimation of the condition number by computing \|R^{-1}\|_1 efficiently.
 * - Avoids explicit inversion of R, which can be computationally expensive and numerically unstable.
 *
 * @param R The upper triangular matrix R.
 * @param size The size of the matrix (number of coefficients).
 * @return The estimated 1-norm of R^{-1}.
 */
static double estimate_R_inverse_norm_1(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size) {
    DEBUG_PRINT_3("Estimating the 1-norm of the inverse of matrix R.\n");
    double x[MAX_POLYNOMIAL_DEGREE + 1];
    double z[MAX_POLYNOMIAL_DEGREE + 1];
    double est = 0.0;
    
    // Initialize vectors
    for (int i = 0; i < size; ++i) {
        x[i] = 1.0 / (double)size;
        DEBUG_PRINT_3("Initialized x[%d]=%.6f\n", i, x[i]);
    }
    
    int iter = 0;
    double prev_est = 0.0;
    const int max_iter = 5;
    const double tol = 1e-6;
    
    do {
        // Solve R * z = x using back substitution
        for (int i = size - 1; i >= 0; --i) {
            double sum = x[i];
            for (int j = i + 1; j < size; ++j) {
                sum -= R[i][j] * z[j];
                DEBUG_PRINT_3("Subtracting R[%d][%d]*z[%d]=%.6f*%.6f from sum.\n", i, j, j, R[i][j], z[j]);
            }
            z[i] = sum / R[i][i];
            DEBUG_PRINT_3("Computed z[%d]=%.6f\n", i, z[i]);
        }
        
        // Compute the 1-norm of z
        est = 0.0;
        for (int i = 0; i < size; ++i) {
            est += fabs(z[i]);
            DEBUG_PRINT_3("Adding |z[%d]|=%.6f to est.\n", i, fabs(z[i]));
        }
        DEBUG_PRINT_2("Current estimate of norm_R_inv: %.6f\n", est);
        
        // Normalize z
        double z_norm = est;
        for (int i = 0; i < size; ++i) {
            z[i] /= z_norm;
            DEBUG_PRINT_3("Normalized z[%d]=%.6f\n", i, z[i]);
        }
        
        // Check for convergence
        if (fabs(est - prev_est) < tol * est) {
            DEBUG_PRINT_2("Convergence achieved after %d iterations.\n", iter);
            break;
        }
        
        // Prepare for next iteration
        memcpy(x, z, sizeof(double) * size);
        prev_est = est;
        iter++;
        DEBUG_PRINT_3("Iteration %d completed. est=%.6f, prev_est=%.6f\n", iter, est, prev_est);
    } while (iter < max_iter);
    
    DEBUG_PRINT_2("Estimated 1-norm of R^{-1}: %.6f after %d iterations.\n", est, iter);
    return est;
}

/**
 * @brief Computes the condition number of the upper triangular matrix R using the 1-norm.
 *
 * This function computes the condition number κ(R) = \|R\|₁ * \|R⁻¹\|₁, providing
 * a more accurate estimation compared to using only the diagonal elements.
 *
 * **Changes Introduced:**
 * - Uses matrix norms to estimate the condition number more accurately.
 * - Introduces efficient estimation of \|R⁻¹\|₁ without explicit inversion.
 *
 * **Why These Changes Were Introduced:**
 * - To improve the reliability of the condition number estimation.
 * - Enhances the ability to detect numerical instability in the regression model.
 *
 * @param R The upper triangular matrix R.
 * @param size The size of the matrix (number of coefficients).
 * @return The estimated condition number.
 */
static double compute_condition_number(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size) {
    DEBUG_PRINT_3("Computing condition number of matrix R using 1-norm.\n");
    double norm_R = compute_R_norm_1(R, size);
    double norm_R_inv = estimate_R_inverse_norm_1(R, size);
    double condition_number = norm_R * norm_R_inv;
    DEBUG_PRINT_2("Computed condition number: %.6e (norm_R=%.6e, norm_R_inv=%.6e)\n", condition_number, norm_R, norm_R_inv);
    return condition_number;
}
#else  // Original method
/**
 * @brief Computes the condition number of the upper triangular matrix R using the 2-norm estimation.
 *
 * @param R The upper triangular matrix R.
 * @param size The size of the matrix (number of coefficients).
 * @return The estimated condition number.
 */
static double compute_condition_number(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size) {
    DEBUG_PRINT_3("Computing condition number of matrix R.\n");

    // Estimate the condition number using the ratio of norms
    // Since R is upper triangular, we can estimate the condition number by the ratio of largest to smallest singular values
    // Use the absolute values of diagonal elements as approximations
    double max_sv = 0.0;
    double min_sv = DBL_MAX;

    for (int i = 0; i < size; ++i) {
        double diag = fabs(R[i][i]);
        if (diag > max_sv) {
            max_sv = diag;
        }
        if (diag < min_sv && diag > 1e-12) {
            min_sv = diag;
        }
    }

    DEBUG_PRINT_3("Maximum singular value estimate: %.6e, Minimum singular value estimate: %.6e\n", max_sv, min_sv);

    if (min_sv < 1e-12) {
        // Matrix is singular or nearly singular
        DEBUG_PRINT_1("Condition number is infinite (matrix is singular or nearly singular).\n");
        return DBL_MAX;
    }

    double condition_number = max_sv / min_sv;
    DEBUG_PRINT_3("Computed condition number: %.6e\n", condition_number);

    return condition_number;
}
#endif  // USE_IMPROVED_CONDITION_NUMBER

/**
 * @brief Calculates the first-order gradient (slope) of the polynomial function at a specific point.
 *
 * @param regression_state Pointer to the RegressionState structure containing the coefficients.
 * @param x The point at which to calculate the first-order gradient.
 * @return The first-order gradient at the given point x.
 */
double calculate_first_order_gradient(const RegressionState *regression_state, double x) {
    DEBUG_PRINT_3("Calculating first-order gradient at x=%.6f\n", x);

    double derivative = 0.0;
    double x_power = 1.0;
    int degree = regression_state->polynomial_degree;

    for (int i = 1; i <= degree; ++i) {
        x_power *= x;
        derivative += i * regression_state->coefficients[i] * x_power / x;
        DEBUG_PRINT_3("Term %d: i=%d, coefficient=%.6f, x_power=%.6f, derivative=%.6f\n", i, i, regression_state->coefficients[i], x_power, derivative);
    }

    DEBUG_PRINT_3("First-order gradient calculated: %.6f\n", derivative);
    return derivative;
}

/**
 * @brief Calculates the second-order gradient (curvature) of the polynomial function at a specific point.
 *
 * @param regression_state Pointer to the RegressionState structure containing the coefficients.
 * @param x The point at which to calculate the second-order gradient.
 * @return The second-order gradient at the given point x.
 */
double calculate_second_order_gradient(const RegressionState *regression_state, double x) {
    DEBUG_PRINT_3("Calculating second-order gradient at x=%.6f\n", x);

    double second_derivative = 0.0;
    double x_power = 1.0;
    int degree = regression_state->polynomial_degree;

    for (int i = 2; i <= degree; ++i) {
        x_power *= x;
        second_derivative += i * (i - 1) * regression_state->coefficients[i] * x_power / (x * x);
        DEBUG_PRINT_3("Term %d: i=%d, coefficient=%.6f, x_power=%.6f, second_derivative=%.6f\n", i, i, regression_state->coefficients[i], x_power, second_derivative);
    }

    DEBUG_PRINT_3("Second-order gradient calculated: %.6f\n", second_derivative);
    return second_derivative;
}


/**
 * @brief Tracks the values added to the RLS array and calculates gradients after each addition.
 *
 * This generalized function calculates gradients (first-order, second-order, etc.) for a given dataset
 * using Recursive Least Squares (RLS) polynomial regression. It accepts a function pointer to determine
 * the type of gradient to calculate, making it flexible for various gradient computations.
 *
 * @param measurements Array of measurement values to add to the RLS system.
 * @param length The number of points to add starting from the given start index.
 * @param start_index The index in the measurements array from which to begin adding values.
 * @param degree The degree of the polynomial to use for regression (e.g., 2 for quadratic, 3 for cubic).
 * @param calculate_gradient Function pointer to the gradient calculation function (e.g., first-order, second-order).
 * @param result Pointer to store the gradient calculation results.
 */
void trackGradients(
    const double *measurements,
    uint16_t length,
    uint16_t start_index,
    uint8_t degree,
    double (*calculate_gradient)(const RegressionState *, double),
    GradientCalculationResult *result
) {
    DEBUG_PRINT_3("Entering trackGradients with startIndex=%u, length=%u, degree=%u\n", start_index, length, degree);

    // Initialize the result struct
    result->size = 0;
    result->valid = false;
    DEBUG_PRINT_2("Initialized GradientCalculationResult: size=0, valid=false\n");

    // Initialize the regression state
    RegressionState regression_state;
    initialize_regression_state(&regression_state, degree, RLS_WINDOW);
    DEBUG_PRINT_3("RegressionState initialized with degree=%u\n", degree);

    // Loop through the measurements and calculate gradients
    for (uint16_t i = 0; i < length; ++i) {
        uint16_t current_index = start_index + i;

        // Add the current measurement to the regression
        add_data_point_to_regression(&regression_state, measurements, current_index);
        DEBUG_PRINT_3("Added measurement %.6f to RegressionState\n", measurements[current_index]);

        // Get the x_value corresponding to the current data point
        double x_value = (double)(regression_state.total_data_points_added - 1);
        DEBUG_PRINT_2("Current x_value: %.2f (total_data_points_added=%u)\n", x_value, regression_state.total_data_points_added);

        // Check if there are enough points to perform regression and calculate gradient
        if (regression_state.current_num_points >= regression_state.polynomial_degree + 1) {
            // Calculate the gradient using the provided function pointer
            double gradient = calculate_gradient(&regression_state, x_value);
            DEBUG_PRINT_1("Calculated gradient: %.6f at x=%.2f\n", gradient, x_value);
            printf("Calculated gradient: %.6f at x=%.2f\n", gradient, x_value);


            // Store the gradient in the result struct if there's space
            if (result->size < RLS_WINDOW) {
                result->gradients[result->size++] = gradient;
                DEBUG_PRINT_2("Stored gradient in result->gradients[%u]\n", result->size - 1);
            } else {
                // Handle overflow: log a warning and stop collecting gradients
                DEBUG_PRINT_1("Gradient array overflow at index %u. Maximum window size reached.\n", result->size);
                break;
            }
        } else {
            // Not enough points to calculate gradient
            DEBUG_PRINT_1("Insufficient points to calculate gradient (current_num_points=%u)\n", regression_state.current_num_points);
        }
    }

    // Post-processing after collecting gradients
    if (result->size > 0) {
        DEBUG_PRINT_2("Collected %u gradients\n", result->size);
    } else {
        DEBUG_PRINT_2("No gradients were collected\n");
    }

    // Note: Median and MAD calculations are handled in the calling function (e.g., identifyTrends)
    DEBUG_PRINT_3("Exiting trackGradients\n");
}

