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

static void recompute_qr_decomposition(RegressionState *regression_state);
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
void initialize_regression_state(RegressionState *regression_state, uint8_t degree) {
    DEBUG_PRINT_3("Initializing RegressionState with degree=%u\n", degree);
    
    regression_state->current_num_points = 0;
    regression_state->max_num_points = RLS_WINDOW;
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

    // Reset the measurement buffer within the structure
    memset(regression_state->measurement_buffer, 0, sizeof(regression_state->measurement_buffer));
    DEBUG_PRINT_2("Measurement buffer reset.\n");
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
void add_data_point_to_regression(RegressionState *regression_state, double measurement) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Adding new data point: measurement=%.6f\n", measurement);

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

    // Add the current measurement to the buffer
    regression_state->measurement_buffer[regression_state->total_data_points_added % regression_state->max_num_points] = measurement;
    DEBUG_PRINT_2("New measurement added to buffer at index %u.\n", regression_state->total_data_points_added % regression_state->max_num_points);

    // If buffer is full, remove the oldest data point
    if (regression_state->current_num_points >= regression_state->max_num_points) {
        uint16_t oldest_index = regression_state->oldest_data_index;
        double old_measurement = regression_state->measurement_buffer[oldest_index];
        DEBUG_PRINT_3("Buffer full. Removing oldest data point at index %u with measurement=%.6f.\n", oldest_index, old_measurement);

        // Generate the old_row (polynomial basis vector for the oldest data point)
        double old_x_value = x_value - regression_state->max_num_points;
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
        regression_state->oldest_data_index = (regression_state->oldest_data_index + 1) % regression_state->max_num_points;
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
        recompute_qr_decomposition(regression_state);
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
        recompute_qr_decomposition(regression_state);
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
 * @brief Recomputes the QR decomposition using optimized Householder reflections.
 *
 * This function performs a full recomputation of the QR decomposition using Householder reflections
 * when the condition number exceeds a certain threshold or periodically for numerical stability.
 * The function has been optimized to reduce computational overhead and improve numerical stability.
 *
 * **Changes Introduced:**
 * - **Optimized Householder Transformations:**
 *   - The implementation now uses in-place transformations and avoids unnecessary computations.
 *   - Loop bounds are set precisely to skip zero elements, enhancing efficiency.
 * - **Polynomial Basis Computation Optimization:**
 *   - Polynomial terms are calculated using incremental multiplication instead of `pow()`.
 * - **Regularization Integration:**
 *   - Regularization terms are added to prevent overfitting and improve numerical stability.
 *
 * **Why These Changes Were Introduced:**
 * - To reduce computational complexity during full recomputation.
 * - To improve numerical stability by minimizing rounding errors.
 * - To ensure that the QR decomposition remains accurate over time.
 *
 * @param regression_state Pointer to the RegressionState structure.
 */
static void recompute_qr_decomposition(RegressionState *regression_state) {
    int num_data_points = regression_state->current_num_points;
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Recomputing QR decomposition for RegressionState.\n");

    // Regularization parameter to prevent overfitting
    double regularization_lambda = REGULARIZATION_PARAMETER;
    DEBUG_PRINT_2("Regularization parameter lambda=%.6f\n", regularization_lambda);

    // Total number of rows after adding regularization terms
    int total_rows = num_data_points + num_coefficients;

    // Maximum possible total rows
    int max_total_rows = regression_state->max_num_points + num_coefficients;

#ifdef USE_GLOBAL_SCRATCH_SPACE
    // Use the global arrays
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
#ifdef USE_GLOBAL_SCRATCH_SPACE
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#else
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);
#endif
    DEBUG_PRINT_2("Augmented matrices initialized to zero.\n");

    // Populate the augmented matrix with original data
    for (int i = 0; i < num_data_points; ++i) {
        int idx = (regression_state->oldest_data_index + i) % regression_state->max_num_points;
        double x = (double)(regression_state->total_data_points_added - num_data_points + i);
        DEBUG_PRINT_3("Processing data point %d: x=%.2f, y=%.6f\n", i, x, regression_state->measurement_buffer[idx]);

        // Construct the design matrix A with polynomial terms using incremental multiplication
        double x_power = 1.0;  // Start with x^0
        for (int j = 0; j < num_coefficients; ++j) {
            augmented_matrix_A[i][j] = x_power;
            x_power *= x;  // Multiply by x to get the next power
        }

        // Measurement vector b (dependent variable)
        augmented_vector_b[i] = regression_state->measurement_buffer[idx];
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

    // Perform Householder QR Decomposition with optimizations
    for (int k = 0; k < num_coefficients; ++k) {
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
        DEBUG_PRINT_3("Householder vector component vk=%.6f for column %d\n", vk, k);

        // Avoid division by zero
        if (fabs(vk) < 1e-12) {
            DEBUG_PRINT_3("vk too small for column %d. Skipping transformation.\n", k);
            continue;
        }

        double beta = 1.0 / (sigma * vk);
        DEBUG_PRINT_3("Householder transformation parameters: beta=%.6f\n", beta);

        // Update the k-th element of the Householder vector
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
    }

    // Extract R and Q^T * b
    for (int i = 0; i < num_coefficients; ++i) {
        for (int j = i; j < num_coefficients; ++j) {
            regression_state->upper_triangular_R[i][j] = augmented_matrix_A[i][j];
        }
        regression_state->Q_transpose_b[i] = augmented_vector_b[i];
        //DEBUG_PRINT_3("Extracted R[%d][%d]=%.6f and Q_transpose_b[%d]=%.6f\n", i, i, regression_state->upper_triangular_R[i][j], i, regression_state->Q_transpose_b[i]);
    }
    DEBUG_PRINT_3("QR decomposition recomputed and extracted.\n");
}

/**
 * @brief Solves the upper triangular system to find the regression coefficients.
 *
 * This function performs back substitution on the upper triangular matrix R to solve for
 * the regression coefficients. It ensures that the latest coefficients are used for gradient
 * calculations and predictions.
 *
 * **Changes Introduced:**
 * - **Immediate Coefficient Update:**
 *   - Ensures that the regression coefficients are updated immediately after QR updates.
 * - **Numerical Stability Enhancements:**
 *   - Safeguards added to handle small diagonal elements and prevent division by zero.
 *
 * **Why These Changes Were Introduced:**
 * - To maintain accurate and up-to-date coefficients for gradient calculations.
 * - To enhance numerical stability during the back substitution process.
 *
 * @param regression_state Pointer to the RegressionState structure.
 */
static inline void solve_for_coefficients(RegressionState *regression_state) {
    int num_coefficients = regression_state->polynomial_degree + 1;
    DEBUG_PRINT_3("Solving for regression coefficients using back substitution.\n");

    // Perform back substitution to solve R * coefficients = Q^T * b
    for (int i = num_coefficients - 1; i >= 0; --i) {
        double sum = regression_state->Q_transpose_b[i];
        for (int j = i + 1; j < num_coefficients; ++j) {
            sum -= regression_state->upper_triangular_R[i][j] * regression_state->coefficients[j];
        }

        if (fabs(regression_state->upper_triangular_R[i][i]) < 1e-12) {
            DEBUG_PRINT_1("Warning: Small diagonal element R[%d][%d]=%.6e during back substitution.\n", i, i, regression_state->upper_triangular_R[i][i]);
            regression_state->coefficients[i] = 0.0;  // Assign zero to avoid division by zero
        } else {
            regression_state->coefficients[i] = sum / regression_state->upper_triangular_R[i][i];
            DEBUG_PRINT_3("Coefficient[%d] solved: %.6f\n", i, regression_state->coefficients[i]);
        }
    }

    DEBUG_PRINT_2("Regression coefficients updated.\n");
}

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
 * @brief Tracks the values added to the RLS array and calculates first-order gradients after each addition.
 *
 * @param measurements Array of measurement values to add to the RLS system.
 * @param length The number of points to add starting from the given start index.
 * @param start_index The index in the measurements array from which to begin adding values.
 * @param degree The degree of the polynomial to use for regression (2 for quadratic, 3 for cubic, etc.).
 */
void trackFirstOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree) {
    DEBUG_PRINT_3("Tracking first-order gradients.\n");
    // Initialize the regression state with the specified degree
    RegressionState regression_state;
    initialize_regression_state(&regression_state, degree);
    DEBUG_PRINT_2("Regression state initialized for first-order gradients.\n");

    // Loop through the measurements starting from start_index and add them to RLS
    for (uint16_t i = 0; i < length; ++i) {
        uint16_t current_index = start_index + i;

        // Add the current measurement to the regression
        add_data_point_to_regression(&regression_state, measurements[current_index]);
        DEBUG_PRINT_2("Added measurement %.6f at index %u to regression.\n", measurements[current_index], current_index);

        // Get the index of the most recently added data point
        double x_value = (double)(regression_state.total_data_points_added - 1);
        DEBUG_PRINT_2("Current x_value: %.2f\n", x_value);

        // Get the regression coefficients
        DEBUG_PRINT_3("Current regression coefficients:\n");
        for (int c = 0; c < regression_state.polynomial_degree + 1; ++c) {
            DEBUG_PRINT_3("Coefficient[%d] = %.6f\n", c, regression_state.coefficients[c]);
        }

        // Print the added x and y values
        DEBUG_PRINT_1("Added to RLS: x = %.2f, y = %.6f\n", x_value, measurements[current_index]);

        // If we have enough points, calculate the first-order gradient
        if (regression_state.current_num_points >= regression_state.polynomial_degree + 1) {
            double first_order_gradient = calculate_first_order_gradient(&regression_state, x_value);
            DEBUG_PRINT_1("First-order gradient after addition: %.6f\n", first_order_gradient);
        } else {
            DEBUG_PRINT_1("Not enough points to calculate first-order gradient.\n");
        }
    }

    DEBUG_PRINT_3("Completed tracking first-order gradients.\n");
}

/**
 * @brief Tracks the values added to the RLS array and calculates second-order gradients after each addition.
 *
 * @param measurements Array of measurement values to add to the RLS system.
 * @param length The number of points to add starting from the given start index.
 * @param start_index The index in the measurements array from which to begin adding values.
 * @param degree The degree of the polynomial to use for regression (2 for quadratic, 3 for cubic, etc.).
 */
void trackSecondOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree) {
    DEBUG_PRINT_3("Tracking second-order gradients.\n");
    // Initialize the regression state with the specified degree
    RegressionState regression_state;
    initialize_regression_state(&regression_state, degree);
    DEBUG_PRINT_2("Regression state initialized for second-order gradients.\n");

    // Loop through the measurements starting from start_index and add them to RLS
    for (uint16_t i = 0; i < length; ++i) {
        uint16_t current_index = start_index + i;

        // Add the current measurement to the regression
        add_data_point_to_regression(&regression_state, measurements[current_index]);
        DEBUG_PRINT_2("Added measurement %.6f at index %u to regression.\n", measurements[current_index], current_index);

        // Get the index of the most recently added data point
        double x_value = (double)(regression_state.total_data_points_added - 1);
        DEBUG_PRINT_2("Current x_value: %.2f\n", x_value);

        // Get the regression coefficients
        DEBUG_PRINT_3("Current regression coefficients:\n");
        for (int c = 0; c < regression_state.polynomial_degree + 1; ++c) {
            DEBUG_PRINT_3("Coefficient[%d] = %.6f\n", c, regression_state.coefficients[c]);
        }

        // Print the added x and y values
        DEBUG_PRINT_1("Added to RLS: x = %.2f, y = %.6f\n", x_value, measurements[current_index]);

        // If we have enough points, calculate the second-order gradient
        if (regression_state.current_num_points >= regression_state.polynomial_degree + 1) {
            double second_order_gradient = calculate_second_order_gradient(&regression_state, x_value);
            DEBUG_PRINT_1("Second-order gradient after addition: %.6f\n", second_order_gradient);
        } else {
            DEBUG_PRINT_1("Not enough points to calculate second-order gradient.\n");
        }
    }

    DEBUG_PRINT_3("Completed tracking second-order gradients.\n");
}
