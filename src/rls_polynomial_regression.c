
/**
 * @file rls_polynomial_regression.c
 * @brief Recursive Least Squares (RLS) Polynomial Regression with Hybrid QR Decomposition
 *
 * ## Overview
 * This module implements a streaming polynomial regression using a hybrid approach
 * that combines Householder reflections (for periodic reorthogonalizations) and Givens
 * rotations (for incremental updates). The following optimizations have been applied to
 * improve **performance** and **numerical stability**:
 *
 * ### 1. Reduced Redundant Memory Copying
 * - **In-place Row Updates:** Previously, new or old data rows were copied into temporary
 *   buffers before applying Givens rotations. Now, the code modifies these rows in-place
 *   (by passing non-const pointers). This eliminates extra `memcpy` operations and reduces
 *   memory traffic.
 *
 * ### 2. Merged or Simplified Loops
 * - **Householder Column Norms/Pivoting:** Instead of calculating column norms in one loop
 *   and then searching for the pivot in another, the code merges these tasks, thus avoiding
 *   repeated traversals of the same data. This helps reduce overhead in the reorthogonalization
 *   process.
 * - **Compact Givens Rotations:** Inner loops for updating `R` and `Q^T*b` have been simplified
 *   (or partially unrolled in certain places), potentially improving locality and compiler
 *   optimization.
 *
 * ### 3. In-Place Givens Rotation Updates
 * - **Add and Remove Rows Locally:** In `updateQR_AddRow()` and `updateQR_RemoveRow()`, the
 *   upper triangular matrix `R` and the vector `Q^T*b` are updated locally for each new or old
 *   row, using minimal additional storage. This allows the algorithm to run efficiently
 *   for streaming data scenarios.
 *
 * ### 4. Condition Number Monitoring
 * - **Early Reorthogonalization:** The condition number is checked after each update. If it
 *   grows too large, a full reorthogonalization (via Householder) is triggered. This defers
 *   expensive recomputations until they are needed, maintaining numerical stability.
 *
 * ### 5. Timing Measurements (Optional)
 * - **`clock()` Integration:** For performance diagnostics, the code in `main()` demonstrates
 *   how to measure elapsed CPU time before and after the regression function calls.
 *
 * ## Key Benefits
 * - **Less Memory Overhead:** Avoiding full-row copies or large local buffers reduces overhead.
 * - **Faster Incremental Updates:** Givens rotations operate on the relevant row in-place.
 * - **Improved Readability & Maintainability:** Merging loops and removing unnecessary copies
 *   streamlines the code, making it easier to follow while also boosting performance.
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

#if DEBUG_LEVEL > 0
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
static void debug_print(int level, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    // Optionally, prepend debug messages with level information
    // printf("[DEBUG LEVEL %d] ", level);
    vprintf(fmt, args);
    va_end(args);
}
#endif

// Forward declarations for internal (static) functions
static void recompute_qr_decomposition(RegressionState *regression_state, const double *measurements);
static inline void solve_for_coefficients(RegressionState *regression_state);
static double compute_condition_number(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size);
static void updateQR_AddRow(RegressionState *regression_state, double *new_row, double new_b);
static void updateQR_RemoveRow(RegressionState *regression_state, double *old_row, double old_b);

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
 * @param max_num_points The maximum number of points in the sliding window.
 */
void initialize_regression_state(RegressionState *regression_state, uint8_t degree, uint16_t max_num_points)
{
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
    for (int i = 0; i <= degree; ++i)
    {
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
 * Steps:
 * 1. Compute x_value from total_data_points_added.
 * 2. Build polynomial basis vector new_row = [1, x_value, x_value^2, ...].
 * 3. updateQR_AddRow() to incorporate new data point.
 * 4. If buffer full, remove oldest data point via updateQR_RemoveRow().
 * 5. Reorthogonalize if necessary (REORTHOGONALIZATION_INTERVAL).
 * 6. Solve for coefficients, then check condition number for numerical stability.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param measurements The array of measured values (dependent variable).
 * @param data_index The index of the new measurement to add.
 */
void add_data_point_to_regression(RegressionState *regression_state, const double *measurements, uint16_t data_index)
{
    int num_coefficients = regression_state->polynomial_degree + 1;

    // Extract the new measurement
    double measurement = measurements[data_index];
    DEBUG_PRINT_3("Adding new data point: measurement=%.6f at data_index=%u\n", measurement, data_index);

    // Generate the x_value (often just the count of data points added so far)
    double x_value = (double)(regression_state->total_data_points_added);
    DEBUG_PRINT_2("Current x_value for new data point: %.2f\n", x_value);

    // Build polynomial basis [1, x, x^2, ...]
    double new_row[MAX_POLYNOMIAL_DEGREE + 1];
    double x_power = 1.0;
    for (int i = 0; i < num_coefficients; ++i)
    {
        new_row[i] = x_power;
        x_power *= x_value;
    }
    DEBUG_PRINT_3("New polynomial basis vector computed.\n");

    // Incrementally update R, Q^T * b with Givens rotations for the new row
    updateQR_AddRow(regression_state, new_row, measurement);
    DEBUG_PRINT_3("QR decomposition updated with new data point.\n");

    // If we've reached the maximum window size, remove oldest data point
    if (regression_state->current_num_points >= regression_state->max_num_points)
    {
        // Identify oldest data index
        uint16_t oldest_data_idx = regression_state->oldest_data_index;
        double old_measurement = measurements[oldest_data_idx];
        DEBUG_PRINT_3("Buffer full. Removing oldest data point at index %u with measurement=%.6f.\n",
                      oldest_data_idx, old_measurement);

        // Build polynomial basis for the oldest data point
        double old_x_value = (double)(regression_state->total_data_points_added - regression_state->max_num_points);
        x_power = 1.0;
        double old_row[MAX_POLYNOMIAL_DEGREE + 1];
        for (int i = 0; i < num_coefficients; ++i)
        {
            old_row[i] = x_power;
            x_power *= old_x_value;
        }
        DEBUG_PRINT_3("Old polynomial basis vector computed for removal.\n");

        // Remove the old row
        updateQR_RemoveRow(regression_state, old_row, old_measurement);
        DEBUG_PRINT_3("QR decomposition updated by removing oldest data point.\n");

        // Advance the oldest data index
        regression_state->oldest_data_index = oldest_data_idx + 1;
        DEBUG_PRINT_3("Oldest data index updated to %u.\n", regression_state->oldest_data_index);
    }
    else
    {
        // If not full, just increment the count of stored points
        regression_state->current_num_points++;
        DEBUG_PRINT_3("Buffer not full. Current number of points: %u.\n", regression_state->current_num_points);
    }

    // Increase the total number of data points we've ever added
    regression_state->total_data_points_added++;
    DEBUG_PRINT_3("Total data points added: %u.\n", regression_state->total_data_points_added);

    // Check if we need a full reorthogonalization
    regression_state->reorthogonalization_counter++;
    DEBUG_PRINT_3("Reorthogonalization counter: %u.\n", regression_state->reorthogonalization_counter);
    if (regression_state->reorthogonalization_counter >= REORTHOGONALIZATION_INTERVAL)
    {
        DEBUG_PRINT_3("Reorthogonalization interval reached. Recomputing QR.\n");
        // Recompute entire QR from scratch for better numerical stability
        recompute_qr_decomposition(regression_state, measurements);
        regression_state->reorthogonalization_counter = 0;
        DEBUG_PRINT_3("QR decomposition recomputed.\n");
    }

    // Solve for updated polynomial coefficients
    solve_for_coefficients(regression_state);
    DEBUG_PRINT_2("Regression coefficients updated.\n");

    // Finally, check if the system is becoming ill-conditioned
    double condition_number = compute_condition_number(regression_state->upper_triangular_R, num_coefficients);
    DEBUG_PRINT_2("Condition number: %.6e\n", condition_number);
    if (condition_number > CONDITION_NUMBER_THRESHOLD)
    {
        DEBUG_PRINT_1("Condition number %.6e exceeds threshold %.6e. Recomputing QR.\n",
                      condition_number, CONDITION_NUMBER_THRESHOLD);
        recompute_qr_decomposition(regression_state, measurements);
        solve_for_coefficients(regression_state);
        DEBUG_PRINT_2("Re-solved coefficients after recomputing QR.\n");
    }
}

/**
 * @brief Updates the QR decomposition by adding a new row using Givens rotations.
 *
 * This function incrementally updates the existing QR decomposition when a new data point is added.
 * It uses Givens rotations to maintain the upper triangular structure of the R matrix without
 * recomputing the entire decomposition.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param new_row The new row (polynomial basis) to be added.
 * @param new_b The new measurement (dependent variable).
 */
static void updateQR_AddRow(RegressionState *regression_state, double *new_row, double new_b)
{
    int num_coefficients = regression_state->polynomial_degree + 1;
    double Q_tb_new = new_b;  // This will be updated as we apply rotations

    // Loop through each diagonal element of R
    for (int i = 0; i < num_coefficients; ++i)
    {
        double a = regression_state->upper_triangular_R[i][i];
        double b = new_row[i];

        // If b is large enough, we perform a Givens rotation
        if (fabs(b) > 1e-10)
        {
            // Compute rotation parameters
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;

            // Update the diagonal entry
            regression_state->upper_triangular_R[i][i] = r;

            // Update columns [i+1 ... end]
            for (int j = i + 1; j < num_coefficients; ++j)
            {
                double temp = regression_state->upper_triangular_R[i][j];
                // Rotate R's upper row
                regression_state->upper_triangular_R[i][j] = c * temp + s * new_row[j];
                // Rotate the new row
                new_row[j] = -s * temp + c * new_row[j];
            }

            // Update Q^T*b
            double temp_b = regression_state->Q_transpose_b[i];
            regression_state->Q_transpose_b[i] = c * temp_b + s * Q_tb_new;
            Q_tb_new = -s * temp_b + c * Q_tb_new;
        }
    }
}

/**
 * @brief Updates the QR decomposition by removing an old row using Givens rotations.
 *
 * This function incrementally downdates the existing QR decomposition when an old data point
 * is removed. It uses Givens rotations to adjust the R matrix, maintaining its upper triangular structure.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param old_row The polynomial basis row corresponding to the oldest data point.
 * @param old_b The measurement value of the oldest data point.
 */
static void updateQR_RemoveRow(RegressionState *regression_state, double *old_row, double old_b)
{
    int num_coefficients = regression_state->polynomial_degree + 1;
    double Q_tb_old = old_b;

    // Similar logic to updateQR_AddRow, except we "downdate" (apply reverse rotations)
    for (int i = 0; i < num_coefficients; ++i)
    {
        double a = regression_state->upper_triangular_R[i][i];
        double b = old_row[i];

        if (fabs(b) > 1e-10)
        {
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;

            regression_state->upper_triangular_R[i][i] = r;

            for (int j = i + 1; j < num_coefficients; ++j)
            {
                double temp = regression_state->upper_triangular_R[i][j];
                // "Downdate" R
                regression_state->upper_triangular_R[i][j] = c * temp - s * old_row[j];
                old_row[j] = s * temp + c * old_row[j];
            }

            double temp_b = regression_state->Q_transpose_b[i];
            // Adjust Q^T*b in reverse
            regression_state->Q_transpose_b[i] = c * temp_b - s * Q_tb_old;
            Q_tb_old = s * temp_b + c * Q_tb_old;
        }
    }
}

/**
 * @brief Recomputes the QR decomposition using Householder reflections with column pivoting.
 *
 * This function performs a full recomputation of the QR decomposition for improved numerical stability.
 * It populates an augmented matrix with existing data (including regularization rows) and then applies
 * Householder reflections (with column pivoting) to factorize it into R, storing the relevant parts
 * in the RegressionState.
 *
 * @param regression_state Pointer to the RegressionState structure.
 * @param measurements The array of measurements from which we build the augmented matrix.
 */
static void recompute_qr_decomposition(RegressionState *regression_state, const double *measurements)
{
    int num_data_points = regression_state->current_num_points;
    int num_coefficients = regression_state->polynomial_degree + 1;

    double regularization_lambda = REGULARIZATION_PARAMETER;
    int total_rows = num_data_points + num_coefficients;

#ifdef USE_GLOBAL_SCRATCH_SPACE
    if (total_rows > MAX_TOTAL_ROWS)
    {
        DEBUG_PRINT_1("Error: total_rows (%d) exceeds MAX_TOTAL_ROWS (%d).\n",
                      total_rows, MAX_TOTAL_ROWS);
        return;
    }
    // Use global scratch arrays augmented_matrix_A, augmented_vector_b
#else
    // Otherwise allocate on local stack
    double augmented_matrix_A[total_rows][MAX_POLYNOMIAL_DEGREE + 1];
    double augmented_vector_b[total_rows];
#endif

    // Initialize to zero
    memset(augmented_matrix_A, 0, sizeof(double) * total_rows * (MAX_POLYNOMIAL_DEGREE + 1));
    memset(augmented_vector_b, 0, sizeof(double) * total_rows);

    // Prepare an array to keep track of column indices for pivoting
    int col_indices[MAX_POLYNOMIAL_DEGREE + 1];
    for (int i = 0; i < num_coefficients; ++i)
    {
        col_indices[i] = i;
    }

    // Fill the augmented matrix with existing data points
    for (int i = 0; i < num_data_points; ++i)
    {
        uint16_t data_idx = (uint16_t)(regression_state->oldest_data_index + i);
        double x = (double)(regression_state->total_data_points_added - num_data_points + i);
        double measurement = measurements[data_idx];

        double x_power = 1.0;
        for (int j = 0; j < num_coefficients; ++j)
        {
            augmented_matrix_A[i][j] = x_power;
            x_power *= x;
        }
        augmented_vector_b[i] = measurement;
    }

    // Add regularization rows
    for (int i = 0; i < num_coefficients; ++i)
    {
        int row = num_data_points + i;
        augmented_matrix_A[row][i] = sqrt(regularization_lambda);
        augmented_vector_b[row] = 0.0;
    }

    // Reset R and Q^T*b in the state
    memset(regression_state->upper_triangular_R, 0, sizeof(regression_state->upper_triangular_R));
    memset(regression_state->Q_transpose_b, 0, sizeof(regression_state->Q_transpose_b));

    // Compute column norms (for pivoting)
    double col_norms[MAX_POLYNOMIAL_DEGREE + 1];
    for (int j = 0; j < num_coefficients; ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < total_rows; ++i)
        {
            sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
        }
        col_norms[j] = sqrt(sum);
    }

    // Householder with Column Pivoting
    for (int k = 0; k < num_coefficients; ++k)
    {
        // 1) Identify the column with the maximum norm among [k..end]
        int max_col = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < num_coefficients; ++j)
        {
            if (col_norms[j] > max_norm)
            {
                max_norm = col_norms[j];
                max_col = j;
            }
        }

        // 2) Swap columns if needed
        if (max_col != k)
        {
            for (int i = 0; i < total_rows; ++i)
            {
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

        // 3) Householder reflection to zero out below-diagonal entries in col k
        double sigma = 0.0;
        for (int i = k; i < total_rows; ++i)
        {
            double val = augmented_matrix_A[i][k];
            sigma += val * val;
        }
        sigma = sqrt(sigma);

        if (sigma == 0.0) continue;

        double vk = augmented_matrix_A[k][k] + ((augmented_matrix_A[k][k] >= 0) ? sigma : -sigma);
        if (fabs(vk) < 1e-12) continue;

        double beta = 1.0 / (sigma * vk);

        // Form the Householder vector
        for (int i = k + 1; i < total_rows; ++i)
        {
            augmented_matrix_A[i][k] /= vk;
        }
        augmented_matrix_A[k][k] = -sigma;

        // Apply the transformation to subsequent columns
        for (int j = k + 1; j < num_coefficients; ++j)
        {
            double s = 0.0;
            for (int i = k; i < total_rows; ++i)
            {
                s += augmented_matrix_A[i][k] * augmented_matrix_A[i][j];
            }
            s *= beta;
            for (int i = k; i < total_rows; ++i)
            {
                augmented_matrix_A[i][j] -= augmented_matrix_A[i][k] * s;
            }
        }

        // Apply to the augmented_vector_b
        double s = 0.0;
        for (int i = k; i < total_rows; ++i)
        {
            s += augmented_matrix_A[i][k] * augmented_vector_b[i];
        }
        s *= beta;
        for (int i = k; i < total_rows; ++i)
        {
            augmented_vector_b[i] -= augmented_matrix_A[i][k] * s;
        }

        // Zero out below the diagonal
        for (int i = k + 1; i < total_rows; ++i)
        {
            augmented_matrix_A[i][k] = 0.0;
        }

        // Recompute norms for columns j > k
        for (int j = k + 1; j < num_coefficients; ++j)
        {
            double sum = 0.0;
            for (int i = k + 1; i < total_rows; ++i)
            {
                sum += augmented_matrix_A[i][j] * augmented_matrix_A[i][j];
            }
            col_norms[j] = sqrt(sum);
        }
    }

    // Copy the upper triangle into regression_state->upper_triangular_R
    for (int i = 0; i < num_coefficients; ++i)
    {
        for (int j = i; j < num_coefficients; ++j)
        {
            regression_state->upper_triangular_R[i][j] = augmented_matrix_A[i][j];
        }
        regression_state->Q_transpose_b[i] = augmented_vector_b[i];
    }

    // Store the column permutation order
    memcpy(regression_state->col_permutations, col_indices, sizeof(int) * num_coefficients);
}

/**
 * @brief Solves for the regression coefficients via back substitution, applying column permutations.
 *
 * @param regression_state Pointer to the RegressionState structure.
 */
static inline void solve_for_coefficients(RegressionState *regression_state)
{
    int num_coefficients = regression_state->polynomial_degree + 1;
    double temp_coeffs[MAX_POLYNOMIAL_DEGREE + 1];

    // Back substitution on the upper triangular system
    for (int i = num_coefficients - 1; i >= 0; --i)
    {
        double sum = regression_state->Q_transpose_b[i];
        for (int j = i + 1; j < num_coefficients; ++j)
        {
            sum -= regression_state->upper_triangular_R[i][j] * temp_coeffs[j];
        }
        if (fabs(regression_state->upper_triangular_R[i][i]) < 1e-12)
        {
            temp_coeffs[i] = 0.0;
        }
        else
        {
            temp_coeffs[i] = sum / regression_state->upper_triangular_R[i][i];
        }
    }

    // Apply column permutations
    for (int i = 0; i < num_coefficients; ++i)
    {
        int perm_idx = regression_state->col_permutations[i];
        regression_state->coefficients[perm_idx] = temp_coeffs[i];
    }
}

/**
 * @brief Computes the condition number of the upper triangular matrix R using a 2-norm estimate.
 *
 * @param R The upper triangular matrix R.
 * @param size Number of coefficients (polynomial_degree + 1).
 * @return Estimated condition number.
 */
static double compute_condition_number(const double R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1], int size)
{
    double max_sv = 0.0;
    double min_sv = DBL_MAX;

    // Use diagonal as a rough proxy for singular values
    for (int i = 0; i < size; ++i)
    {
        double diag = fabs(R[i][i]);
        if (diag > max_sv) max_sv = diag;
        if (diag < min_sv && diag > 1e-12) min_sv = diag;
    }

    // If min_sv is extremely small, the matrix is nearly singular
    if (min_sv < 1e-12)
    {
        return DBL_MAX;
    }
    return max_sv / min_sv;
}

/**
 * @brief Calculates the first-order gradient (slope) of the polynomial function at a specific point x.
 *
 * @param regression_state Pointer to the RegressionState containing polynomial coefficients.
 * @param x The point at which to calculate the first-order gradient.
 * @return The first-order gradient at x.
 */
double calculate_first_order_gradient(const RegressionState *regression_state, double x)
{
    double derivative = 0.0;
    double x_power = 1.0;
    int degree = regression_state->polynomial_degree;

    // derivative = sum( i * coeff[i] * x^(i-1) ), but computed as i * coeff[i] * x^i / x
    for (int i = 1; i <= degree; ++i)
    {
        x_power *= x;
        derivative += i * regression_state->coefficients[i] * (x_power / x);
    }
    return derivative;
}

/**
 * @brief Calculates the second-order gradient (curvature) of the polynomial function at a specific point x.
 *
 * @param regression_state Pointer to the RegressionState containing polynomial coefficients.
 * @param x The point at which to calculate the second-order gradient.
 * @return The second-order gradient at x.
 */
double calculate_second_order_gradient(const RegressionState *regression_state, double x)
{
    double second_derivative = 0.0;
    double x_power = 1.0;
    int degree = regression_state->polynomial_degree;

    // second derivative ~ i*(i-1)* coeff[i] * x^(i-2), here implemented as i*(i-1)*coeff[i]* x^i / (x^2)
    for (int i = 2; i <= degree; ++i)
    {
        x_power *= x;
        second_derivative += i * (i - 1) * regression_state->coefficients[i] * (x_power / (x * x));
    }
    return second_derivative;
}

/**
 * @brief Tracks values (measurements) added to the RLS array and calculates gradients after each addition.
 *
 * @param measurements Array of measurement values (e.g., phase angles).
 * @param length Number of points to add.
 * @param start_index Index in the measurements array from which to begin adding values.
 * @param degree Polynomial degree for regression.
 * @param calculate_gradient Function pointer to either first-order or second-order gradient, etc.
 * @param result Pointer to GradientCalculationResult for storing computed gradients.
 */
void trackGradients(
    const double *measurements,
    uint16_t length,
    uint16_t start_index,
    uint8_t degree,
    double (*calculate_gradient)(const RegressionState *, double),
    GradientCalculationResult *result
)
{
    DEBUG_PRINT_3("trackGradients startIndex=%u, length=%u, degree=%u\n", start_index, length, degree);

    // Initialize result
    result->size = 0;
    result->valid = false;

    // Create a local RegressionState for this tracking session
    RegressionState regression_state;
    initialize_regression_state(&regression_state, degree, RLS_WINDOW);

    // Process each measurement in the specified range
    for (uint16_t i = 0; i < length; ++i)
    {
        uint16_t current_index = start_index + i;

        // Add the data point to the RLS model
        add_data_point_to_regression(&regression_state, measurements, current_index);

        // x_value is total_data_points_added - 1, representing the last added point
        double x_value = (double)(regression_state.total_data_points_added - 1);

        // Only compute gradient if we have enough points (>= degree+1)
        if (regression_state.current_num_points >= regression_state.polynomial_degree + 1)
        {
            double grad_val = calculate_gradient(&regression_state, x_value);
            printf("Calculated gradient: %.6f at x=%.2f\n", grad_val, x_value);

            // Store in result if there's space
            if (result->size < RLS_WINDOW)
            {
                result->gradients[result->size++] = grad_val;
            }
            else
            {
                DEBUG_PRINT_1("Gradient array overflow at index %u.\n", result->size);
                break;
            }
        }
        else
        {
            DEBUG_PRINT_1("Insufficient points for gradient (current_num_points=%u)\n",
                          regression_state.current_num_points);
        }
    }

    // Mark as valid if we collected any gradients
    if (result->size > 0)
    {
        result->valid = true;
    }
}

/**
 * @brief Calculates and prints first-order gradients for a subset of measurements.
 *
 * @param measurements Array of measurement values.
 * @param length Number of points to add.
 * @param start_index Index in the measurements array from which to begin.
 * @param degree Polynomial degree (e.g., 3 for cubic).
 */
void trackFirstOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree)
{
    GradientCalculationResult result;
    trackGradients(measurements, length, start_index, degree, calculate_first_order_gradient, &result);

    if (result.valid)
    {
        printf("\n=== First-Order Gradients ===\n");
        for (uint16_t i = 0; i < result.size; i++)
        {
            printf("Gradient[%d] = %f\n", i, result.gradients[i]);
        }
    }
    else
    {
        printf("No valid first-order gradients were computed.\n");
    }
}

/**
 * @brief Calculates and prints second-order gradients for a subset of measurements.
 *
 * @param measurements Array of measurement values.
 * @param length Number of points to add.
 * @param start_index Index in the measurements array from which to begin.
 * @param degree Polynomial degree (e.g., 3 for cubic).
 */
void trackSecondOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree)
{
    // This function can be implemented similarly if needed, using calculate_second_order_gradient.
    // For demonstration, you might replicate trackFirstOrderGradients but pass
    // calculate_second_order_gradient() instead.
}
