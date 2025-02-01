#ifndef RLS_CHEBYSHEV_H
#define RLS_CHEBYSHEV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ********************************************************************
 * Configuration Macros
 * ********************************************************************/

/**
 * @brief Maximum polynomial degree for the Chebyshev basis.
 */
#define MAX_POLYNOMIAL_DEGREE 4

/** Size of the sliding window (number of data points used for regression) */
#define RLS_WINDOW 30


/* ********************************************************************
 * Data Structure Definitions
 * ********************************************************************/

/**
 * @brief Structure for a raw data point.
 *
 * Contains the measured phase angle.
 */
typedef struct {
    double phaseAngle; /**< Phase angle measurement */
} MqsRawDataPoint_t;

/**
 * @brief Structure holding the regression state.
 *
 * This structure contains the current number of data points in the sliding window,
 * the QR decomposition factors, the regression coefficients, and associated indices.
 */
typedef struct {
    int polynomial_degree;                              /**< Degree of the Chebyshev polynomial basis */
    unsigned short current_num_points;                  /**< Current number of data points in the window */
    unsigned short max_num_points;                      /**< Maximum number of data points (window size) */
    unsigned short oldest_data_index;                   /**< Index of the oldest data point in the window */
    unsigned int total_data_points_added;               /**< Total number of data points processed */
    double coefficients[MAX_POLYNOMIAL_DEGREE + 1];       /**< Regression coefficients (solution vector) */
    double upper_triangular_R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1]; /**< R factor from the QR decomposition */
    double Q_transpose_b[MAX_POLYNOMIAL_DEGREE + 1];      /**< Vector representing Qᵀ*b */
    int col_permutations[MAX_POLYNOMIAL_DEGREE + 1];      /**< Column permutation indices for pivoting */
} RegressionState;

/**
 * @brief Structure for storing computed gradients.
 *
 * Contains an array of gradients computed from the regression solution.
 */
typedef struct {
    double gradients[RLS_WINDOW];  /**< Array of computed gradients */
    unsigned short size;           /**< Number of gradients stored */
} GradientCalculationResult;

/* ********************************************************************
 * Function Prototypes
 * ********************************************************************/

/**
 * @brief Initializes the regression state.
 *
 * Resets all counters and zeroes the QR factors and coefficient vector.
 *
 * @param state Pointer to the RegressionState structure.
 * @param degree Degree of the Chebyshev polynomial basis.
 * @param max_num_points Maximum number of data points (window size).
 */
void initialize_regression_state(RegressionState *state, int degree, unsigned short max_num_points);

/**
 * @brief Calculates the first-order derivative (gradient) of the fitted function.
 *
 * Uses the Chebyshev basis representation and chain rule to compute df/dx.
 *
 * @param state Pointer to the RegressionState.
 * @param x The current x value.
 * @return The computed first-order derivative.
 */
double calculate_first_order_gradient_chebyshev(const RegressionState *state, double x);

/**
 * @brief Calculates the second-order derivative (curvature) of the fitted function.
 *
 * Computes f''(x) by obtaining the second derivative with respect to the normalized variable
 * and then applying the chain rule.
 *
 * @param state Pointer to the RegressionState.
 * @param x The current x value.
 * @return The computed second-order derivative.
 */
double calculate_second_order_gradient_chebyshev(const RegressionState *state, double x);

/**
 * @brief Adds a new data point to the regression model.
 *
 * Performs normalization, (optional) forgetting factor scaling, QR update (add row), and
 * possibly a QR downdate if the sliding window is full. Also recomputes the regression
 * coefficients and checks the condition number.
 *
 * @param state Pointer to the RegressionState.
 * @param data Pointer to the array of data points.
 * @param data_index Index of the new data point.
 */
void add_data_point_to_regression(RegressionState *state, const MqsRawDataPoint_t *data, unsigned short data_index);

/**
 * @brief Processes a sequence of data points and computes gradients.
 *
 * Iterates over a specified segment of data, updating the regression state and computing
 * the gradient at each step using the provided gradient calculation function.
 *
 * @param measurements Pointer to the array of data points.
 * @param length Number of data points to process.
 * @param start_index Starting index in the data array.
 * @param degree Degree of the Chebyshev basis.
 * @param calculate_gradient Function pointer for gradient calculation.
 * @param result Pointer to the structure that will store the computed gradients.
 */
void trackGradients(const MqsRawDataPoint_t *measurements, unsigned short length, unsigned short start_index, unsigned char degree,
                    double (*calculate_gradient)(const RegressionState *, double), GradientCalculationResult *result);

#ifdef __cplusplus
}
#endif

#endif /* RLS_CHEBYSHEV_H */
