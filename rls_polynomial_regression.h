/**
 * @file rls_polynomial_regression.h
 * @brief Header for the polynomial regression implementation using RLS and QR decomposition.
 */

#ifndef RLS_POLYNOMIAL_REGRESSION_H
#define RLS_POLYNOMIAL_REGRESSION_H

#include <stdint.h>

/** Define the size of the sliding window for RLS */
#define RLS_WINDOW 30  /**< Size of the sliding window for RLS */

#define MAX_POLYNOMIAL_DEGREE 3  /**< Maximum degree of polynomial supported (e.g., cubic) */

/**
 * @brief Structure to hold the state of the polynomial regression.
 */
typedef struct {
    uint16_t current_num_points;        /**< Current number of data points in the buffer */
    uint16_t max_num_points;            /**< Maximum number of data points (size of the buffer) */
    uint16_t oldest_data_index;         /**< Index of the oldest data point in the buffer */
    uint32_t total_data_points_added;   /**< Total number of data points added */
    uint8_t polynomial_degree;          /**< Degree of the polynomial (e.g., 2 for quadratic, 3 for cubic) */
    double coefficients[MAX_POLYNOMIAL_DEGREE + 1];             /**< Regression coefficients [aN, ..., a0] */
    double upper_triangular_R[MAX_POLYNOMIAL_DEGREE + 1][MAX_POLYNOMIAL_DEGREE + 1];    /**< Upper triangular matrix R */
    double Q_transpose_b[MAX_POLYNOMIAL_DEGREE + 1];            /**< Vector Q^T * b */
    double measurement_buffer[RLS_WINDOW];  /**< Circular buffer of measurement values */
    int reorthogonalization_counter;    /**< Counter for updates to trigger reorthogonalization */
} RegressionState;

/**
 * @brief Tracks the values added to the RLS array and calculates first-order gradients after each addition.
 *
 * @param measurements Array of measurement values to add to the RLS system.
 * @param length The number of points to add starting from the given start index.
 * @param start_index The index in the measurements array from which to begin adding values.
 * @param degree The degree of the polynomial to use for regression (2 for quadratic, 3 for cubic, etc.).
 */
void trackFirstOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree);

/**
 * @brief Tracks the values added to the RLS array and calculates second-order gradients after each addition.
 *
 * @param measurements Array of measurement values to add to the RLS system.
 * @param length The number of points to add starting from the given start index.
 * @param start_index The index in the measurements array from which to begin adding values.
 * @param degree The degree of the polynomial to use for regression (2 for quadratic, 3 for cubic, etc.).
 */
void trackSecondOrderGradients(const double *measurements, uint16_t length, uint16_t start_index, uint8_t degree);

#endif // RLS_POLYNOMIAL_REGRESSION_H
