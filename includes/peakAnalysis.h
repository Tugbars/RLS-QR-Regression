/**
 * @file peakSignificanceAnalysis.h
 * @brief Header file for peak significance analysis functions.
 *
 * This header defines the structures and function prototypes for analyzing the significance
 * of detected peaks within gradient trends.
 */

#ifndef PEAK_SIGNIFICANCE_ANALYSIS_H
#define PEAK_SIGNIFICANCE_ANALYSIS_H

#include "trend_detection.h"  // To access GradientTrendResult and related structures
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Structure to hold peak analysis results.
 */
typedef struct {
    bool isOnPeak;            /**< Indicates if the current window is on a peak */
    bool isSignificantPeak;   /**< Indicates if a significant peak was detected */
    bool isCenteringNeeded;   /**< Indicates if centering is needed */
    MoveDirection moveDirection;  /**< Move direction based on the gradient trend */
    bool isValidPeak;         /**< Indicates if the peak is valid based on second-order gradient check */
} PeakAnalysisResult;


/**
 * @brief Structure to hold the result of quadratic peak verification.
 */
typedef struct {
    bool peak_found;            /**< Indicates if a peak was found and verified */
    uint16_t peak_index;        /**< Index of the detected peak in the dataset */
    bool is_truncated_left;     /**< Indicates if the left side of the peak is truncated */
    bool is_truncated_right;    /**< Indicates if the right side of the peak is truncated */
} QuadraticPeakAnalysisResult;

/**
 * @brief Performs peak analysis by identifying gradient trends and analyzing them.
 *
 * @param values             Array of data points to analyze.
 * @param startIndex         The starting index in the values array.
 * @param analysisLength     The number of data points to include in the analysis window.
 * @param degree             The degree of the polynomial used for regression.
 * @param gradientOrder      The order of the gradient (first-order or second-order).
 *
 * @return PeakAnalysisResult The result of the peak analysis.
 */
PeakAnalysisResult performPeakAnalysis(
    const double *values,
    uint16_t dataLength,
    uint16_t startIndex,
    uint16_t analysisLength,
    uint8_t degree,
    GradientOrder gradientOrder
);

/**
 * @brief Finds and verifies peaks in the data using polynomial RLS and trend analysis.
 *
 * @param values Array of data points.
 * @param length Length of the data array.
 * @param start_index The start index in the data array from which to begin the gradient calculation.
 * @param degree The degree of the polynomial to use for regression (e.g., 2 for quadratic, 3 for cubic).
 * @return QuadraticPeakAnalysisResult Structure containing the peak detection status, truncation flags, and the peak index if found and verified.
 */
QuadraticPeakAnalysisResult find_and_verify_quadratic_peak(
    const double *values,
    uint16_t length,
    uint16_t start_index,
    uint8_t degree
);

#endif // PEAK_SIGNIFICANCE_ANALYSIS_H


