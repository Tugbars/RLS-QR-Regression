/**
 * @file peakAnalysis.c
 * @brief Implements peak significance analysis functions using MAD-based statistics.
 *
 * This file provides functions to analyze the significance of detected peaks within gradient trends.
 * It utilizes robust statistical measures to ensure accurate and reliable peak detection.
 */

#include "peakAnalysis.h"
#include "rls_polynomial_regression.h"  // Ensure RLS functions are accessible
#include "trend_detection.h"            // Include trend detection structures and functions
#include "statistics.h"

#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>
#include <stdarg.h>  // For va_list, va_start, va_end

// Define minimum trend length for analysis
#define MIN_TREND_LENGTH 5  // Adjust as necessary

/** 
 * @def SIGNIFICANT_PEAK_THRESHOLD
 * @brief Threshold for maximum gradient to consider a peak as significant.
 */
#define SIGNIFICANT_PEAK_THRESHOLD 4.0

/** 
 * @def PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT
 * @brief Minimum number of consecutive points required to consider a trend as consistent during peak verification.
 */
#define PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT 5

/** 
 * @def PEAK_VERIFICATION_ALLOWABLE_INCONSISTENCY_COUNT
 * @brief Maximum number of consecutive minor fluctuations allowed during trend verification before considering the trend as broken.
 */
#define PEAK_VERIFICATION_ALLOWABLE_INCONSISTENCY_COUNT 1

/** 
 * @def SLOPE_THRESHOLD
 * @brief Threshold for second-order gradient to validate a peak.
 */
#define SLOPE_THRESHOLD 2.0

/** 
 * @def START_INDEX_OFFSET
 * @brief Number of positions to subtract from the start index for gradient tracking.
 */
#define START_INDEX_OFFSET 3


// Define debug levels
#define DEBUG_LEVEL 2  // Set to 0, 1, 2, or 3 to enable different levels of debugging

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
 
 
/**
 * @brief Analyzes gradient trends to determine peak characteristics.
 *
 * This function evaluates the gradient trends detected in a dataset to determine:
 * - Whether the current data window is centered on a peak (`isOnPeak`).
 * - Whether there is a significant peak within the data (`isSignificantPeak`).
 * - Whether centering is needed to align the window with the peak (`isCenteringNeeded`).
 * - The direction to move the data window based on the trends (`moveDirection`).
 *
 * The analysis is based on the following criteria:
 * 
 * **1. Check if On a Peak:**
 *    - Calculates the lengths of both the increasing and decreasing trends.
 *    - Compares these lengths against a minimum trend length threshold (`MIN_TREND_LENGTH`).
 *    - If both trends have lengths greater than or equal to `MIN_TREND_LENGTH`, it sets `isOnPeak` to `true`, indicating that the window is centered on a peak.
 *
 * **2. Check for a Significant Peak:**
 *    - Evaluates the maximum gradient value (`maxValue`) from the increasing trend.
 *    - Compares `maxValue` against a significant peak threshold (`SIGNIFICANT_PEAK_THRESHOLD`).
 *    - If `maxValue` is greater than or equal to `SIGNIFICANT_PEAK_THRESHOLD`, it sets `isSignificantPeak` to `true`, indicating the presence of a significant peak.
 *
 * **3. Determine if Centering is Needed:**
 *    - If a significant peak is detected (`isSignificantPeak` is `true`) but the data window is not centered on the peak (`isOnPeak` is `false`), it sets `isCenteringNeeded` to `true`.
 *    - This suggests that the window should be adjusted to better align with the peak.
 *
 * **4. Set Move Direction:**
 *    - Retrieves the move direction from the provided `trendResult`.
 *    - The move direction indicates how the data window should shift based on the detected trends:
 *        - `MOVE_RIGHT`: Shift the window to the right towards increasing trends.
 *        - `MOVE_LEFT`: Shift the window to the left towards decreasing trends.
 *        - `ON_THE_PEAK`: The window is centered on the peak; no movement needed.
 *        - `NO_TREND`: No dominant trend detected; movement is undecided.
 *
 * **Constants Used:**
 * - `MIN_TREND_LENGTH`: Defines the minimum length for a trend to be considered significant. Avoids using magic numbers.
 * - `SIGNIFICANT_PEAK_THRESHOLD`: Defines the threshold for the `maxValue` to consider a peak as significant.
 *
 * @param trendResult Pointer to the `GradientTrendResultAbsolute` structure containing the trend detection results.
 *
 * @return `PeakAnalysisResult` structure containing:
 * - `isOnPeak`: Indicates if the data window is centered on a peak.
 * - `isSignificantPeak`: Indicates if a significant peak is detected.
 * - `isCenteringNeeded`: Indicates if centering is required to align with the peak.
 * - `moveDirection`: The direction to move the data window based on trend analysis.
 */
PeakAnalysisResult analyzeGradientTrends(const GradientTrendResultAbsolute *trendResult) {
    PeakAnalysisResult result;
    result.isOnPeak = false;
    result.isSignificantPeak = false;
    result.isCenteringNeeded = false;
    result.moveDirection = NO_TREND;  // Default value

    // Check if both increase and decrease trends have lengths greater than MIN_TREND_LENGTH
    uint16_t increaseTrendLength = 0;
    uint16_t decreaseTrendLength = 0;

    if (trendResult->absoluteIncrease.startIndex != UINT16_MAX && trendResult->absoluteIncrease.endIndex != UINT16_MAX) {
        increaseTrendLength = trendResult->absoluteIncrease.endIndex - trendResult->absoluteIncrease.startIndex + 1;
    }

    if (trendResult->absoluteDecrease.startIndex != UINT16_MAX && trendResult->absoluteDecrease.endIndex != UINT16_MAX) {
        decreaseTrendLength = trendResult->absoluteDecrease.endIndex - trendResult->absoluteDecrease.startIndex + 1;
    }

    // Check if we are on a peak
    if (increaseTrendLength >= MIN_TREND_LENGTH && decreaseTrendLength >= MIN_TREND_LENGTH) {
        result.isOnPeak = true;
    }

    // Check if there is a significant peak
    double maxValue = trendResult->absoluteIncrease.maxValue;  // For increase trend
    if (maxValue >= SIGNIFICANT_PEAK_THRESHOLD) {
        result.isSignificantPeak = true;
    }

    // Determine if centering is needed
    if (result.isSignificantPeak && !result.isOnPeak) {
        result.isCenteringNeeded = true;
    }

    // If neither significant peak nor on peak, output is undecided
    if (!result.isSignificantPeak && !result.isOnPeak) {
        // Optionally, set a flag or leave as default
        // For this implementation, we'll consider isCenteringNeeded as false
    }

    // Get the moveDirection from the trendResult
    result.moveDirection = trendResult->moveDirection;

    return result;
}

/**
 * @brief Performs peak analysis by identifying gradient trends and analyzing them.
 *
 * This function wraps the process of calling `identifyGradientTrends` to obtain the trend results
 * and then `analyzeGradientTrends` to analyze the trends. It simplifies the usage by encapsulating
 * both steps and returning a `PeakAnalysisResult`.
 *
 * **Additional Functionality:**
 *
 * If a peak or significant increase is detected (`isOnPeak` or `isSignificantPeak` is `true`),
 * the function performs an additional check:
 *
 * - Uses the interval from the significant increase (`trendResult.absoluteIncrease.startIndex` to `endIndex`).
 * - Adjusts the start index to 3 positions before `startIndex`, ensuring it doesn't go below zero.
 * - Calls `trackGradients` over this adjusted interval to calculate second-order gradients.
 * - Checks if any of the second-order gradients exceed a threshold (e.g., 2.0).
 * - If they do, it sets `isValidPeak` to `true`, indicating it's a valid peak based on curvature.
 *
 * If neither `isOnPeak` nor `isSignificantPeak` is `true`, it calls `identifySplitMoveDirection`
 * to verify the `moveDirection`.
 *
 * @param values             Array of data points to analyze.
 * @param dataLength         The total length of the data array.
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
) {
    // Initialize Result Structures
    GradientTrendResultAbsolute trendResult;
    GradientCalculationResult gradCalcResultInc;
    GradientCalculationResult gradCalcResultDec;

    initializeGradientTrendResultAbsolute(&trendResult);
    memset(&gradCalcResultInc, 0, sizeof(GradientCalculationResult));
    memset(&gradCalcResultDec, 0, sizeof(GradientCalculationResult));

    // Perform Trend Detection
    identifyGradientTrends(
        values,
        startIndex,
        analysisLength,
        degree,
        gradientOrder,
        TREND_TYPE_GRADIENT_TRENDS,
        &trendResult,
        &gradCalcResultInc,
        &gradCalcResultDec
    );

    // Analyze Gradient Trends
    PeakAnalysisResult peakAnalysisResult = analyzeGradientTrends(&trendResult);

    // Initialize isValidPeak to false
    peakAnalysisResult.isValidPeak = false;

    // If a peak or significant increase is detected, perform additional check
    if (peakAnalysisResult.isOnPeak || peakAnalysisResult.isSignificantPeak) {
        // Get the significant increase interval
        uint16_t increaseStartIndex = trendResult.absoluteIncrease.startIndex;
        uint16_t increaseEndIndex = trendResult.absoluteIncrease.endIndex;

        // Check if indices are valid
        if (increaseStartIndex != UINT16_MAX && increaseEndIndex != UINT16_MAX) {
            // Adjust start index to start SLOPE_THRESHOLD_OFFSET positions earlier, ensure it doesn't go below zero
            uint16_t adjustedStartIndex = (increaseStartIndex >= START_INDEX_OFFSET) ? (increaseStartIndex - START_INDEX_OFFSET) : 0;
            uint16_t adjustedAnalysisLength = increaseEndIndex - adjustedStartIndex + 1;

            // Ensure adjustedAnalysisLength does not exceed data length
            if (adjustedStartIndex + adjustedAnalysisLength > dataLength) {
                adjustedAnalysisLength = dataLength - adjustedStartIndex;
            }

            // Prepare to calculate second-order gradients over this interval
            GradientCalculationResult secondOrderGradients;
            memset(&secondOrderGradients, 0, sizeof(GradientCalculationResult));

            // Use trackGradients to calculate second-order gradients
            trackGradients(
                values,
                adjustedAnalysisLength,
                adjustedStartIndex,
                degree,
                calculate_second_order_gradient,
                &secondOrderGradients
            );

            // Check if any of the second-order gradients exceed the threshold (SLOPE_THRESHOLD)
            for (uint16_t i = 0; i < secondOrderGradients.size; ++i) {
                if (secondOrderGradients.gradients[i] > SLOPE_THRESHOLD) {
                    // If it does, it's a valid peak
                    peakAnalysisResult.isValidPeak = true;
                    DEBUG_PRINT_2("Valid peak detected: second-order gradient %.2f exceeds threshold %.2f\n", secondOrderGradients.gradients[i], SLOPE_THRESHOLD);
                    break;
                }
            }
        } else {
            // Invalid indices, cannot perform additional check
            // isValidPeak remains false
            DEBUG_PRINT_1("Invalid increase interval indices: startIndex=%u, endIndex=%u\n", increaseStartIndex, increaseEndIndex);
        }
    } else {
        // If not on peak and no significant peak, verify moveDirection using identifySplitMoveDirection
        // Call identifySplitMoveDirection to verify moveDirection
        MoveDirection splitMoveDirection = identifySplitMoveDirection(
            values,
            startIndex,
            analysisLength,
            degree,
            gradientOrder
        );

        // Update the moveDirection in peakAnalysisResult
        peakAnalysisResult.moveDirection = splitMoveDirection;
    }

    return peakAnalysisResult;
}

/**
 * @brief Verifies the detected peak by checking for consistent trends on both sides.
 *
 * This function verifies whether a detected peak is a true peak by analyzing the second-order gradients
 * calculated using polynomial regression. Specifically, it checks for a consistent increasing trend on the left
 * side and a consistent decreasing trend on the right side of the peak.
 *
 * @param second_order_gradients Array containing the precomputed second-order gradients.
 * @param peak_index The index of the detected peak within the second_order_gradients array.
 * @param start_index The starting index in the original data array corresponding to the first element of second_order_gradients.
 * @return QuadraticPeakAnalysisResult Struct containing the verification result and truncation flags.
 */
QuadraticPeakAnalysisResult verify_quadratic_peak(
    const double *second_order_gradients,
    uint16_t peak_index,
    uint16_t start_index
) {
    QuadraticPeakAnalysisResult result = { 
        .peak_found = false, 
        .peak_index = start_index + peak_index, 
        .is_truncated_left = false, 
        .is_truncated_right = false 
    };
    uint16_t left_trend_count = 0;
    uint16_t right_trend_count = 0;
    uint16_t inconsistency_count_left = 0;
    uint16_t inconsistency_count_right = 0;

    // Debug statement
    printf("Verifying peak at index %u\n", result.peak_index);

    // Analyze left side: consistent increasing trend
    for (int i = peak_index; i > 0; --i) {
        double gradient = second_order_gradients[i - 1];
        if (gradient > 0) {
            left_trend_count++;
            // Update the longest consistent increasing trend
            if (left_trend_count >= PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT) {
                break;
            }
        } else {
            inconsistency_count_left++;
            if (inconsistency_count_left > PEAK_VERIFICATION_ALLOWABLE_INCONSISTENCY_COUNT) {
                break;
            }
        }
    }

    // Check for truncation on the left side
    if ((result.peak_index - left_trend_count) <= 0) {
        result.is_truncated_left = true;
        printf("Left side of peak is truncated.\n");
    }

    // Analyze right side: consistent decreasing trend
    for (uint16_t i = peak_index; i < (RLS_WINDOW - 1); ++i) {
        double gradient = second_order_gradients[i + 1];
        if (gradient < 0) {
            right_trend_count++;
            // Update the longest consistent decreasing trend
            if (right_trend_count >= PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT) {
                break;
            }
        } else {
            inconsistency_count_right++;
            if (inconsistency_count_right > PEAK_VERIFICATION_ALLOWABLE_INCONSISTENCY_COUNT) {
                break;
            }
        }
    }

    // Check for truncation on the right side
    if ((result.peak_index + right_trend_count) >= (start_index + RLS_WINDOW - 1)) {
        result.is_truncated_right = true;
        printf("Right side of peak is truncated.\n");
    }

    // Verify if the peak meets the trend criteria
    if (left_trend_count >= PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT &&
        right_trend_count >= PEAK_VERIFICATION_MIN_CONSISTENT_TREND_COUNT) {
        result.peak_found = true;
        printf("Peak at index %u verified successfully.\n", result.peak_index);
    } else {
        printf("Peak at index %u failed verification.\n", result.peak_index);
    }

    return result;
}


/**
 * @brief Verifies the validity of a detected peak using second-order gradients.
 *
 * This function calculates the second-order gradients around the detected peak and verifies
 * whether the peak meets the curvature criteria to be considered valid.
 *
 * @param values           Array of data points to analyze.
 * @param dataLength       The total length of the data array.
 * @param increaseStart    The start index of the significant increase interval.
 * @param increaseEnd      The end index of the significant increase interval.
 * @param degree           The degree of the polynomial used for regression.
 * @return QuadraticPeakAnalysisResult Struct containing the verification result and truncation flags.
 */
QuadraticPeakAnalysisResult verifyPeakValidity(
    const double *values,
    uint16_t dataLength,
    uint16_t increaseStart,
    uint16_t increaseEnd,
    uint8_t degree
) {
    QuadraticPeakAnalysisResult verificationResult = { 
        .peak_found = false, 
        .peak_index = 0, 
        .is_truncated_left = false, 
        .is_truncated_right = false 
    };

    // Adjust the start index by subtracting the offset, ensuring it doesn't go below zero
    uint16_t adjustedStartIndex = (increaseStart >= START_INDEX_OFFSET) ? (increaseStart - START_INDEX_OFFSET) : 0;
    uint16_t adjustedAnalysisLength = increaseEnd - adjustedStartIndex + 1;

    // Ensure adjustedAnalysisLength does not exceed data length
    if (adjustedStartIndex + adjustedAnalysisLength > dataLength) {
        adjustedAnalysisLength = dataLength - adjustedStartIndex;
    }

    // Calculate second-order gradients over the adjusted interval
    GradientCalculationResult secondOrderGradients;
    memset(&secondOrderGradients, 0, sizeof(GradientCalculationResult));

    trackGradients(
        values,
        adjustedAnalysisLength,
        adjustedStartIndex,
        degree,
        calculate_second_order_gradient,
        &secondOrderGradients
    );

    // Verify the peak using the verify_quadratic_peak function
    verificationResult = verify_quadratic_peak(
        secondOrderGradients.gradients,
        increaseEnd - adjustedStartIndex, // peak_index_relative within the gradients array
        adjustedStartIndex
    );

    return verificationResult;
}

/*
static void adjust_window_position(GradientTrendResult* gradient_trends) {
    uint16_t increase_duration = (gradient_trends->increase_info.end_index + buffer_manager.buffer_size 
                                  - gradient_trends->increase_info.start_index) % buffer_manager.buffer_size;
    uint16_t decrease_duration = (gradient_trends->decrease_info.end_index + buffer_manager.buffer_size 
                                  - gradient_trends->decrease_info.start_index) % buffer_manager.buffer_size;

    int shift_amount = 0;
    int direction = UNDECIDED;

    if (increase_duration > decrease_duration) {
        shift_amount = (increase_duration - decrease_duration) / CENTERING_RATIO;
        direction = RIGHT_SIDE;
        printf("Increase duration (%u) > decrease duration (%u). Moving right by %d.\n", increase_duration, decrease_duration, shift_amount);
    } else if (decrease_duration > increase_duration) {
        shift_amount = (decrease_duration - increase_duration) / CENTERING_RATIO;
        direction = LEFT_SIDE;
        printf("Decrease duration (%u) > increase duration (%u). Moving left by %d.\n", decrease_duration, increase_duration, shift_amount);
    } else {
        currentStatus.isCentered = 1;
        printf("Increase and decrease durations are equal. Peak is centered.\n");
        STATE_FUNCS[SWP_PEAK_CENTERING].isComplete = true;
    }

    if (shift_amount != 0) {
        move_window_and_update_if_needed(direction, shift_amount);
    }
}
*/

//centering left. 

//peak verification geldi peak bulma gelmedi. 

//sonra centering. 
