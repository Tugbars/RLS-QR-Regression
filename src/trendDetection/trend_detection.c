/**
 * @file trend_detection.c
 * @brief Implements trend detection functions for identifying consistent increasing and decreasing trends
 *        using gradients calculated via Recursive Least Squares (RLS) polynomial regression.
 *
 * This file provides functions to identify and analyze consistent trends within a dataset of gradient values.
 * It separately tracks the maximum and minimum gradients within increasing and decreasing trends,
 * respectively, and accounts for minor fluctuations using configurable parameters. 
 * 
 * The trend detection mechanisms are essential for applications like signal processing, financial data analysis,
 * or any time-series data analysis where discerning underlying trends is crucial.
 *
 * The functions utilize Recursive Least Squares (RLS) regression from `rls_polynomial_regression.h`
 * to calculate first-order and second-order gradients over a sliding window, offering a robust
 * method for online trend analysis.
 */

#include "trend_detection.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>  // For va_list, va_start, va_end
#include "statistics.h"

// Preprocessor constants for trend detection configuration

/** 
 * @def SIGNIFICANCE_FACTOR
 * @brief Factor multiplied by MAD to determine significance threshold.
 *
 * This factor adjusts the sensitivity of trend detection. A higher value ensures that only larger deviations
 * from the median are considered significant.
 */
#define SIGNIFICANCE_FACTOR 0.3  // Adjust this factor as needed

/** 
 * @def MAX_ALLOWED_DECREASES
 * @brief Maximum number of consecutive minor decreases allowed within an increasing trend before considering the trend broken.
 *
 * This parameter allows the trend detection algorithm to tolerate small fluctuations within an increasing trend.
 * If the number of consecutive decreases exceeds this value, the trend is considered to have ended.
 */
#define MAX_ALLOWED_DECREASES 1

/** 
 * @def MAX_ALLOWED_INCREASES
 * @brief Maximum number of consecutive minor increases allowed within a decreasing trend before considering the trend broken.
 *
 * Similar to `MAX_ALLOWED_DECREASES`, this parameter allows small increases within a decreasing trend.
 */
#define MAX_ALLOWED_INCREASES 1

// Define debug levels
#define DEBUG_LEVEL 0  // Set to 0, 1, 2, or 3 to enable different levels of debugging

// Define separate debug print macros based on DEBUG_LEVEL

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

// Function to print trend details
void printTrendDetails(const char* trendName, const GradientTrendIndices* trendIndices) {
    if (trendIndices->startIndex != UINT16_MAX && trendIndices->endIndex != UINT16_MAX) {
        printf("%s Trend:\n", trendName);
        printf("Start Index: %u\n", trendIndices->startIndex);
        printf("End Index: %u\n", trendIndices->endIndex);
        printf("Max Value: %.6f at Index %u\n", trendIndices->maxValue, trendIndices->maxValueIndex);
        printf("Sum of Gradients: %.6f\n", trendIndices->sumGradients);
    } else {
        printf("No %s Trend Detected.\n", trendName);
    }
}

/**
 * @brief Initializes the GradientTrendResultAbsolute structure.
 *
 * This function resets the fields of the GradientTrendResultAbsolute structure to their default values.
 * It also sets up the base pointers to the respective fields, allowing for generic access in functions
 * that operate on the base structure.
 *
 * @param result Pointer to the GradientTrendResultAbsolute structure to initialize.
 */
void initializeGradientTrendResultAbsolute(GradientTrendResultAbsolute *result) {
    result->absoluteIncrease.startIndex = UINT16_MAX;
    result->absoluteIncrease.endIndex = UINT16_MAX;
    result->absoluteIncrease.maxValue = 0.0;
    result->absoluteIncrease.maxValueIndex = UINT16_MAX;
    result->absoluteIncrease.sumGradients = 0.0;

    result->absoluteDecrease.startIndex = UINT16_MAX;
    result->absoluteDecrease.endIndex = UINT16_MAX;
    result->absoluteDecrease.maxValue = 0.0;
    result->absoluteDecrease.maxValueIndex = UINT16_MAX;
    result->absoluteDecrease.sumGradients = 0.0;

    result->moveDirection = NO_TREND;
    result->isSignificantPeak = false;
    result->moveAmountAbsolute = 0;

    // Initialize base pointers
    result->base.increase = &(result->absoluteIncrease);
    result->base.decrease = &(result->absoluteDecrease);
    result->base.moveDirection = &(result->moveDirection);
    result->base.isSignificantPeak = &(result->isSignificantPeak);
    result->base.moveAmount = &(result->moveAmountAbsolute);
}


/**
 * @brief Initializes the GradientTrendResultSignificant structure.
 *
 * Similar to `initializeGradientTrendResultAbsolute`, this function resets the fields of the
 * GradientTrendResultSignificant structure and sets up the base pointers.
 *
 * @param result Pointer to the GradientTrendResultSignificant structure to initialize.
 */
void initializeGradientTrendResultSignificant(GradientTrendResultSignificant *result) {
    result->significantIncrease.startIndex = UINT16_MAX;
    result->significantIncrease.endIndex = UINT16_MAX;
    result->significantIncrease.maxValue = 0.0;
    result->significantIncrease.maxValueIndex = UINT16_MAX;
    result->significantIncrease.sumGradients = 0.0;

    result->significantDecrease.startIndex = UINT16_MAX;
    result->significantDecrease.endIndex = UINT16_MAX;
    result->significantDecrease.maxValue = 0.0;
    result->significantDecrease.maxValueIndex = UINT16_MAX;
    result->significantDecrease.sumGradients = 0.0;

    result->moveDirection = NO_TREND;
    result->isSignificantPeak = false;
    result->moveAmountSignificant = 0;

    // Initialize base pointers
    result->base.increase = &(result->significantIncrease);
    result->base.decrease = &(result->significantDecrease);
    result->base.moveDirection = &(result->moveDirection);
    result->base.isSignificantPeak = &(result->isSignificantPeak);
    result->base.moveAmount = &(result->moveAmountSignificant);
}

/**
 * @brief Finds the longest consistent increasing trend in the gradient array based on a specified threshold.
 *
 * This function scans through an array of gradient values to identify the longest consistent increasing trend
 * that exceeds a specified threshold. It allows a specified number of minor consecutive decreases to account
 * for small fluctuations. The function returns a `GradientTrendIndices` struct containing the start and end
 * indices of the longest consistent increase, the maximum gradient value within that trend,
 * and the sum of gradients within the trend.
 *
 * @param gradients The array of gradient values to analyze.
 * @param startIndex The starting index in the original data array.
 * @param windowSize The number of elements in the gradient array to analyze.
 * @param threshold The threshold above which gradients are considered significant.
 * @return GradientTrendIndices A structure containing the trend information.
 */
/**
 * @brief Finds the longest consistent increasing trend in the gradient array based on a specified threshold.
 *
 * This function scans through an array of gradient values to identify the longest consistent increasing trend
 * that exceeds a specified threshold. It allows a specified number of minor consecutive decreases to account
 * for small fluctuations. The function returns a `GradientTrendIndices` struct containing the start and end
 * indices of the longest consistent increase, the maximum gradient value within that trend,
 * and the sum of gradients within the trend.
 *
 * @param gradients   The array of gradient values to analyze.
 * @param windowSize  The number of elements in the gradient array to analyze.
 * @param threshold   The threshold above which gradients are considered significant.
 * @return GradientTrendIndices A structure containing the trend information.
 */
GradientTrendIndices findConsistentIncreaseInternal(
    const double *gradients,
    uint16_t windowSize,
    double threshold
) {
    DEBUG_PRINT_3("Entering findConsistentIncreaseInternal\n");

    // Structure to store the longest increasing trend found
    GradientTrendIndices longestIncreaseInfo;
    memset(&longestIncreaseInfo, 0, sizeof(GradientTrendIndices));
    longestIncreaseInfo.startIndex = UINT16_MAX;
    longestIncreaseInfo.endIndex = UINT16_MAX;
    longestIncreaseInfo.maxValueIndex = UINT16_MAX;

    // Structure to store the current increasing trend being tracked
    GradientTrendIndices currentIncreaseInfo;
    memset(&currentIncreaseInfo, 0, sizeof(GradientTrendIndices));
    currentIncreaseInfo.startIndex = UINT16_MAX;
    currentIncreaseInfo.endIndex = UINT16_MAX;
    currentIncreaseInfo.maxValueIndex = UINT16_MAX;

    bool trackingIncrease = false;       // Flag to indicate if we're currently tracking an increase
    int consecutiveDecreaseCount = 0;    // Counter for consecutive decreases within an increasing trend
    double currentMaxGradient = -INFINITY;  // Maximum gradient within the current increasing trend
    uint16_t currentMaxGradientIndex = UINT16_MAX; // Index of the maximum gradient within the current increasing trend

    DEBUG_PRINT_1("Threshold for Increase: %.6f\n", threshold);

    // Iterate through the gradient array within the specified window size
    for (uint16_t i = 0; i < windowSize; ++i) {
        double gradient = gradients[i];
        uint16_t currentIndex = i;

        // Ignore NaN values
        if (isnan(gradient)) continue;

        DEBUG_PRINT_2("Gradient at index %u: %.6f\n", currentIndex, gradient);

        if (gradient > threshold) {  // Significant positive gradient
            if (!trackingIncrease) {
                // Start tracking a new increasing trend
                currentIncreaseInfo.startIndex = currentIndex;
                currentIncreaseInfo.endIndex = currentIndex;
                trackingIncrease = true;
                consecutiveDecreaseCount = 0;
                currentMaxGradient = gradient;
                currentMaxGradientIndex = currentIndex;
                currentIncreaseInfo.sumGradients = gradient;

                DEBUG_PRINT_3("Started new increasing trend at index %u\n", currentIndex);
            } else {
                // Continue tracking the current increasing trend
                currentIncreaseInfo.endIndex = currentIndex;

                // Update the maximum gradient within the current trend
                if (gradient > currentMaxGradient) {
                    currentMaxGradient = gradient;
                    currentMaxGradientIndex = currentIndex;
                }

                // Reset the decrease counter since we have an increase
                consecutiveDecreaseCount = 0;

                // Add to sum of gradients
                currentIncreaseInfo.sumGradients += gradient;
            }

        } else {
            if (trackingIncrease) {
                consecutiveDecreaseCount++;

                // Add to sum of gradients even for minor decreases
                currentIncreaseInfo.sumGradients += gradient;

                if (consecutiveDecreaseCount > MAX_ALLOWED_DECREASES) {
                    // Check if the current trend is the longest so far
                    uint16_t currentTrendLength = currentIncreaseInfo.endIndex - currentIncreaseInfo.startIndex + 1;
                    uint16_t longestTrendLength = (longestIncreaseInfo.startIndex != UINT16_MAX) ?
                        (longestIncreaseInfo.endIndex - longestIncreaseInfo.startIndex + 1) : 0;

                    if (currentTrendLength > longestTrendLength) {
                        longestIncreaseInfo = currentIncreaseInfo;
                        longestIncreaseInfo.maxValue = currentMaxGradient;
                        longestIncreaseInfo.maxValueIndex = currentMaxGradientIndex;

                        DEBUG_PRINT_1("New longest increasing trend from index %u to %u\n",
                                     longestIncreaseInfo.startIndex, longestIncreaseInfo.endIndex);
                    }

                    // Stop tracking the current increasing trend
                    trackingIncrease = false;
                    consecutiveDecreaseCount = 0;

                    // Reset currentIncreaseInfo
                    memset(&currentIncreaseInfo, 0, sizeof(GradientTrendIndices));
                    currentIncreaseInfo.startIndex = UINT16_MAX;
                    currentIncreaseInfo.endIndex = UINT16_MAX;
                    currentIncreaseInfo.maxValueIndex = UINT16_MAX;
                }
            }
        }
    }

    // After iterating, check if the current trend is the longest
    if (trackingIncrease) {
        uint16_t currentTrendLength = currentIncreaseInfo.endIndex - currentIncreaseInfo.startIndex + 1;
        uint16_t longestTrendLength = (longestIncreaseInfo.startIndex != UINT16_MAX) ?
            (longestIncreaseInfo.endIndex - longestIncreaseInfo.startIndex + 1) : 0;

        if (currentTrendLength > longestTrendLength) {
            longestIncreaseInfo = currentIncreaseInfo;
            longestIncreaseInfo.maxValue = currentMaxGradient;
            longestIncreaseInfo.maxValueIndex = currentMaxGradientIndex;

            printf("Final longest increasing trend from index %u to %u\n",
                   longestIncreaseInfo.startIndex, longestIncreaseInfo.endIndex);
        }
    }

    DEBUG_PRINT_3("Exiting findConsistentIncreaseInternal\n");

    return longestIncreaseInfo;  // Return the longest increasing trend information
}


/**
 * @brief Finds the longest consistent decreasing trend in the gradient array based on a specified threshold.
 *
 * This function scans through an array of gradient values to identify the longest consistent decreasing trend
 * that exceeds a specified threshold. It allows a specified number of minor consecutive increases to account
 * for small fluctuations. The function returns a `GradientTrendIndices` struct containing the start and end
 * indices of the longest consistent decrease, the minimum gradient value within that trend,
 * and the sum of gradients within the trend.
 *
 * @param gradients The array of gradient values to analyze.
 * @param startIndex The starting index in the original data array.
 * @param windowSize The number of elements in the gradient array to analyze.
 * @param threshold The threshold below which gradients are considered significant.
 * @return GradientTrendIndices A structure containing the trend information.
 */
GradientTrendIndices findConsistentDecreaseInternal(
    const double *gradients,
    uint16_t windowSize,
    double threshold
) {
    DEBUG_PRINT_3("Entering findConsistentDecreaseInternal\n");

    // Structure to store the longest decreasing trend found
    GradientTrendIndices longestDecreaseInfo;
    memset(&longestDecreaseInfo, 0, sizeof(GradientTrendIndices));
    longestDecreaseInfo.startIndex = UINT16_MAX;
    longestDecreaseInfo.endIndex = UINT16_MAX;
    longestDecreaseInfo.maxValueIndex = UINT16_MAX;

    // Structure to store the current decreasing trend being tracked
    GradientTrendIndices currentDecreaseInfo;
    memset(&currentDecreaseInfo, 0, sizeof(GradientTrendIndices));
    currentDecreaseInfo.startIndex = UINT16_MAX;
    currentDecreaseInfo.endIndex = UINT16_MAX;
    currentDecreaseInfo.maxValueIndex = UINT16_MAX;

    bool trackingDecrease = false;        // Flag to indicate if we're currently tracking a decrease
    int consecutiveIncreaseCount = 0;     // Counter for consecutive increases within a decreasing trend
    double currentMinGradient = INFINITY; // Minimum gradient within the current decreasing trend
    uint16_t currentMinGradientIndex = UINT16_MAX; // Index of the minimum gradient within the current decreasing trend

    DEBUG_PRINT_1("Threshold for Decrease: %.6f\n", threshold);

    // Iterate through the gradient array within the specified window size
    for (uint16_t i = 0; i < windowSize; ++i) {
        double gradient = gradients[i];
        uint16_t currentIndex = i;

        // Ignore NaN values
        if (isnan(gradient)) continue;

        DEBUG_PRINT_2("Gradient at index %u: %.6f\n", currentIndex, gradient);

        if (gradient < threshold) {  // Significant negative gradient
            if (!trackingDecrease) {
                // Start tracking a new decreasing trend
                currentDecreaseInfo.startIndex = currentIndex;
                currentDecreaseInfo.endIndex = currentIndex;
                trackingDecrease = true;
                consecutiveIncreaseCount = 0;
                currentMinGradient = gradient;
                currentMinGradientIndex = currentIndex;
                currentDecreaseInfo.sumGradients = gradient;

                DEBUG_PRINT_3("Started new decreasing trend at index %u\n", currentIndex);
            } else {
                // Continue tracking the current decreasing trend
                currentDecreaseInfo.endIndex = currentIndex;

                // Update the minimum gradient within the current trend
                if (gradient < currentMinGradient) {
                    currentMinGradient = gradient;
                    currentMinGradientIndex = currentIndex;
                }

                // Reset the increase counter since we have a decrease
                consecutiveIncreaseCount = 0;

                // Add to sum of gradients
                currentDecreaseInfo.sumGradients += gradient;
            }

        } else {
            if (trackingDecrease) {
                consecutiveIncreaseCount++;

                // Add to sum of gradients even for minor increases
                currentDecreaseInfo.sumGradients += gradient;

                if (consecutiveIncreaseCount > MAX_ALLOWED_INCREASES) {
                    // Check if the current trend is the longest so far
                    uint16_t currentTrendLength = currentDecreaseInfo.endIndex - currentDecreaseInfo.startIndex + 1;
                    uint16_t longestTrendLength = (longestDecreaseInfo.startIndex != UINT16_MAX) ?
                        (longestDecreaseInfo.endIndex - longestDecreaseInfo.startIndex + 1) : 0;

                    if (currentTrendLength > longestTrendLength) {
                        longestDecreaseInfo = currentDecreaseInfo;
                        longestDecreaseInfo.maxValue = currentMinGradient;
                        longestDecreaseInfo.maxValueIndex = currentMinGradientIndex;

                        DEBUG_PRINT_1("New longest decreasing trend from index %u to %u\n",
                                     longestDecreaseInfo.startIndex, longestDecreaseInfo.endIndex);
                    }

                    // Stop tracking the current decreasing trend
                    trackingDecrease = false;
                    consecutiveIncreaseCount = 0;

                    // Reset currentDecreaseInfo
                    memset(&currentDecreaseInfo, 0, sizeof(GradientTrendIndices));
                    currentDecreaseInfo.startIndex = UINT16_MAX;
                    currentDecreaseInfo.endIndex = UINT16_MAX;
                    currentDecreaseInfo.maxValueIndex = UINT16_MAX;
                }
            }
        }
    }

    // After iterating, check if the current trend is the longest
    if (trackingDecrease) {
        uint16_t currentTrendLength = currentDecreaseInfo.endIndex - currentDecreaseInfo.startIndex + 1;
        uint16_t longestTrendLength = (longestDecreaseInfo.startIndex != UINT16_MAX) ?
            (longestDecreaseInfo.endIndex - longestDecreaseInfo.startIndex + 1) : 0;

        if (currentTrendLength > longestTrendLength) {
            longestDecreaseInfo = currentDecreaseInfo;
            longestDecreaseInfo.maxValue = currentMinGradient;
            longestDecreaseInfo.maxValueIndex = currentMinGradientIndex;

            printf("Final longest decreasing trend from index %u to %u\n",
                   longestDecreaseInfo.startIndex, longestDecreaseInfo.endIndex);
        }
    }

    DEBUG_PRINT_3("Exiting findConsistentDecreaseInternal\n");

    return longestDecreaseInfo;  // Return the longest decreasing trend information
}


// SIGNIFICANCE_FACTOR * gradCalcResultAll.mad  // Adjust threshold as needed

/**
 * @brief Identifies gradient trends within a window using RLS regression and MAD-based statistics.
 *
 * This function calculates gradients using the provided `calculate_gradient` function, identifies the
 * longest consistent increasing and decreasing trends using `findConsistentIncreaseInternal` and
 * `findConsistentDecreaseInternal`, computes separate statistical measures (median and MAD)
 * for each trend, and stores the results in the provided `gradCalcResultInc` and `gradCalcResultDec` structures.
 *
 * @param values            Array of data points to analyze.
 * @param startIndex        The starting index in the values array.
 * @param analysisLength    The number of data points to include in the gradient analysis window.
 * @param degree            The degree of the polynomial used for regression.
 * @param calculate_gradient Function pointer to the gradient calculation function.
 * @param gradCalcResultInc Pointer to store the gradient calculation results for increases.
 * @param gradCalcResultDec Pointer to store the gradient calculation results for decreases.
 *
 * @return void
 */
void identifyTrends(
    const double *values,
    uint16_t startIndex,
    uint16_t analysisLength,
    uint8_t degree,
    double (*calculate_gradient)(const RegressionState *, double),
    GradientCalculationResult *gradCalcResultInc,
    GradientCalculationResult *gradCalcResultDec
) {
    // Initialize gradCalcResult structures
    memset(gradCalcResultInc, 0, sizeof(GradientCalculationResult));
    memset(gradCalcResultDec, 0, sizeof(GradientCalculationResult));

    // Debugging: Starting the identifyTrends function
    DEBUG_PRINT_1("Starting identifyTrends with startIndex=%u, analysisLength=%u\n", startIndex, analysisLength);

    // Calculate all gradients
    GradientCalculationResult gradCalcResultAll;
    memset(&gradCalcResultAll, 0, sizeof(GradientCalculationResult));

    // Debugging: Calling trackGradients to calculate gradients
    DEBUG_PRINT_2("Calling trackGradients to calculate gradients.\n");
    trackGradients(
        values,
        analysisLength,
        startIndex,
        degree,
        calculate_gradient,
        &gradCalcResultAll
    );

    uint16_t count = gradCalcResultAll.size;

    // Debugging: Number of gradients calculated
    DEBUG_PRINT_1("Total gradients calculated: %u\n", count);

    if (count == 0) {
        DEBUG_PRINT_1("No gradients calculated. Exiting identifyTrends.\n");
        return;
    }

    // Debugging: Finding the longest consistent increasing trend
    DEBUG_PRINT_1("Finding the longest consistent increasing trend.\n");
    GradientTrendIndices longestIncrease = findConsistentIncreaseInternal(
        gradCalcResultAll.gradients,
        count,
        0.0 // Adjust threshold as needed
    );

    // Adjust indices to absolute positions for the increasing trend
    if (longestIncrease.startIndex != UINT16_MAX) {
        DEBUG_PRINT_2("Adjusting indices of the longest increasing trend to absolute positions.\n");
        DEBUG_PRINT_3("Before adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestIncrease.startIndex, longestIncrease.endIndex, longestIncrease.maxValueIndex);

        longestIncrease.startIndex += startIndex;
        longestIncrease.endIndex += startIndex;
        longestIncrease.maxValueIndex += startIndex;

        DEBUG_PRINT_3("After adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestIncrease.startIndex, longestIncrease.endIndex, longestIncrease.maxValueIndex);
                      
        printf("INCREASE After adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestIncrease.startIndex, longestIncrease.endIndex, longestIncrease.maxValueIndex);
    } else {
        DEBUG_PRINT_1("No increasing trend found.\n");
    }

    // Calculate median and MAD for the longest increasing trend
    if (longestIncrease.startIndex != UINT16_MAX && longestIncrease.endIndex != UINT16_MAX) {
        uint16_t relativeStart = longestIncrease.startIndex - startIndex;
        uint16_t incLength = longestIncrease.endIndex - longestIncrease.startIndex + 1;

        DEBUG_PRINT_2("Calculating median and MAD for the longest increasing trend.\n");
        DEBUG_PRINT_3("Relative start index in gradients array: %u\n", relativeStart);
        DEBUG_PRINT_3("Length of increasing trend: %u\n", incLength);

        if (incLength > 0 && (relativeStart + incLength <= count)) {
            gradCalcResultInc->size = incLength;
            memcpy(gradCalcResultInc->gradients, &gradCalcResultAll.gradients[relativeStart], incLength * sizeof(double));

            calculateMedianAndMAD(
                gradCalcResultInc->gradients,
                gradCalcResultInc->size,
                &gradCalcResultInc->median,
                &gradCalcResultInc->mad
            );

            gradCalcResultInc->startIndex = longestIncrease.startIndex;
            gradCalcResultInc->endIndex = longestIncrease.endIndex;

            // Debugging: Display calculated median and MAD for the increasing trend
            DEBUG_PRINT_2("Median of increasing trend gradients: %.6f\n", gradCalcResultInc->median);
            DEBUG_PRINT_2("MAD of increasing trend gradients: %.6f\n", gradCalcResultInc->mad);
        } else {
            DEBUG_PRINT_1("Invalid indices for increasing trend. Skipping median and MAD calculation.\n");
        }
    }

    // Debugging: Finding the longest consistent decreasing trend
    DEBUG_PRINT_1("Finding the longest consistent decreasing trend.\n");
    GradientTrendIndices longestDecrease = findConsistentDecreaseInternal(
        gradCalcResultAll.gradients,
        count,
        0.0 // Adjust threshold as needed
    );

    // Adjust indices to absolute positions for the decreasing trend
    if (longestDecrease.startIndex != UINT16_MAX) {
        DEBUG_PRINT_2("Adjusting indices of the longest decreasing trend to absolute positions.\n");
        DEBUG_PRINT_3("Before adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestDecrease.startIndex, longestDecrease.endIndex, longestDecrease.maxValueIndex);

        longestDecrease.startIndex += startIndex;
        longestDecrease.endIndex += startIndex;
        longestDecrease.maxValueIndex += startIndex;

        DEBUG_PRINT_3("After adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestDecrease.startIndex, longestDecrease.endIndex, longestDecrease.maxValueIndex);
                      
        DEBUG_PRINT_3("DECREASE After adjustment: startIndex=%u, endIndex=%u, maxValueIndex=%u\n",
                      longestDecrease.startIndex, longestDecrease.endIndex, longestDecrease.maxValueIndex);              
    } else {
        DEBUG_PRINT_1("No decreasing trend found.\n");
    }

    // Calculate median and MAD for the longest decreasing trend
    if (longestDecrease.startIndex != UINT16_MAX && longestDecrease.endIndex != UINT16_MAX) {
        uint16_t relativeStart = longestDecrease.startIndex - startIndex;
        uint16_t decLength = longestDecrease.endIndex - longestDecrease.startIndex + 1;

        DEBUG_PRINT_2("Calculating median and MAD for the longest decreasing trend.\n");
        DEBUG_PRINT_3("Relative start index in gradients array: %u\n", relativeStart);
        DEBUG_PRINT_3("Length of decreasing trend: %u\n", decLength);

        if (decLength > 0 && (relativeStart + decLength <= count)) {
            gradCalcResultDec->size = decLength;
            memcpy(gradCalcResultDec->gradients, &gradCalcResultAll.gradients[relativeStart], decLength * sizeof(double));

            calculateMedianAndMAD(
                gradCalcResultDec->gradients,
                gradCalcResultDec->size,
                &gradCalcResultDec->median,
                &gradCalcResultDec->mad
            );

            gradCalcResultDec->startIndex = longestDecrease.startIndex;
            gradCalcResultDec->endIndex = longestDecrease.endIndex;

            // Debugging: Display calculated median and MAD for the decreasing trend
            DEBUG_PRINT_2("Median of decreasing trend gradients: %.6f\n", gradCalcResultDec->median);
            DEBUG_PRINT_2("MAD of decreasing trend gradients: %.6f\n", gradCalcResultDec->mad);
        } else {
            DEBUG_PRINT_1("Invalid indices for decreasing trend. Skipping median and MAD calculation.\n");
        }
    }

    // Debugging: Completed the identifyTrends function
    DEBUG_PRINT_1("Completed identifyTrends.\n");
}

/**
 * @brief Adjusts the window start index based on the move amount.
 *
 * This function calculates a new start index for the analysis window by adding the move amount
 * to the current start index. It ensures that the new start index stays within the bounds
 * of the dataset, preventing out-of-bounds access.
 *
 * @param currentStartIndex The current start index of the window.
 * @param moveAmount The amount to move the window (positive or negative).
 * @param dataLength The total length of the data array.
 * @param windowSize The size of the window.
 * @return uint16_t The new start index after adjustment.
 */
uint16_t adjustWindowStartIndex(
    uint16_t currentStartIndex,
    int16_t moveAmount,
    uint16_t dataLength,
    uint16_t windowSize
) {
    int32_t newStartIndex = (int32_t)currentStartIndex + moveAmount;

    if (newStartIndex < 0) {
        newStartIndex = 0;
    } else if ((uint32_t)newStartIndex + windowSize > dataLength) {
        if (dataLength >= windowSize) {
            newStartIndex = dataLength - windowSize;
        } else {
            newStartIndex = 0;  // If dataLength < windowSize, start at 0
        }
    }

    return (uint16_t)newStartIndex;
}

/**
 * @brief Detects trends (absolute or significant) and calculates move amounts using separate statistics for increases and decreases.
 *
 * This function analyzes the gradient calculation results to detect increasing and decreasing trends.
 * It uses separate medians and MADs for increasing and decreasing trends to determine significant trends.
 * The function populates a trend result structure with the detected trends and move amounts.
 *
 * @param gradCalcResultInc Pointer to the gradient calculation results for increases.
 * @param gradCalcResultDec Pointer to the gradient calculation results for decreases.
 * @param trendResultBase Pointer to the base of the trend result structure.
 * @param trendType Specifies whether to detect absolute or significant trends.
 */
void detectTrends(
    const GradientCalculationResult *gradCalcResultInc,
    const GradientCalculationResult *gradCalcResultDec,
    GradientTrendResultBase *trendResultBase,
    TrendDetectionType trendType
) {
    DEBUG_PRINT_3("Detecting trends.\n");
    
    // Initialize thresholds
    double thresholdIncrease = 0.0;
    double thresholdDecrease = 0.0;
    
    // Depending on trendType, set thresholds
    if (trendType == TREND_TYPE_ABSOLUTE) {
        thresholdIncrease = gradCalcResultInc->median;
        thresholdDecrease = gradCalcResultDec->median;
    } else if (trendType == TREND_TYPE_SIGNIFICANT) {
        thresholdIncrease = gradCalcResultInc->median + (2 * gradCalcResultInc->mad);
        thresholdDecrease = gradCalcResultDec->median - (2 * gradCalcResultDec->mad);
    } else {
        DEBUG_PRINT_1("Invalid trend detection type specified.\n");
        return;
    }
    
    // Initialize move amount
    *(trendResultBase->moveAmount) = 0;
    
    // Detect increases
    if (gradCalcResultInc->size > 0) {
        for (uint16_t i = 0; i < gradCalcResultInc->size; ++i) {
            double gradient = gradCalcResultInc->gradients[i];
            if (gradient >= thresholdIncrease) {
                // Detected an increase
                if (trendResultBase->increase->startIndex == UINT16_MAX &&
                    trendResultBase->increase->endIndex == UINT16_MAX &&
                    trendResultBase->increase->sumGradients == 0.0) {
                    trendResultBase->increase->startIndex = i;
                    trendResultBase->increase->endIndex = i;
                    trendResultBase->increase->maxValue = gradient;
                    trendResultBase->increase->maxValueIndex = i;
                    trendResultBase->increase->sumGradients = gradient;
                } else {
                    trendResultBase->increase->endIndex = i;
                    trendResultBase->increase->sumGradients += gradient;
                    if (gradient > trendResultBase->increase->maxValue) {
                        trendResultBase->increase->maxValue = gradient;
                        trendResultBase->increase->maxValueIndex = i;
                    }
                }
    
                // Increment move amount
                (*(trendResultBase->moveAmount))++;
    
                DEBUG_PRINT_3("Detected increase at index %u.\n", i);
            }
        }
    }
    
    // Detect decreases
    if (gradCalcResultDec->size > 0) {
        for (uint16_t i = 0; i < gradCalcResultDec->size; ++i) {
            double gradient = gradCalcResultDec->gradients[i];
            if (gradient <= thresholdDecrease) {
                // Detected a decrease
                if (trendResultBase->decrease->startIndex == UINT16_MAX &&
                    trendResultBase->decrease->endIndex == UINT16_MAX &&
                    trendResultBase->decrease->sumGradients == 0.0) {
                    trendResultBase->decrease->startIndex = i;
                    trendResultBase->decrease->endIndex = i;
                    trendResultBase->decrease->maxValue = gradient;
                    trendResultBase->decrease->maxValueIndex = i;
                    trendResultBase->decrease->sumGradients = gradient;
                } else {
                    trendResultBase->decrease->endIndex = i;
                    trendResultBase->decrease->sumGradients += gradient;
                    if (gradient < trendResultBase->decrease->maxValue) {
                        trendResultBase->decrease->maxValue = gradient;
                        trendResultBase->decrease->maxValueIndex = i;
                    }
                }
    
                // Increment move amount
                (*(trendResultBase->moveAmount))++;
    
                DEBUG_PRINT_3("Detected decrease at index %u.\n", i);
            }
        }
    }
    
    // Determine if a significant peak exists
    if (*(trendResultBase->moveAmount) > 0) {
        *(trendResultBase->isSignificantPeak) = true;
    }
    
    DEBUG_PRINT_2("Trend detection completed. Move amount: %d\n", *(trendResultBase->moveAmount));
}

/**
 * @brief Identifies gradient trends within a specified window.
 *
 * This function detects consistent increasing and decreasing trends based on first-order or second-order gradients
 * calculated using RLS regression. It delegates the detection to specific functions based on the
 * trend detection type (absolute, significant, or gradient trends).
 *
 * @param values             Array of data points to analyze.
 * @param startIndex         The starting index in the values array.
 * @param analysisLength     The number of data points to include in the gradient analysis window.
 * @param degree             The degree of the polynomial used for regression.
 * @param gradientOrder      The order of the gradient (first-order or second-order).
 * @param trendType          The type of trend detection to perform.
 * @param trendResult        Pointer to store the trend detection result.
 * @param gradCalcResultInc  Pointer to store the gradient calculation results for increases.
 * @param gradCalcResultDec  Pointer to store the gradient calculation results for decreases.
 */
void identifyGradientTrends(
    const double *values,
    uint16_t startIndex,
    uint16_t analysisLength,
    uint8_t degree,
    GradientOrder gradientOrder,
    TrendDetectionType trendType,
    void *trendResult,
    GradientCalculationResult *gradCalcResultInc,
    GradientCalculationResult *gradCalcResultDec
) {
    // Determine the gradient calculation function based on the gradient order
    double (*calculate_gradient)(const RegressionState *, double) = NULL;
    const char* gradientOrderStr = NULL;

    switch (gradientOrder) {
        case GRADIENT_ORDER_FIRST:
            calculate_gradient = calculate_first_order_gradient;
            gradientOrderStr = "First-Order";
            break;
        case GRADIENT_ORDER_SECOND:
            calculate_gradient = calculate_second_order_gradient;
            gradientOrderStr = "Second-Order";
            break;
        default:
            DEBUG_PRINT_1("Invalid gradient order specified.\n");
            return;
    }

    // Calculate gradients using the specified gradient function and identify trends
    printf("=== Calculating %s Gradients ===\n", gradientOrderStr);
    identifyTrends(
        values,
        startIndex,
        analysisLength,
        degree,
        calculate_gradient,
        gradCalcResultInc,
        gradCalcResultDec
    );

    // Check if any gradients have been calculated
    if (gradCalcResultInc->size == 0 && gradCalcResultDec->size == 0) {
        DEBUG_PRINT_1("No gradients detected, skipping trend detection.\n");
        return;
    }

    // Handle different trend detection types using switch-case
    switch (trendType) {
        case TREND_TYPE_ABSOLUTE:
        case TREND_TYPE_SIGNIFICANT: {
            // Determine which result structure to use based on trendType
            GradientTrendResultBase *trendResultBase = NULL;
            if (trendType == TREND_TYPE_ABSOLUTE) {
                GradientTrendResultAbsolute *absResult = (GradientTrendResultAbsolute *)trendResult;
                initializeGradientTrendResultAbsolute(absResult);
                trendResultBase = &(absResult->base);
                printf("=== Detecting Absolute %s Gradient Trends ===\n", gradientOrderStr);
            } else { // TREND_TYPE_SIGNIFICANT
                GradientTrendResultSignificant *sigResult = (GradientTrendResultSignificant *)trendResult;
                initializeGradientTrendResultSignificant(sigResult);
                trendResultBase = &(sigResult->base);
                printf("=== Detecting Significant %s Gradient Trends ===\n", gradientOrderStr);
            }

            // Detect trends using the combined function
            detectTrends(
                gradCalcResultInc,
                gradCalcResultDec,
                trendResultBase,
                trendType
            );

            // Decide move direction based on detected trends
            printf("Step: Deciding move direction based on detected %s gradients.\n", gradientOrderStr);
            if (*(trendResultBase->moveAmount) > 0) {
                if (gradCalcResultInc->size > gradCalcResultDec->size) {
                    *(trendResultBase->moveDirection) = MOVE_RIGHT;
                } else if (gradCalcResultInc->size < gradCalcResultDec->size) {
                    *(trendResultBase->moveDirection) = MOVE_LEFT;
                } else {
                    *(trendResultBase->moveDirection) = ON_THE_PEAK;
                }
            } else {
                *(trendResultBase->moveDirection) = NO_TREND;
            }

            // Print results based on trendType
            if (trendType == TREND_TYPE_ABSOLUTE) {
                GradientTrendResultAbsolute *absResult = (GradientTrendResultAbsolute *)trendResult;
                printf("Absolute %s Gradient Trends:\n", gradientOrderStr);
                printf("Move amount: %d\n", absResult->moveAmountAbsolute);
                printTrendDetails("Absolute Increase", &absResult->absoluteIncrease);
                printTrendDetails("Absolute Decrease", &absResult->absoluteDecrease);
                printf("Move Direction: %s\n\n",
                       absResult->moveDirection == MOVE_RIGHT ? "MOVE_RIGHT" :
                       (absResult->moveDirection == MOVE_LEFT ? "MOVE_LEFT" :
                        (absResult->moveDirection == ON_THE_PEAK ? "ON_THE_PEAK" : "NO_TREND")));
            } else { // TREND_TYPE_SIGNIFICANT
                GradientTrendResultSignificant *sigResult = (GradientTrendResultSignificant *)trendResult;
                printf("Significant %s Gradient Trends:\n", gradientOrderStr);
                printf("Move amount: %d\n", sigResult->moveAmountSignificant);
                printTrendDetails("Significant Increase", &sigResult->significantIncrease);
                printTrendDetails("Significant Decrease", &sigResult->significantDecrease);
                printf("Move Direction: %s\n\n",
                       sigResult->moveDirection == MOVE_RIGHT ? "MOVE_RIGHT" :
                       (sigResult->moveDirection == MOVE_LEFT ? "MOVE_LEFT" :
                        (sigResult->moveDirection == ON_THE_PEAK ? "ON_THE_PEAK" : "NO_TREND")));
            }

            break;
        }

        case TREND_TYPE_GRADIENT_TRENDS: {
            GradientTrendResultAbsolute *gradResult = (GradientTrendResultAbsolute *)trendResult;
            initializeGradientTrendResultAbsolute(gradResult);

            printf("=== Detecting %s Gradient Trends ===\n", gradientOrderStr);
            printf("Step: Extracting consistent increases and decreases for %s gradients.\n", gradientOrderStr);

            // Construct absoluteIncrease from gradCalcResultInc
            if (gradCalcResultInc->size > 0) {
                gradResult->absoluteIncrease.startIndex = gradCalcResultInc->startIndex;
                gradResult->absoluteIncrease.endIndex = gradCalcResultInc->endIndex;

                // Find maxValue and maxValueIndex for increase
                double maxValue = -INFINITY;
                uint16_t maxValueIndex = UINT16_MAX;
                for (uint16_t i = 0; i < gradCalcResultInc->size; ++i) {
                    double gradient = gradCalcResultInc->gradients[i];
                    if (gradient > maxValue) {
                        maxValue = gradient;
                        maxValueIndex = gradCalcResultInc->startIndex + i; // Adjust to original data index
                    }
                }
                gradResult->absoluteIncrease.maxValue = maxValue;
                gradResult->absoluteIncrease.maxValueIndex = maxValueIndex;

                // Sum of gradients for increase
                double sumGradients = 0.0;
                for (uint16_t i = 0; i < gradCalcResultInc->size; ++i) {
                    sumGradients += gradCalcResultInc->gradients[i];
                }
                gradResult->absoluteIncrease.sumGradients = sumGradients;
            } else {
                // No increasing trend detected
                gradResult->absoluteIncrease.startIndex = UINT16_MAX;
                gradResult->absoluteIncrease.endIndex = UINT16_MAX;
                gradResult->absoluteIncrease.maxValue = 0.0;
                gradResult->absoluteIncrease.maxValueIndex = UINT16_MAX;
                gradResult->absoluteIncrease.sumGradients = 0.0;
            }

            // Construct absoluteDecrease from gradCalcResultDec
            if (gradCalcResultDec->size > 0) {
                gradResult->absoluteDecrease.startIndex = gradCalcResultDec->startIndex;
                gradResult->absoluteDecrease.endIndex = gradCalcResultDec->endIndex;

                // Find minValue and minValueIndex for decrease
                double minValue = INFINITY;
                uint16_t minValueIndex = UINT16_MAX;
                for (uint16_t i = 0; i < gradCalcResultDec->size; ++i) {
                    double gradient = gradCalcResultDec->gradients[i];
                    if (gradient < minValue) {
                        minValue = gradient;
                        minValueIndex = gradCalcResultDec->startIndex + i; // Adjust to original data index
                    }
                }
                gradResult->absoluteDecrease.maxValue = minValue;
                gradResult->absoluteDecrease.maxValueIndex = minValueIndex;

                // Sum of gradients for decrease
                double sumGradients = 0.0;
                for (uint16_t i = 0; i < gradCalcResultDec->size; ++i) {
                    sumGradients += gradCalcResultDec->gradients[i];
                }
                gradResult->absoluteDecrease.sumGradients = sumGradients;
            } else {
                // No decreasing trend detected
                gradResult->absoluteDecrease.startIndex = UINT16_MAX;
                gradResult->absoluteDecrease.endIndex = UINT16_MAX;
                gradResult->absoluteDecrease.maxValue = 0.0;
                gradResult->absoluteDecrease.maxValueIndex = UINT16_MAX;
                gradResult->absoluteDecrease.sumGradients = 0.0;
            }

            // Decide move direction based on the detected trends
            printf("Step: Deciding move direction based on detected %s gradients.\n", gradientOrderStr);
            if (gradResult->absoluteIncrease.startIndex != UINT16_MAX && gradResult->absoluteDecrease.startIndex == UINT16_MAX) {
                gradResult->moveDirection = MOVE_RIGHT;
            } else if (gradResult->absoluteIncrease.startIndex == UINT16_MAX && gradResult->absoluteDecrease.startIndex != UINT16_MAX) {
                gradResult->moveDirection = MOVE_LEFT;
            } else if (gradResult->absoluteIncrease.startIndex != UINT16_MAX && gradResult->absoluteDecrease.startIndex != UINT16_MAX) {
                // Both trends are valid; decide based on which has a longer consistent trend
                uint16_t increaseTrendLength = gradResult->absoluteIncrease.endIndex - gradResult->absoluteIncrease.startIndex + 1;
                uint16_t decreaseTrendLength = gradResult->absoluteDecrease.endIndex - gradResult->absoluteDecrease.startIndex + 1;

                if (increaseTrendLength > decreaseTrendLength) {
                    gradResult->moveDirection = MOVE_RIGHT;
                } else if (increaseTrendLength < decreaseTrendLength) {
                    gradResult->moveDirection = MOVE_LEFT;
                } else {
                    gradResult->moveDirection = ON_THE_PEAK;
                }
            } else {
                gradResult->moveDirection = NO_TREND;
            }

            // Determine if a significant peak exists
            gradResult->isSignificantPeak = (gradResult->moveDirection == ON_THE_PEAK);

            // Print details about the detected increases
            printTrendDetails("Increase", &gradResult->absoluteIncrease);

            // Print details about the detected decreases
            printTrendDetails("Decrease", &gradResult->absoluteDecrease);

            // Print move direction
            printf("Move Direction based on %s gradients: ", gradientOrderStr);
            switch (gradResult->moveDirection) {
                case MOVE_RIGHT:
                    printf("MOVE_RIGHT\n");
                    break;
                case MOVE_LEFT:
                    printf("MOVE_LEFT\n");
                    break;
                case ON_THE_PEAK:
                    printf("ON_THE_PEAK\n");
                    break;
                case NO_TREND:
                    printf("NO_TREND\n");
                    break;
                default:
                    printf("UNKNOWN_DIRECTION\n");
                    break;
            }

            // Additional separator for clarity in output
            printf("============================================\n");
            break;
        }

        default:
            DEBUG_PRINT_1("Invalid trend detection type specified.\n");
            break;
    }
}

/**
 * @brief Identifies the move direction based on gradient trends by splitting the gradient array into halves.
 *
 * This function calculates the gradients using `trackGradients`, splits the gradient array into two halves,
 * assesses each part to determine the direction towards the impedance peak, and returns the corresponding move direction.
 *
 * @param values          Array of data points to analyze.
 * @param startIndex      The starting index in the values array.
 * @param analysisLength  The number of data points to include in the gradient analysis window.
 * @param degree          The degree of the polynomial used for regression.
 * @param gradientOrder   The order of the gradient (first-order or second-order).
 * @param trendType       The type of trend detection to perform (currently unused in this implementation).
 * @param trendResult     Pointer to store the trend detection result (optional; not used here).
 *
 * @return MoveDirection The determined move direction based on trend analysis.
 */
MoveDirection identifySplitMoveDirection(
    const double *values,
    uint16_t startIndex,
    uint16_t analysisLength,
    uint8_t degree,
    GradientOrder gradientOrder
) {
    // Initialize gradient calculation results
    GradientCalculationResult gradCalcResultAll;
    memset(&gradCalcResultAll, 0, sizeof(GradientCalculationResult));

    // Determine the gradient calculation function based on gradientOrder
    double (*calculate_gradient)(const RegressionState *, double) = NULL;
    const char* gradientOrderStr = NULL;

    switch (gradientOrder) {
        case GRADIENT_ORDER_FIRST:
            calculate_gradient = calculate_first_order_gradient;
            gradientOrderStr = "First-Order";
            break;
        case GRADIENT_ORDER_SECOND:
            calculate_gradient = calculate_second_order_gradient;
            gradientOrderStr = "Second-Order";
            break;
        default:
            DEBUG_PRINT_1("Invalid gradient order specified.\n");
            return NO_TREND;
    }

    printf("=== Calculating %s Gradients ===\n", gradientOrderStr);

    // Calculate gradients using trackGradients
    trackGradients(
        values,
        analysisLength,
        startIndex,
        degree,
        calculate_gradient,
        &gradCalcResultAll
    );

    // Check if any gradients have been calculated
    if (gradCalcResultAll.size == 0) {
        DEBUG_PRINT_1("No gradients calculated. Exiting identifySplitMoveDirection.\n");
        return NO_TREND;
    }

    printf("Total gradients calculated: %u\n", gradCalcResultAll.size);

    // Split the gradient array into two halves
    uint16_t midIndex = gradCalcResultAll.size / 2;
    uint16_t leftSize = midIndex;
    uint16_t rightSize = gradCalcResultAll.size - midIndex;

    double leftSum = 0.0;
    double rightSum = 0.0;

    // Calculate sum of gradients in the left half
    for (uint16_t i = 0; i < leftSize; ++i) {
        leftSum += gradCalcResultAll.gradients[i];
    }

    // Calculate sum of gradients in the right half
    for (uint16_t i = midIndex; i < gradCalcResultAll.size; ++i) {
        rightSum += gradCalcResultAll.gradients[i];
    }

    DEBUG_PRINT_2("Left Sum: %.6f, Right Sum: %.6f\n", leftSum, rightSum);

    // Determine move direction based on sums
    if (leftSum > rightSum) {
        // Gradients are increasing towards the right; move window to the right
        printf("Determined Move Direction: MOVE_RIGHT\n");
        return MOVE_RIGHT;
    } else if (rightSum > leftSum) {
        // Gradients are increasing towards the left; move window to the left
        printf("Determined Move Direction: MOVE_LEFT\n");
        return MOVE_LEFT;
    } else {
        // Gradients are balanced; assume window is at the peak
        printf("Determined Move Direction: ON_THE_PEAK\n");
        return ON_THE_PEAK;
    }
}


/**
 * @brief Wrapper function that identifies trends and detects the move direction.
 *
 * This function calculates the gradients using RLS polynomial regression by calling `identifyTrends`,
 * and then it uses those gradients to determine the move direction by calling `detectTrends`.
 * It encapsulates the entire process of gradient calculation and trend detection, simplifying the usage.
 *
 * @param values             Array of data points to analyze.
 * @param startIndex         The starting index in the values array.
 * @param analysisLength     The number of data points to include in the gradient analysis window.
 * @param degree             The degree of the polynomial used for regression.
 * @param trendType          The type of trend detection to perform.
 * @param trendResult        Pointer to store the trend detection result.
 * @param calculate_gradient Function pointer to the gradient calculation function.
 *
 * @return MoveDirection The determined move direction based on trend analysis.
 */
/* 
MoveDirection identifyAndDetectTrends( //şu anki hali anlamsız. 
    const double *values,
    uint16_t startIndex,
    uint16_t analysisLength,
    uint8_t degree,
    GradientOrder gradientOrder,
    TrendDetectionType trendType,
    void *trendResult
) {
    GradientCalculationResult gradCalcResultInc;
    GradientCalculationResult gradCalcResultDec;

    // Initialize the gradient calculation results
    memset(&gradCalcResultInc, 0, sizeof(GradientCalculationResult));
    memset(&gradCalcResultDec, 0, sizeof(GradientCalculationResult));

    // Identify gradients and trends
    identifyGradientTrends(
        values,
        startIndex,
        analysisLength,
        degree,
        gradientOrder,
        trendType,
        trendResult,
        &gradCalcResultInc,
        &gradCalcResultDec
    );

    // Check if any gradients have been calculated
    if (gradCalcResultInc.size == 0 && gradCalcResultDec.size == 0) {
        DEBUG_PRINT_1("No gradients calculated. Exiting identifyAndDetectTrends.\n");
        return NO_TREND;
    }

    // Combine the gradients from both increasing and decreasing trends
    double combinedGradients[MAX_WINDOW_SIZE * 2]; // Ensure the array is large enough
    uint16_t totalSize = 0;

    if (gradCalcResultInc.size > 0) {
        memcpy(&combinedGradients[totalSize], gradCalcResultInc.gradients, gradCalcResultInc.size * sizeof(double));
        totalSize += gradCalcResultInc.size;
    }
    if (gradCalcResultDec.size > 0) {
        memcpy(&combinedGradients[totalSize], gradCalcResultDec.gradients, gradCalcResultDec.size * sizeof(double));
        totalSize += gradCalcResultDec.size;
    }

    // Define thresholds based on median and MAD
    double thresholdIncrease = (gradCalcResultInc.size > 0) ? (gradCalcResultInc.median + gradCalcResultInc.mad) : 0.0;
    double thresholdDecrease = (gradCalcResultDec.size > 0) ? (gradCalcResultDec.median - gradCalcResultDec.mad) : 0.0;

    // Call detectTrends to get a move direction
    MoveDirection moveDirectionDetected = detectTrends(
        combinedGradients,
        totalSize,
        thresholdIncrease,
        thresholdDecrease
    );

    return moveDirectionDetected;
}
*/
