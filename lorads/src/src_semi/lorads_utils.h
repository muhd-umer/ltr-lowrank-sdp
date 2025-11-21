/**
 * @file lorads_utils.h
 * @brief LORADS Utility Functions and Macros
 * @details This header file provides utility functions and macros for memory management,
 * error handling, mathematical operations, and matrix manipulations used throughout LORADS.
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_UTILS_H
#define LORADS_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lorads.h"

/* Define macros */
/**
 * @def LORADS_ERROR_TRACE
 * @brief Prints error trace with file name and line number.
 */
#define LORADS_ERROR_TRACE printf("File [%30s] Line [%d]\n", __FILE__, __LINE__)

/**
 * @def LORADS_FREE
 * @brief Safely frees memory and sets pointer to NULL.
 * @param vars Pointer to memory to be freed.
 */
#define LORADS_FREE(vars)    \
    do                    \
    {                     \
        if (vars)          \
        {                 \
            free((vars));  \
            (vars) = NULL; \
        }                 \
    } while (0)

/**
 * @def LORADS_INIT
 * @brief Allocates and initializes memory with zeros.
 * @param var Variable to be allocated.
 * @param type Data type.
 * @param size Number of elements.
 */
#define LORADS_INIT(var, type, size) (var) = (type *)calloc(size, sizeof(type))

/**
 * @def LORADS_REALLOC
 * @brief Reallocates memory with new size.
 * @param var Variable to be reallocated.
 * @param type Data type.
 * @param size New number of elements.
 */
#define LORADS_REALLOC(var, type, size) (var) = (type *)realloc(var, sizeof(type) * (size))

/**
 * @def LORADS_MEMCPY
 * @brief Copies memory from source to destination.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param type Data type.
 * @param size Number of elements.
 */
#define LORADS_MEMCPY(dst, src, type, size) memcpy(dst, src, sizeof(type) * (size))

/**
 * @def LORADS_ZERO
 * @brief Initializes memory with zeros.
 * @param var Variable to be zeroed.
 * @param type Data type.
 * @param size Number of elements.
 */
#define LORADS_ZERO(var, type, size) memset(var, 0, sizeof(type) * (size))

/**
 * @def LORADS_NULLCHECK
 * @brief Checks if pointer is NULL and prints error trace.
 * @param var Pointer to check.
 */
#define LORADS_NULLCHECK(var)            \
    if (!(var))                        \
    {                                  \
        LORADS_ERROR_TRACE;             \
    }

/**
 * @def LORADS_MEMCHECK
 * @brief Checks if pointer is NULL and prints error trace.
 * @param var Pointer to check.
 */
#define LORADS_MEMCHECK(var)             \
    if (!(var))                        \
    {                                    \
        LORADS_ERROR_TRACE;                                 \
    }

/**
 * @def LORADS_MAX
 * @brief Returns maximum of two values.
 * @param x First value.
 * @param y Second value.
 * @return Maximum value.
 */
#define LORADS_MAX(x, y) ((x) > (y) ? (x) : (y))

/**
 * @def LORADS_MIN
 * @brief Returns minimum of two values.
 * @param x First value.
 * @param y Second value.
 * @return Minimum value.
 */
#define LORADS_MIN(x, y) ((x) < (y) ? (x) : (y))

/**
 * @def LORADS_ABS
 * @brief Returns absolute value.
 * @param x Input value.
 * @return Absolute value.
 */
#define LORADS_ABS(x) fabs(x)

/**
 * @def LORADS_DIMAC_ERROR_CONSTRVIO_L1
 * @brief DIMAC error indices for constraint violation (L1 norm).
 */
#define LORADS_DIMAC_ERROR_CONSTRVIO_L1 (0)

/**
 * @def LORADS_DIMAC_ERROR_PDGAP
 * @brief DIMAC error indices for primal-dual gap.
 */
#define LORADS_DIMAC_ERROR_PDGAP (1)

/**
 * @def LORADS_DIMAC_ERROR_DUALFEASIBLE_L1
 * @brief DIMAC error indices for dual feasibility (L1 norm).
 */
#define LORADS_DIMAC_ERROR_DUALFEASIBLE_L1 (2)

/**
 * @def PI
 * @brief Value of pi.
 */
#define PI (3.1415926)

/**
 * @def PACK_NNZ
 * @brief Calculates number of non-zero elements in packed symmetric matrix.
 * @param n Matrix dimension.
 * @return Number of non-zero elements.
 */
#define PACK_NNZ(n) ((n) * ((n) + 1) / 2)

/**
 * @def PACK_IDX
 * @brief Calculates index in packed symmetric matrix.
 * @param n Matrix dimension.
 * @param i Row index.
 * @param j Column index.
 * @return Index in packed array.
 */
#define PACK_IDX(n, i, j) (lorads_int)((2 * (n) - (j)-1) * (j) / 2) + (i)

/**
 * @def FULL_IDX
 * @brief Calculates index in full matrix.
 * @param n Matrix dimension.
 * @param i Row index.
 * @param j Column index.
 * @return Index in full array.
 */
#define FULL_IDX(n, i, j) ((j) * (n) + (i))

/**
 * @def PACK_ENTRY
 * @brief Accesses element in packed symmetric matrix.
 * @param A Matrix array.
 * @param n Matrix dimension.
 * @param i Row index.
 * @param j Column index.
 * @return Matrix element.
 */
#define PACK_ENTRY(A, n, i, j) (A[(lorads_int)((2 * (n) - (j)-1) * (j) / 2) + (i)])

/**
 * @def FULL_ENTRY
 * @brief Accesses element in full matrix.
 * @param A Matrix array.
 * @param n Matrix dimension.
 * @param i Row index.
 * @param j Column index.
 * @return Matrix element.
 */
#define FULL_ENTRY(A, n, i, j) (A[(j) * (n) + (i)])

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Gets current timestamp
 * @return Current timestamp in seconds
 */
extern double LUtilGetTimeStamp(void);

/**
 * @brief Symmetrizes a matrix
 * @param n Matrix dimension
 * @param v Matrix array
 */
extern void LUtilMatSymmetrize(lorads_int n, double *v);

/**
 * @brief Checks if array is in ascending order
 * @param n Array length
 * @param idx Array to check
 * @return 1 if ascending, 0 otherwise
 */
extern lorads_int LUtilCheckIfAscending(lorads_int n, lorads_int *idx);

/**
 * @brief Sorts integer array in descending order based on reference array
 * @param data Array to sort
 * @param ref Reference array
 * @param low Start index
 * @param up End index
 */
extern void LUtilDescendSortIntByInt(lorads_int *data, lorads_int *ref, lorads_int low, lorads_int up);

/**
 * @brief Sorts integer array based on double reference array
 * @param data Array to sort
 * @param ref Reference array
 * @param low Start index
 * @param up End index
 */
extern void LUtilSortIntbyDbl(lorads_int *data, double *ref, lorads_int low, lorads_int up);
extern void LUtilAscendSortDblByInt(double *data, lorads_int *ref, lorads_int low, lorads_int up);

/**
 * @brief Sorts double array in ascending order based on integer reference array
 * @param data Array to sort
 * @param ref Reference array
 * @param low Start index
 * @param up End index
 */
extern void LUtilAscendSortDblByInt(double *data, lorads_int *ref, lorads_int low, lorads_int up);

/**
 * @brief Starts checking for Ctrl+C interrupt
 */
extern void LUtilStartCtrlCCheck(void);

/**
 * @brief Checks if Ctrl+C was pressed
 * @return 1 if Ctrl+C was pressed, 0 otherwise
 */
extern lorads_int LUtilCheckCtrlC(void);

/**
 * @brief Resets Ctrl+C check
 */
extern void LUtilResetCtrl(void);

/**
 * @brief Gets number of MKL threads
 * @return Number of threads
 */
extern lorads_int LUtilGetGlobalMKLThreads(void);

/**
 * @brief Sets number of MKL threads
 * @param nTargetThreads Target number of threads
 */
extern void LUtilSetGlobalMKLThreads(lorads_int nTargetThreads);

extern lorads_int LUtilUpdateCheckEma(double *current_ema, double *old_ema, double new_value, double alpha, double threshold, lorads_int update_interval, lorads_int *counter);
/**
 * @brief Updates exponential moving average
 * @param current_ema Current EMA value
 * @param old_ema Old EMA value
 * @param new_value New value to incorporate
 * @param alpha Smoothing factor
 * @param threshold Threshold for update
 * @param update_interval Update interval
 * @param counter Counter for updates
 * @return 1 if updated, 0 otherwise
 */
extern lorads_int LUtilUpdateCheckEma(double *current_ema, double *old_ema, double new_value, 
                                    double alpha, double threshold, lorads_int update_interval, 
                                    lorads_int *counter);

/**
 * @brief Reallocates memory for double array
 * @param data Array to reallocate
 * @param nOld Old size
 * @param nNew New size
 */
extern void REALLOC(double **data, lorads_int nOld, lorads_int nNew);

#ifdef __cplusplus
}
#endif

#endif /* LORADS_UTILS_H */
