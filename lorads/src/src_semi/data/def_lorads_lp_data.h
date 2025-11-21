/**
 * @file def_lorads_lp_data.h
 * @brief Definitions for LP data structures and coefficient types in LORADS
 * @details This header file defines the data structures and types for handling
 * Linear Programming (LP) data in the LORADS solver, including different coefficient
 * storage formats and their associated operations.
 */

#ifndef DEF_LORADS_LP_DATA
#define DEF_LORADS_LP_DATA

#include "lorads.h"

/**
 * @brief Enumeration of LP coefficient storage types
 * @details Defines the possible ways to store LP coefficients:
 * - LP_COEFF_ZERO: Zero coefficients (no storage needed)
 * - LP_COEFF_SPARSE: Sparse storage format
 * - LP_COEFF_DENSE: Dense storage format
 */
typedef enum{
    LP_COEFF_ZERO,
    LP_COEFF_SPARSE,
    LP_COEFF_DENSE
}lp_coeff_type;

/**
 * @brief Structure for LP coefficients with polymorphic operations
 * @details This structure provides a unified interface for different coefficient storage types:
 * - Contains basic dimension information
 * - Stores the actual coefficient data
 * - Provides function pointers for common operations
 */
typedef struct {
    lorads_int nRows;
    lp_coeff_type dataType;
    void *dataMat;

    void  (*create)              (void **, lorads_int, lorads_int, lorads_int *, double *);
    void  (*destroy)             (void **);
    void  (*mul_inner_rk_double) (void *, double *, double *);
    void  (*weight_sum)          (void *, double *, double *);
//    void         (*scaleData)           (void *, double );
}lp_coeff;

/**
 * @brief Structure for zero LP coefficients
 * @details Represents a set of LP coefficients that are all zero.
 * Only stores the number of rows since no actual data is needed.
 */
typedef struct {
    lorads_int nRows;
}lp_coeff_zero;

/**
 * @brief Structure for sparse LP coefficients
 * @details Stores LP coefficients in sparse format:
 * - Uses compressed sparse row (CSR) format
 * - Stores row pointers and non-zero values
 * - Includes count of non-zero elements
 */
typedef struct {
    lorads_int nRows;
    lorads_int nnz;
    lorads_int *rowPtr;
    double *val;
}lp_coeff_sparse;

/**
 * @brief Structure for dense LP coefficients
 * @details Stores LP coefficients in dense format:
 * - Stores all values in a contiguous array
 * - Includes number of rows
 */
typedef struct {
    lorads_int nRows;
    double *val;
}lp_coeff_dense;

#endif
