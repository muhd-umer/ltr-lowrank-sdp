/**
 * @file def_lorads_elements.h
 * @brief Definition of basic data structures and types for LORADS solver
 * @details This header file defines fundamental data structures used throughout the LORADS solver,
 * including vector types, matrix representations, and their associated operations.
 */

#ifndef DEF_LORADS_ELEMENTS
#define DEF_LORADS_ELEMENTS

#include "lorads.h"

/**
 * @brief Enumeration of vector storage types
 * @details Defines the two possible storage formats for vectors in the solver:
 * - Dense storage: all elements are stored
 * - Sparse storage: only non-zero elements are stored
 */
typedef enum{
    LORADS_DENSE_VEC,  ///< Dense vector storage type
    LORADS_SPARSE_VEC  ///< Sparse vector storage type
}vecType;

/**
 * @brief Generic vector structure with polymorphic operations
 * @details This structure provides a unified interface for vector operations,
 * supporting both dense and sparse storage formats through function pointers.
 */
typedef struct{
    vecType type;  ///< Storage type of the vector
    void *data;    ///< Pointer to the actual vector data
    void (*add) (double *, void*, double *);  ///< Function pointer for vector addition
    void (*zero) (void *);  ///< Function pointer for zeroing vector
}lorads_vec;

/**
 * @brief Structure for sparse vector representation
 * @details Stores a sparse vector using compressed format with indices
 * and values for non-zero elements.
 */
typedef struct{
    lorads_int nnz;     ///< Number of non-zero elements
    lorads_int *nnzIdx; ///< Array of indices for non-zero elements
    double *val;        ///< Array of values for non-zero elements
}sparse_vec;

/**
 * @brief Structure for dense vector representation
 * @details Stores a dense vector with all elements in a contiguous array.
 */
typedef struct{
    lorads_int nnz;  ///< Number of elements (all elements are stored)
    double *val;     ///< Array of all vector elements
}dense_vec;

/**
 * @brief Structure for dense SDP (Semidefinite Programming) matrix
 * @details Represents a dense matrix used in SDP problems with rank information.
 */
typedef struct{
    lorads_int rank;     ///< Rank of the matrix
    lorads_int nRows;    ///< Number of rows in the matrix
    double *matElem;     ///< Array storing matrix elements
}lorads_sdp_dense;

/**
 * @brief Structure for dense LP (Linear Programming) matrix
 * @details Represents a dense matrix used in LP problems.
 */
typedef struct{
    lorads_int nCols;    ///< Number of columns in the matrix
    double *matElem;     ///< Array storing matrix elements
}lorads_lp_dense;

#endif