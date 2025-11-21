/**
 * @file def_lorads_sdp_data.h
 * @brief Definition of SDP (Semidefinite Programming) data structures for LORADS solver
 * @details This header file defines the data structures and types used for handling
 * SDP data in the LORADS solver, including coefficient storage formats and their operations.
 */

#ifndef DEF_LORADS_SDP_DATA
#define DEF_LORADS_SDP_DATA

#include "lorads.h"
#include "def_lorads_elements.h"

/**
 * @brief Structure representing a sparse matrix element
 * @details Stores the position and index of a non-zero element in a sparse matrix.
 */
typedef struct {
    lorads_int row;    ///< Row index of the element
    lorads_int col;    ///< Column index of the element
    lorads_int index;  ///< Global index of the element
} SparseElement;

/**
 * @brief Node structure for dictionary implementation
 * @details Used in the hash table implementation for sparse matrix operations.
 */
typedef struct DictNode {
    SparseElement element;  ///< The sparse element stored in this node
    struct DictNode *next;  ///< Pointer to the next node in the chain
} DictNode;

/**
 * @brief Dictionary structure for sparse matrix operations
 * @details Implements a hash table for efficient sparse matrix element lookup.
 */
typedef struct {
    DictNode **table;  ///< Array of pointers to dictionary nodes
    lorads_int size;   ///< Size of the hash table
} Dict;

/**
 * @brief Enumeration of SDP coefficient storage types
 * @details Defines the three possible storage formats for SDP coefficients:
 * - Zero matrix: no storage needed
 * - Sparse matrix: only non-zero elements stored
 * - Dense matrix: all elements stored
 */
typedef enum {
    SDP_COEFF_ZERO,   ///< Zero matrix storage type
    SDP_COEFF_SPARSE, ///< Sparse matrix storage type
    SDP_COEFF_DENSE,  ///< Dense matrix storage type
} sdp_coeff_type;

/**
 * @brief Structure for SDP coefficient with polymorphic operations
 * @details Provides a unified interface for SDP coefficient operations,
 * supporting different storage formats through function pointers.
 */
typedef struct{
    lorads_int        nSDPCol;  ///< Number of columns in the SDP matrix

    sdp_coeff_type dataType;    ///< Storage type of the coefficient
    void      *dataMat;         ///< Pointer to the actual matrix data

    void (*create)                      ( void **, lorads_int, lorads_int, lorads_int *, double * );  ///< Creates a new matrix
    void (*scal)                        ( void *, double );  ///< Scales the matrix by a factor
    lorads_int (*getnnz)                ( void * );  ///< Gets number of non-zero elements
    void (*getmatnz)                    ( void *, lorads_int * );  ///< Gets non-zero pattern
    void (*add2buffer)                  ( void *, double, lorads_int *, double *);  ///< Adds to buffer
    void (*destroy)                     ( void ** );  ///< Destroys the matrix
    void (*view)                        ( void * );  ///< Prints matrix information

    void (*mul_rk)                      (void *, lorads_sdp_dense *, double *);  ///< Multiplies with rank-k matrix
    void (*mv)                          (void *, double *, double *, lorads_int);  ///< Matrix-vector multiplication
    void (*mul_inner_rk_double)         (void *, lorads_sdp_dense *, lorads_sdp_dense *, double *, void *, sdp_coeff_type);  ///< Inner product with rank-k matrices
    void (*zeros)                       (void *);  ///< Sets matrix to zero
    void (*add_sdp_coeff)               (void *, void *, double, sdp_coeff_type);  ///< Adds another SDP coefficient
    void (*nrm1)                        (void *, double *);  ///< Computes L1 norm
    void (*nrm2Square)                  (void *, double *);  ///< Computes squared L2 norm
    void (*nrmInf)                      (void *, double *);  ///< Computes L inf norm
    void (*statNnz)                     (lorads_int *);  ///< Updates non-zero counter
    void (*scaleData)                   (void *, double);  ///< Scales matrix data
    void (*collectNnzPos)               (void *, lorads_int *, lorads_int *, lorads_int *);  ///< Collects non-zero positions
    void (*reConstructIndex)            (void *, Dict *);  ///< Reconstructs indices using dictionary
}sdp_coeff;

/**
 * @brief Structure for zero SDP coefficient
 * @details Represents a zero matrix in SDP problems, storing only dimension information.
 */
typedef struct {
    lorads_int nSDPCol;  ///< Number of columns in the zero matrix
} sdp_coeff_zero;

/**
 * @brief Structure for sparse SDP coefficient
 * @details Represents a sparse matrix in SDP problems using triplet format.
 */
typedef struct {
    lorads_int      nSDPCol;        ///< Number of columns in the matrix
    lorads_int      nTriMatElem;    ///< Number of non-zero elements
    lorads_int      *triMatCol;     ///< Array of column indices
    lorads_int      *triMatRow;     ///< Array of row indices
    double          *triMatElem;    ///< Array of non-zero values
    lorads_int      *nnzIdx2ResIdx; ///< Mapping from non-zero index to result index
} sdp_coeff_sparse;

/**
 * @brief Structure for dense SDP coefficient
 * @details Represents a dense matrix in SDP problems using packed storage format.
 */
typedef struct {
    lorads_int     nSDPCol;     ///< Number of columns in the matrix
    double        *dsMatElem;   ///< Array storing matrix elements in packed format
    double         *fullMat;    ///< Full matrix storage for UV^T operations
} sdp_coeff_dense;

#endif