/**
 * @file lorads_sparse_opts.h
 * @brief LORADS Sparse Matrix Operations Interface
 * @details This header file declares the interface for sparse matrix operations in LORADS,
 * including sparse matrix-vector multiplication, column operations, and matrix format
 * conversion utilities. It provides efficient implementations for handling sparse matrices
 * in different formats (CSC, COO).
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_SPARSE_OPT_H
#define LORADS_SPARSE_OPT_H

#include <stdio.h>
#include "lorads.h"

/**
 * @brief Count number of non-zero columns in a sparse matrix
 * @param n Number of columns in the matrix
 * @param Ap Column pointer array (CSC format)
 * @return Number of columns containing non-zero elements
 */
extern lorads_int csp_nnz_cols ( lorads_int n, lorads_int *Ap );

/**
 * @brief Perform sparse matrix-vector multiplication
 * @param n Dimension of the matrix
 * @param nnz Number of non-zero elements
 * @param a Scalar multiplier
 * @param Ai Row indices array
 * @param Aj Column indices array
 * @param Ax Non-zero values array
 * @param x Input vector
 * @param k Vector dimension
 * @param y Output vector
 */
extern void spMul(lorads_int n, lorads_int nnz, double a,
                  lorads_int *Ai, lorads_int *Aj, double *Ax, double *x,
                  lorads_int k, double *y);

/**
 * @brief Decompress a column to triplet matrix format
 * @param n Dimension of the matrix
 * @param nnz Number of non-zero elements
 * @param Ci Column indices array
 * @param Cx Non-zero values array
 * @param Ai Output row indices array
 * @param Aj Output column indices array
 * @param Ax Output non-zero values array
 */
extern void tsp_decompress( lorads_int n, lorads_int nnz,
                            lorads_int *Ci, double *Cx,
                            lorads_int *Ai, lorads_int *Aj, double *Ax );
#endif
