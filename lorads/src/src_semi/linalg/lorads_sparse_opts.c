/**
 * @file lorads_sparse_opts.c
 * @brief LORADS Sparse Matrix Operations Implementation
 * @details This file implements various sparse matrix operations for LORADS,
 * including sparse matrix-vector multiplication, column operations, and matrix
 * decompression utilities. It provides efficient implementations for handling
 * sparse matrices in different formats.
 * 
 * @author LORADS Team
 * @date 2024
 */

#include <math.h>
#include <assert.h>
#include "lorads_sparse_opts.h"
#include "lorads_vec_opts.h"

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief Count number of non-zero columns in a sparse matrix
 * @param n Number of columns in the matrix
 * @param Ap Column pointer array (CSC format)
 * @return Number of columns containing non-zero elements
 * @details Counts the number of columns that have at least one non-zero element
 * in a sparse matrix stored in compressed sparse column (CSC) format
 */
extern lorads_int csp_nnz_cols ( lorads_int n, lorads_int *Ap ) {
    
    lorads_int nzcols = 0;
    
    for ( lorads_int i = 0; i < n; ++i ) {
        nzcols += ( Ap[i + 1] - Ap[i] > 0 );
    }
    return nzcols;
}

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
 * @details Computes y += a * A * x where A is a sparse matrix in COO format
 * and x, y are dense vectors. If a is zero, returns immediately.
 */
extern void spMul(lorads_int n, lorads_int nnz, double a,
                  lorads_int *Ai, lorads_int *Aj, double *Ax, double *x,
                  lorads_int k, double *y){
    if (a == 0.0){
        return;
    }
    lorads_int row = 0;
    lorads_int col = 0;
    for (int i = 0; i < nnz; ++i){
        row = Ai[i];
        col = Aj[i];
        axpy(&k, &Ax[i], &x[col], &n, &y[row], &n);
    }
}

/**
 * @brief Decompress a column to triplet matrix format
 * @param n Dimension of the matrix
 * @param nnz Number of non-zero elements
 * @param Ci Column indices array
 * @param Cx Non-zero values array
 * @param Ai Output row indices array
 * @param Aj Output column indices array
 * @param Ax Output non-zero values array
 * @details Converts a sparse matrix from compressed column format to triplet
 * (COO) format. The input matrix is assumed to be stored in a compressed format
 * where Ci contains the linear indices of non-zero elements.
 */
extern void tsp_decompress( lorads_int n, lorads_int nnz, lorads_int *Ci, double *Cx, lorads_int *Ai, lorads_int *Aj, double *Ax ) {

    lorads_int j = 0, idthresh = n;

    for ( lorads_int k = 0; k < nnz; ++k ) {
        while ( Ci[k] >= idthresh ) {
            j += 1;
            idthresh += n - j;
        }
        Ai[k] = Ci[k] - idthresh + n;
        Aj[k] = j;
        Ax[k] = Cx[k];
    }

    return;
}
