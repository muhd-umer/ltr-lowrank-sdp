/**
 * @file lorads_elements.c
 * @brief LORADS Basic Element Operations
 * @details This file implements basic operations for dense and sparse vectors in LORADS.
 * The operations include:
 * - Vector addition with scaling
 * - Vector zeroing
 * - Support for both dense and sparse formats
 * 
 * @author LORADS Team
 * @date 2024
 */

#include "def_lorads_elements.h"
#include "lorads_utils.h"
#include "lorads_vec_opts.h"

/**
 * @brief Add scaled dense vector to target vector
 * @param alpha Scaling factor
 * @param constrVal Pointer to dense vector structure
 * @param vec Target vector to add to
 * @details Performs the operation: vec += alpha * dense_vector
 * Uses BLAS axpy operation for efficient computation
 */
extern void addDense(double *alpha, void *constrVal, double *vec)
{
    dense_vec *dense = (dense_vec *)constrVal;
    lorads_int incx = 1;
    axpy(&dense->nnz, alpha, dense->val, &incx, vec, &incx);
}

/**
 * @brief Add scaled sparse vector to target vector
 * @param alpha Scaling factor
 * @param constrVal Pointer to sparse vector structure
 * @param vec Target vector to add to
 * @details Performs the operation: vec += alpha * sparse_vector
 * Only adds non-zero elements for efficiency
 */
extern void addSparse(double *alpha, void *constrVal, double *vec)
{
    sparse_vec *sparse = (sparse_vec *)constrVal;
    int idx = 0;
    for (lorads_int i = 0; i < sparse->nnz; ++i)
    {
        idx = sparse->nnzIdx[i];
        vec[idx] += alpha[0] * sparse->val[i];
    }
}

/**
 * @brief Zero out dense vector
 * @param constrVal Pointer to dense vector structure
 * @details Sets all elements of dense vector to zero
 * Uses LORADS_ZERO macro for efficient zeroing
 */
extern void zeroDense(void *constrVal)
{
    dense_vec *dense = (dense_vec *)constrVal;
    LORADS_ZERO(dense->val, double, dense->nnz);
}

/**
 * @brief Zero out sparse vector
 * @param constrVal Pointer to sparse vector structure
 * @details Sets all non-zero elements of sparse vector to zero
 * Only modifies non-zero elements for efficiency
 */
extern void zeroSparse(void *constrVal)
{
    sparse_vec *sparse = (sparse_vec *)constrVal;
    for (lorads_int i = 0; i < sparse->nnz; ++i){
        sparse->val[i] = 0;
    }
}
