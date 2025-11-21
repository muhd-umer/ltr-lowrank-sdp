/**
 * @file lorads_lp_data.c
 * @brief Implementation of LP (Linear Programming) data structures and operations for LORADS
 * @details This file implements the core functionality for handling LP data structures,
 * including zero, sparse, and dense coefficient storage and their associated operations.
 */

#include <assert.h>
#include "lorads_lp_data.h"
#include "def_lorads_lp_data.h"
#include "lorads_utils.h"
#include "lorads_vec_opts.h"

/**
 * @brief Creates a zero matrix implementation for LP coefficients
 * @param pA Pointer to store the created matrix
 * @param nRows Number of rows
 * @param nnz Number of non-zero elements (must be 0)
 * @param dataMatIdx Index array (unused)
 * @param dataMatElem Element array (unused)
 */
static void LPdataMatCreateZeroImpl(void **pA, lorads_int nRows, lorads_int nnz, lorads_int *dataMatIdx, double *dataMatElem){

    assert(nnz == 0);

    if ( !pA ) {
        LORADS_ERROR_TRACE;
    }

    lp_coeff_zero *zero;
    LORADS_INIT(zero, lp_coeff_zero, 1);
    LORADS_MEMCHECK(zero);

    zero->nRows = nRows;
    *pA = (void *) zero;
}

/**
 * @brief Destroys a zero matrix implementation
 * @param pA Pointer to pointer of the zero matrix
 */
static void LPdataMatDestroyZeroImpl(void **pA){
    return;
}

/**
 * @brief Multiplies a zero matrix with a rank-k matrix (no-op)
 * @param pA Zero matrix
 * @param puv Input vector
 * @param constrVal Output vector (set to zero)
 */
static void LPdataMatZeroMulInnerRkDouble(void *pA, double *puv, double *constrVal){
    lp_coeff_zero *zero = (lp_coeff_zero *)pA;
    LORADS_ZERO(constrVal, double, zero->nRows);
    return;
}

/**
 * @brief Computes weighted sum for zero matrix (no-op)
 * @param pA Zero matrix
 * @param weight Weight vector (unused)
 * @param res Result pointer (unused)
 */
static void LPdataMatZeroWeightSum(void *pA, double *weight, double *res){
    return;
}

/**
 * @brief Scales a zero matrix (no-op)
 * @param pA Zero matrix
 * @param scaleFactor Scaling factor (unused)
 */
static void LPdataMatZeroScaleData(void *pA, double scaleFactor){
    return;
}

/**
 * @brief Creates a sparse matrix implementation for LP coefficients
 * @param pA Pointer to store the created matrix
 * @param nRows Number of rows
 * @param nnz Number of non-zero elements
 * @param dataMatIdx Index array
 * @param dataMatElem Element array
 */
static void LPdataMatCreateSparseImpl(void **pA, lorads_int nRows, lorads_int nnz, lorads_int *dataMatIdx, double *dataMatElem){
    if ( !pA ) {
        LORADS_ERROR_TRACE;
    }
    lp_coeff_sparse *sparse;
    LORADS_INIT(sparse, lp_coeff_sparse, 1);
    LORADS_MEMCHECK(sparse);

    sparse->nRows = nRows;
    sparse->nnz = nnz;
    LORADS_INIT(sparse->rowPtr, lorads_int, nnz);
    LORADS_MEMCHECK(sparse->rowPtr);
    LORADS_INIT(sparse->val, double, nnz);
    LORADS_MEMCHECK(sparse->val);

    LORADS_MEMCPY(sparse->rowPtr, dataMatIdx, lorads_int, nnz);
    LORADS_MEMCPY(sparse->val, dataMatElem, double, nnz);

    *pA = (void *) sparse;
}

/**
 * @brief Destroys a sparse matrix implementation
 * @param pA Pointer to pointer of the sparse matrix
 */
static void LPdataMatDestroySparseImpl(void **pA){
    lp_coeff_sparse *sparse = (lp_coeff_sparse *) *pA;
    LORADS_FREE(sparse->rowPtr);
    LORADS_FREE(sparse->val);
    LORADS_FREE(sparse);
    return;
}

/**
 * @brief Multiplies a sparse matrix with a rank-k matrix
 * @param pA Sparse matrix
 * @param puv Input vector
 * @param constrVal Output vector
 * @details Uses different implementations based on sparsity:
 * - Direct multiplication for very sparse matrices (< 5 non-zeros)
 * - BLAS scal operation for larger matrices
 */
static void LPdataMatSparseMulInnerRkDouble(void *pA, double *puv, double *constrVal){
    lp_coeff_sparse *sparse = (lp_coeff_sparse *)pA;
//    double uv = puv[0];
//    for (lorads_int i = 0; i < sparse->nnz; ++i){
//        constrVal[i] = sparse->val[i] * uv;
//    }
    if (sparse->nnz < 5){
        for (lorads_int i = 0; i < sparse->nnz; ++i){
            constrVal[i] = puv[0] * sparse->val[i];
        }
    }else{
        LORADS_MEMCPY(constrVal, sparse->val, double, sparse->nnz);
        lorads_int incx = 1;
        scal(&sparse->nnz, puv, constrVal, &incx);
    }
    return;
}

/**
 * @brief Computes weighted sum for sparse matrix
 * @param pA Sparse matrix
 * @param weight Weight vector
 * @param res Result pointer
 */
static void LPdataMatSparseWeightSum(void *pA, double *weight, double *res){
    lp_coeff_sparse *sparse = (lp_coeff_sparse *)pA;
    for (lorads_int i = 0; i < sparse->nnz; ++i){
        res[0] += sparse->val[i] * weight[sparse->rowPtr[i]];
    }
    return;
}

/**
 * @brief Scales a sparse matrix by a factor
 * @param pA Sparse matrix
 * @param scaleFactor Scaling factor
 */
static void LPdataMatSparseScaleData(void *pA, double scaleFactor){
    lp_coeff_sparse *sparse = (lp_coeff_sparse *)pA;
    vvscl(&sparse->nnz, &scaleFactor, sparse->val);
    return;
}

/**
 * @brief Creates a dense matrix implementation for LP coefficients
 * @param pA Pointer to store the created matrix
 * @param nRows Number of rows
 * @param nnz Number of non-zero elements
 * @param dataMatIdx Index array
 * @param dataMatElem Element array
 */
static void LPdataMatCreateDenseImpl(void **pA, lorads_int nRows, lorads_int nnz, lorads_int *dataMatIdx, double *dataMatElem){
    if ( !pA ) {
        LORADS_ERROR_TRACE;
    }

    lp_coeff_dense *dense;
    LORADS_INIT(dense, lp_coeff_dense, 1);
    LORADS_MEMCHECK(dense);
    LORADS_INIT(dense->val, double, nRows);
    LORADS_MEMCHECK(dense->val);

    for (lorads_int i = 0; i < dense->nRows; ++i){
        dense->val[dataMatIdx[i]] = dataMatElem[i];
    }
    *pA = (void *) dense;
}

/**
 * @brief Destroys a dense matrix implementation
 * @param pA Pointer to pointer of the dense matrix
 */
static void LPdataMatDestroyDenseImpl(void **pA){
    lp_coeff_dense *dense = (lp_coeff_dense *) *pA;
    LORADS_FREE(dense->val);
    LORADS_ZERO(dense, lp_coeff_dense, 1);
    return;
}

/**
 * @brief Multiplies a dense matrix with a rank-k matrix
 * @param pA Dense matrix
 * @param puv Input vector
 * @param constrVal Output vector
 */
static void LPdataMatDenseMulInnerRkDouble(void *pA, double *puv, double *constrVal){
    lp_coeff_dense *dense = (lp_coeff_dense *)pA;
    LORADS_MEMCPY(constrVal, dense->val, double, dense->nRows);
    double uv = puv[0];
    lorads_int incx = 1;
    scal(&(dense->nRows), &uv, constrVal, &incx);
    return;
}

/**
 * @brief Computes weighted sum for dense matrix
 * @param pA Dense matrix
 * @param weight Weight vector
 * @param res Result pointer
 */
static void LPdataMatDenseWeightSum(void *pA, double *weight, double *res){
    lp_coeff_dense *dense = (lp_coeff_dense *)pA;
    lorads_int incx = 1;
    res[0] += dot(&(dense->nRows), dense->val, &incx, weight, &incx);
    return;
}

/**
 * @brief Scales a dense matrix by a factor
 * @param pA Dense matrix
 * @param scaleFactor Scaling factor
 */
static void LPdataMatDenseScaleData(void *pA, double scaleFactor){
    lp_coeff_dense *dense = (lp_coeff_dense *)pA;
    lorads_int incx = 1;
    vvscl(&dense->nRows, &scaleFactor, dense->val);
    return;
}

/**
 * @brief Initializes function pointers for an LP coefficient structure
 * @param lpCoeff LP coefficient structure to initialize
 * @param dataType Type of coefficient (zero, sparse, or dense)
 * @details Sets up function pointers for various operations based on the coefficient type:
 * - create: Creates a new matrix of the specified type
 * - destroy: Destroys the matrix
 * - mul_inner_rk_double: Multiplies with rank-k matrix
 * - weight_sum: Computes weighted sum
 */
extern void LPDataMatIChooseType(lp_coeff *lpCoeff, lp_coeff_type dataType){
    lpCoeff->dataType = dataType;
    switch (dataType) {
        case LP_COEFF_ZERO:
            lpCoeff->create = LPdataMatCreateZeroImpl;
            lpCoeff->destroy = LPdataMatDestroyZeroImpl;
            lpCoeff->mul_inner_rk_double = LPdataMatZeroMulInnerRkDouble;
            lpCoeff->weight_sum = LPdataMatZeroWeightSum;
//            lpCoeff->scaleData = LPdataMatZeroScaleData;
            break;
        case LP_COEFF_DENSE:
            lpCoeff->create = LPdataMatCreateDenseImpl;
            lpCoeff->destroy = LPdataMatDestroyDenseImpl;
            lpCoeff->mul_inner_rk_double = LPdataMatDenseMulInnerRkDouble;
            lpCoeff->weight_sum = LPdataMatDenseWeightSum;
//            lpCoeff->scaleData = LPdataMatDenseScaleData;
            break;
        case LP_COEFF_SPARSE:
            lpCoeff->create = LPdataMatCreateSparseImpl;
            lpCoeff->destroy = LPdataMatDestroySparseImpl;
            lpCoeff->mul_inner_rk_double = LPdataMatSparseMulInnerRkDouble;
            lpCoeff->weight_sum = LPdataMatSparseWeightSum;
            break;
        default:
            assert(0);
            break;
    }
    return;
}
