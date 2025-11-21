/**
 * @file lorads_sdp_data.c
 * @brief Implementation of SDP data structures and operations for LORADS
 * @details This file implements the core functionality for handling SDP (Semidefinite Programming)
 * data structures, including sparse and dense matrix operations, coefficient management,
 * and various utility functions for the LORADS solver.
 */

#include "lorads_utils.h"
#include "def_lorads_sdp_data.h"
#include "lorads_sdp_data.h"
#include "def_lorads_elements.h"
#include "lorads_vec_opts.h"
#include "lorads_sparse_opts.h"
#include "lorads_dense_opts.h"

/**
 * @brief Counts and prints the number of non-zero elements in a vector
 * @param vec Input vector
 * @param n Length of the vector
 * @details Counts elements with absolute value greater than 1e-10
 */
extern void statNnz(double *vec, lorads_int n){
    lorads_int nnz = 0;
    for (lorads_int i = 0; i < n; ++i){
        if (fabs(vec[i]) > 1e-10){
            nnz += 1;
        }
    }
#ifdef INT32
    printf("nnz = %d\n", nnz);
#endif
#ifdef UNIX_INT64
    printf("nnz = %ld\n", nnz);
#endif
#ifdef MAC_INT64
    printf("nnz = %lld\n", nnz);
#endif
}

/**
 * @brief Computes a hash value for a matrix element
 * @param row Row index
 * @param col Column index
 * @param size Size of the hash table
 * @return Hash value for the element
 */
extern lorads_int hash_function(lorads_int row, lorads_int col, lorads_int size) {
    return (row + col) % size;
}

/**
 * @brief Finds the index of a matrix element in a dictionary
 * @param dict Dictionary structure
 * @param row Row index
 * @param col Column index
 * @return Index of the element if found, -1 otherwise
 */
lorads_int find_index(Dict *dict, lorads_int row, lorads_int col) {
    lorads_int hash_index = hash_function(row, col, dict->size);
    DictNode *node = dict->table[hash_index];
    while (node != NULL) {
        if (node->element.row == row && node->element.col == col) {
            return node->element.index;
        }
        node = node->next;
    }
    return -1;
}

/**
 * @brief Clears the data in an SDP coefficient structure
 * @param sdpCoeff Pointer to the SDP coefficient structure
 */
extern void sdpDataMatClear( sdp_coeff *sdpCoeff ) {
    if ( !sdpCoeff ) {
        return;
    }
    sdpCoeff->destroy(&sdpCoeff->dataMat);
    LORADS_ZERO(sdpCoeff, sdp_coeff, 1);
}

/**
 * @brief Destroys an SDP coefficient structure
 * @param psdpCoeff Pointer to pointer of the SDP coefficient structure
 */
extern void sdpDataMatDestroy( sdp_coeff **psdpCoeff ) {
    if ( !psdpCoeff ) {
        return;
    }
    sdpDataMatClear(*psdpCoeff);
    LORADS_FREE(*psdpCoeff);
}

/**
 * @brief Creates a zero matrix implementation
 * @param pA Pointer to store the created matrix
 * @param nSDPCol Number of columns
 * @param dataMatNnz Number of non-zero elements (unused)
 * @param dataMatIdx Index array (unused)
 * @param dataMatElem Element array (unused)
 */
static void dataMatCreateZeroImpl( void **pA, lorads_int nSDPCol, lorads_int dataMatNnz, lorads_int *dataMatIdx, double *dataMatElem ) {
    (void) dataMatNnz;
    (void) dataMatIdx;
    (void) dataMatElem;
    if ( !pA ) {
        LORADS_ERROR_TRACE;
    }
    sdp_coeff_zero *zero;
    LORADS_INIT(zero, sdp_coeff_zero, 1);
    LORADS_MEMCHECK(zero);
    if (zero){
        zero->nSDPCol = nSDPCol;
    }else{
        LORADS_ERROR_TRACE;
    }

    *pA = (void *) zero;
}

static void dataMatViewZeroImpl( void *A );
static void dataMatViewSparseImpl( void *A );
static void dataMatViewDenseImpl( void *A );

/**
 * @brief Creates a sparse matrix implementation
 * @param pA Pointer to store the created matrix
 * @param nSDPCol Number of columns
 * @param dataMatNnz Number of non-zero elements
 * @param dataMatIdx Index array
 * @param dataMatElem Element array
 */
static void dataMatCreateSparseImpl( void **pA, lorads_int nSDPCol, lorads_int dataMatNnz, lorads_int *dataMatIdx, double *dataMatElem ) {
    if ( !pA ) {
        goto exit_cleanup;
    }

    sdp_coeff_sparse *sparse;
    LORADS_INIT(sparse, sdp_coeff_sparse, 1);
    LORADS_MEMCHECK(sparse);
    sparse->nSDPCol = nSDPCol;
    sparse->nTriMatElem = dataMatNnz;

    LORADS_INIT(sparse->triMatRow, lorads_int, dataMatNnz);
    LORADS_INIT(sparse->triMatCol, lorads_int, dataMatNnz);
    LORADS_INIT(sparse->triMatElem, double, dataMatNnz);

    LORADS_MEMCHECK(sparse->triMatRow);
    LORADS_MEMCHECK(sparse->triMatCol);
    LORADS_MEMCHECK(sparse->triMatElem);

    // Note tsp_decompress work when ascending sort
    if ( !LUtilCheckIfAscending(dataMatNnz, dataMatIdx) ) {
        LUtilAscendSortDblByInt(dataMatElem, dataMatIdx, 0, dataMatNnz - 1);
    }

    tsp_decompress(sparse->nSDPCol, sparse->nTriMatElem, dataMatIdx, dataMatElem,
                   sparse->triMatRow, sparse->triMatCol, sparse->triMatElem);

    *pA = (void *) sparse;
    exit_cleanup:
        return;
}

/**
 * @brief Creates a dense matrix implementation
 * @param pA Pointer to store the created matrix
 * @param nSDPCol Number of columns
 * @param dataMatNnz Number of non-zero elements
 * @param dataMatIdx Index array
 * @param dataMatElem Element array
 */
static void dataMatCreateDenseImpl( void **pA, lorads_int nSDPCol, lorads_int dataMatNnz, lorads_int *dataMatIdx, double *dataMatElem ) {
    if ( !pA ) {
        LORADS_ERROR_TRACE;
    }

    sdp_coeff_dense *dense;
    LORADS_INIT(dense, sdp_coeff_dense, 1);
    LORADS_MEMCHECK(dense);

    dense->nSDPCol = nSDPCol;
    LORADS_INIT(dense->dsMatElem, double, PACK_NNZ(nSDPCol));
    LORADS_MEMCHECK(dense->dsMatElem);

    pds_decompress(dataMatNnz, dataMatIdx, dataMatElem, dense->dsMatElem);
    *pA = (void *) dense;
}

/**
 * @brief Scales a zero matrix by a factor (no-op)
 * @param A Zero matrix
 * @param alpha Scaling factor
 */
static void dataMatScalZeroImpl( void *A, double alpha ) {
    return;
}

/**
 * @brief Scales a sparse matrix by a factor
 * @param A Sparse matrix
 * @param alpha Scaling factor
 */
static void dataMatScalSparseImpl( void *A, double alpha ) {
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    lorads_int incx = 1;
    scal(&spA->nTriMatElem, &alpha, spA->triMatElem, &incx);
    return;
}

/**
 * @brief Computes the L1 norm of a sparse matrix
 * @param A Sparse matrix
 * @param res Pointer to store the result
 */
static void dataMatSparseNrm1(void *A, double *res){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    double nrmA = 0.0;
    for (lorads_int i = 0; i < spA->nTriMatElem; ++i){
        nrmA += 2 * fabs(spA->triMatElem[i]);
        lorads_int row = spA->triMatRow[i];
        lorads_int col = spA->triMatCol[i];
        if (row == col){
            nrmA -= fabs(spA->triMatElem[i]);
        }
    }
    res[0] = nrmA;
}

/**
 * @brief Computes the squared L2 norm of a sparse matrix
 * @param A Sparse matrix
 * @param res Pointer to store the result
 */
static void dataMatSparseNrm2Square(void *A, double *res){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    double nrmA = 0.0;
    for (lorads_int i = 0; i < spA->nTriMatElem; ++i){
        nrmA += 2 * pow(spA->triMatElem[i], 2);
        lorads_int row = spA->triMatRow[i];
        lorads_int col = spA->triMatCol[i];
        if (row == col){
            nrmA -= pow(spA->triMatElem[i], 2);
        }
    }
    res[0] = nrmA;
}

/**
 * @brief Computes the L inf norm of a sparse matrix
 * @param A Sparse matrix
 * @param res Pointer to store the result
 */
static void dataMatSparseNrmInf(void *A, double *res){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    double nrmA = 0.0;
    for ( lorads_int i = 0; i < spA->nTriMatElem; ++i ) {
        nrmA =  LORADS_MAX( LORADS_ABS(spA->triMatElem[i]), nrmA);
    }
    res[0] = nrmA;
}

/**
 * @brief Sets all elements of a sparse matrix to zero
 * @param A Sparse matrix to zero out
 */
extern void dataMatSparseZeros(void *A){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    LORADS_ZERO(spA->triMatElem, double, spA->nTriMatElem);
}

/**
 * @brief Updates the non-zero element counter
 * @param nnzStat Pointer to the counter
 */
static void dataMatSparseStatNnz(lorads_int *nnzStat){
    nnzStat[0]++;
    return;
}

/**
 * @brief Scales a sparse matrix by a factor
 * @param A Sparse matrix
 * @param scaleFactor Scaling factor
 */
extern void dataMatSparseScale(void *A, double scaleFactor){
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *)A;
    lorads_int incx = 1;
    scal(&sparse->nTriMatElem, &scaleFactor, sparse->triMatElem, &incx);
}

/**
 * @brief Collects non-zero positions from a sparse matrix
 * @param A Sparse matrix
 * @param nnzRow Array to store row indices
 * @param nnzCol Array to store column indices
 * @param nnzIdx Current index in the arrays
 */
static void dataMatSparseCollectNnzPos(void *A, lorads_int *nnzRow, lorads_int *nnzCol, lorads_int *nnzIdx){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    LORADS_MEMCPY(&nnzRow[nnzIdx[0]], spA->triMatRow, lorads_int, spA->nTriMatElem);
    LORADS_MEMCPY(&nnzCol[nnzIdx[0]], spA->triMatCol, lorads_int, spA->nTriMatElem);
    nnzIdx[0] += spA->nTriMatElem;
    return;
}

/**
 * @brief Reconstructs indices for a sparse matrix using a dictionary
 * @param A Sparse matrix
 * @param dict Dictionary containing index mappings
 */
static void dataMatSparseReConstructIndex(void *A, Dict *dict){
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    lorads_int nnz = spA->nTriMatElem;
    lorads_int row, col;
    lorads_int idx = 0;
    LORADS_INIT(spA->nnzIdx2ResIdx, lorads_int, spA->nTriMatElem);
    for (lorads_int i = 0; i < nnz; ++i){
        row = spA->triMatRow[i];
        col = spA->triMatCol[i];
        idx = find_index(dict, row, col);
        if (idx == -1){
            LORADS_ERROR_TRACE;
        }
        spA->nnzIdx2ResIdx[i] = idx;
    }
    return;
}

/**
 * @brief Computes the L1 norm of a dense matrix
 * @param A Dense matrix
 * @param res Pointer to store the result
 */
static void dataMatDenseNrm1(void *A, double *res){
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    double nrmA = 0.0;
    lorads_int nElem = dsA->nSDPCol * (dsA->nSDPCol + 1) / 2;
    lorads_int colLen; ///< Exclude diagonal
    double colNrm; ///< Exclude diagonal
    double *p = dsA->dsMatElem;
    lorads_int nCol = dsA->nSDPCol;
    lorads_int incx = 1;
    for ( lorads_int i = 0; i < nCol; ++i ) {
        nrmA += fabs(p[0]);
        colLen = nCol - i - 1;
        nrmA += nrm1(&colLen, p + 1, &incx) * 2;
        p = p + colLen + 1;
    }
    res[0] = nrmA;
}

/**
 * @brief Computes the squared L2 norm of a dense matrix
 * @param A Dense matrix
 * @param res Pointer to store the result
 */
static void dataMatDenseNrm2Square(void *A, double *res){
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    double nrmA = 0.0;
    lorads_int nCol = dsA->nSDPCol;
    lorads_int incx = 1;
    lorads_int colLen; ///< Exclude diagonal
    double colNrm; ///< Exclude diagonal
    double *p = dsA->dsMatElem;
    for ( lorads_int i = 0; i < nCol; ++i ) {
        nrmA += pow(p[0], 2);
        colLen = nCol - i - 1;
        nrmA += pow(nrm2(&colLen, p + 1, &incx), 2) * 2;
        p = p + colLen + 1;
    }
    res[0] = nrmA;
}

/**
 * @brief Computes the L inf norm of a dense matrix
 * @param A Dense matrix
 * @param res Pointer to store the result
 */
static void dataMatDenseNrmInf(void *A, double *res){
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    lorads_int nElem = dsA->nSDPCol * (dsA->nSDPCol + 1) / 2;
    double *p = dsA->dsMatElem;
    res[0] = 0;
    for (lorads_int i = 0; i < nElem; ++i ){
        res[0] =  LORADS_MAX(res[0],  LORADS_ABS(p[i]));
    }
    return;
}

/**
 * @brief Sets all elements of a dense matrix to zero
 * @param A Dense matrix to zero out
 */
extern void dataMatDenseZeros(void *A){
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    LORADS_ZERO(dsA->dsMatElem, double, dsA->nSDPCol * (dsA->nSDPCol + 1) / 2);
}

/**
 * @brief Scales a dense matrix by a factor
 * @param A Dense matrix
 * @param scaleFactor Scaling factor
 */
extern void dataMatDenseScale(void *A, double scaleFactor){
    sdp_coeff_dense *dense = (sdp_coeff_dense *)A;
    lorads_int n = dense->nSDPCol * (dense->nSDPCol + 1) / 2;
    scal(&n, &scaleFactor, dense->dsMatElem, &AIntConstantOne);
}

/**
 * @brief Updates the non-zero element counter for dense matrices
 * @param nnzStat Pointer to the counter
 */
static void dataMatDenseStatNnz(lorads_int *nnzStat){
    nnzStat[0]++;
    return;
}

/**
 * @brief Reconstructs indices for a dense matrix using a dictionary
 * @param A Dense matrix
 * @param dict Dictionary containing index mappings
 */
static void dataMatDenseReConstructIndex(void *A, Dict *dict){
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    return;
}

/**
 * @brief Gets the number of non-zero elements in a zero matrix
 * @param A Zero matrix
 * @return Always returns 0
 */
static lorads_int dataMatGetNnzZeroImpl( void *A ) {
    return 0;
}

/**
 * @brief Gets the number of non-zero elements in a sparse matrix
 * @param A Sparse matrix
 * @return Number of non-zero elements
 */
static lorads_int dataMatGetNnzSparseImpl( void *A ) {
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    return spA->nTriMatElem;
}

/**
 * @brief Gets the number of non-zero elements in a dense matrix
 * @param A Dense matrix
 * @return Number of elements in the matrix
 */
static lorads_int dataMatGetNnzDenseImpl( void *A ) {
    sdp_coeff_dense *dsA = (sdp_coeff_dense *) A;
    return dsA->nSDPCol * (dsA->nSDPCol + 1) / 2;
}

/**
 * @brief Gets the sparsity pattern of a zero matrix
 * @param A Zero matrix
 * @param spout Output array for sparsity pattern
 */
static void dataMatGetZeroSparsityImpl( void *A, lorads_int *spout ) {
    return;
}

/**
 * @brief Gets the sparsity pattern of a sparse matrix
 * @param A Sparse matrix
 * @param spout Output array for sparsity pattern
 */
static void dataMatGetSparseSparsityImpl( void *A, lorads_int *spout ) {
    sdp_coeff_sparse *spA = (sdp_coeff_sparse *) A;
    for ( lorads_int k = 0; k < spA->nTriMatElem; ++k ) {
        spout[FULL_IDX(spA->nSDPCol, spA->triMatRow[k], spA->triMatCol[k])] = 1;
    }
    return;
}

/**
 * @brief Gets the sparsity pattern of a dense matrix
 * @param A Dense matrix
 * @param spout Output array for sparsity pattern
 * @note This method should not be invoked
 */
static void dataMatGetDenseSparsityImpl( void *A, lorads_int *spout ) {
    /* This method should not be invoked */
    assert( 0 );
}

/**
 * @brief Clears a zero matrix implementation
 * @param A Zero matrix to clear
 */
static void dataMatClearZeroImpl( void *A ) {
    if ( !A ) {
        return;
    }
    LORADS_ZERO(A, sdp_coeff_zero, 1);
    return;
}

/**
 * @brief Clears a sparse matrix implementation
 * @param A Sparse matrix to clear
 */
static void dataMatClearSparseImpl( void *A) {
    if ( !A ) {
        return;
    }
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;
    LORADS_FREE(sparse->triMatRow);
    LORADS_FREE(sparse->triMatCol);
    LORADS_FREE(sparse->triMatElem);
    if (sparse->nnzIdx2ResIdx){
        LORADS_FREE(sparse->nnzIdx2ResIdx);
    }
    LORADS_ZERO(A, sdp_coeff_sparse, 1);
    return;
}

/**
 * @brief Clears a dense matrix implementation
 * @param A Dense matrix to clear
 */
static void dataMatClearDenseImpl( void *A ) {
    if ( !A ) {
        return;
    }
    sdp_coeff_dense *dense = (sdp_coeff_dense *) A;
    LORADS_FREE(dense->dsMatElem);
    LORADS_ZERO(A, sdp_coeff_dense, 1);
    return;
}

/**
 * @brief Destroys a zero matrix implementation
 * @param pA Pointer to pointer of the zero matrix
 */
static void dataMatDestroyZeroImpl( void **pA ) {
    if ( !pA ) {
        return;
    }
    dataMatClearZeroImpl(*pA);
    sdp_coeff *p = (sdp_coeff *)*pA;
    LORADS_FREE(*pA);
    return;
}

/**
 * @brief Destroys a sparse matrix implementation
 * @param pA Pointer to pointer of the sparse matrix
 */
static void dataMatDestroySparseImpl( void **pA ) {
    if ( !pA ) {
        return;
    }
    sdp_coeff *p = (sdp_coeff *)*pA;
    dataMatClearSparseImpl(*pA);
    LORADS_FREE(*pA);
    return;
}

/**
 * @brief Destroys a dense matrix implementation
 * @param pA Pointer to pointer of the dense matrix
 */
extern void dataMatDestroyDenseImpl( void **pA ) {
    if ( !pA ) {
        return;
    }
    dataMatClearDenseImpl(*pA);
    sdp_coeff *p = (sdp_coeff *)*pA;
    LORADS_FREE(*pA);
    return;
}

/**
 * @brief Prints information about a zero matrix
 * @param A Zero matrix to view
 */
static void dataMatViewZeroImpl( void *A ) {
#ifdef INT32
    printf("Zero matrix of size %d L1 = [%5.3e] L2 = [%5.3e] \n",
           ((sdp_coeff_zero *) A)->nSDPCol, 0.0, 0.0);
#endif
#ifdef UNIX_INT64
    printf("Zero matrix of size %ld L1 = [%5.3e] L2 = [%5.3e] \n",
           ((sdp_coeff_zero *) A)->nSDPCol, 0.0, 0.0);
#endif
#ifdef MAC_INT64
    printf("Zero matrix of size %lld L1 = [%5.3e] L2 = [%5.3e] \n",
           ((sdp_coeff_zero *) A)->nSDPCol, 0.0, 0.0);
#endif
    return;
}

/**
 * @brief Prints information about a sparse matrix
 * @param A Sparse matrix to view
 */
static void dataMatViewSparseImpl( void *A ) {
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;
#ifdef INT32
    printf("Sparse matrix of size %d and %d nnzs. \n",
           sparse->nSDPCol, sparse->nTriMatElem);
#endif
#ifdef MAC_INT64
    printf("Sparse matrix of size %lld and %lld nnzs. \n",
           sparse->nSDPCol, sparse->nTriMatElem);
#endif

#ifdef UNIX_INT64
    printf("Sparse matrix of size %ld and %ld nnzs. \n",
           sparse->nSDPCol, sparse->nTriMatElem);
#endif
    return;
}

/**
 * @brief Prints information about a dense matrix
 * @param A Dense matrix to view
 */
static void dataMatViewDenseImpl( void *A ) {
    sdp_coeff_dense *dense = (sdp_coeff_dense *) A;
#ifdef INT32
    printf("Dense matrix of size %d. \n", dense->nSDPCol);
#endif
#ifdef UNIX_INT64
    printf("Dense matrix of size %ld. \n", dense->nSDPCol);
#endif
#ifdef MAC_INT64
    printf("Dense matrix of size %lld. \n", dense->nSDPCol);
#endif
    return;
}

/**
 * @brief Multiplies a zero matrix with a rank-k matrix (no-op)
 * @param A Zero matrix
 * @param X Rank-k matrix
 * @param AX Output matrix
 */
static void dataMatZeroMultiRkMat(void *A, lorads_sdp_dense *X, double *AX){
    // AX = A * X = 0
    return;
}

/**
 * @brief Computes inner product of a zero matrix with rank-k matrices
 * @param A Zero matrix
 * @param U First rank-k matrix
 * @param V Second rank-k matrix
 * @param res Pointer to store result
 * @param UVt UV^T matrix (unused)
 * @param UVtType Type of UV^T matrix (unused)
 */
static void dataMatZeroMultiRkMatInnerProRkMat(void *A, lorads_sdp_dense *U, lorads_sdp_dense *V, double *res, void *UVt, sdp_coeff_type UVtType){
    // res = <AU, V> = 0
    res[0] = 0.0; // no change
    return;
}

/**
 * @brief Adds a zero matrix to a dense SDP coefficient
 * @param A Zero matrix
 * @param B Dense SDP coefficient
 * @param weight Weight factor
 * @param Btype Type of B (unused)
 */
static void dataMatZeroAddDenseSDPCoeff(void *A, void *B, double weight, sdp_coeff_type Btype){
    return;
}

/**
 * @brief Computes L1 norm of a zero matrix
 * @param A Zero matrix
 * @param res Pointer to store result (always 0)
 */
static void dataMatZeroNrm1(void *A, double *res){
    res[0] = 0.0;
    return;
}

/**
 * @brief Computes squared L2 norm of a zero matrix
 * @param A Zero matrix
 * @param res Pointer to store result (always 0)
 */
static void dataMatZeroNrm2Square(void *A, double *res){
    res[0] = 0.0;
    return;
}

/**
 * @brief Computes L inf norm of a zero matrix
 * @param A Zero matrix
 * @param res Pointer to store result (always 0)
 */
static void dataMatZeroNrmInf(void *A, double *res){
    res[0] = 0.0;
    return;
}

/**
 * @brief Updates non-zero element counter for zero matrix
 * @param nnzStat Pointer to counter
 */
static void dataMatZeroStatNnz (lorads_int *nnzStat){
    return;
}

/**
 * @brief Scales a zero matrix (no-op)
 * @param A Zero matrix
 * @param scaleFactor Scaling factor (unused)
 */
static void dataMatZeroScale(void *A, double scaleFactor){
    return;
}

/**
 * @brief Collects non-zero positions from a zero matrix (no-op)
 * @param A Zero matrix
 * @param nnzRow Array for row indices (unused)
 * @param nnzCol Array for column indices (unused)
 * @param nnzIdx Current index (unused)
 */
static void dataMatZeroCollectNnzPos(void *A, lorads_int *nnzRow, lorads_int *nnzCol, lorads_int *nnzIdx){
    return;
}

/**
 * @brief Reconstructs indices for a zero matrix (no-op)
 * @param A Zero matrix
 * @param dict Dictionary containing index mappings (unused)
 */
static void dataMatReConstructIndex(void *A, Dict *dict){
    return;
}

/**
 * @brief Multiplies a sparse matrix with a rank-k matrix
 * @param A Sparse matrix
 * @param X Rank-k matrix
 * @param AX Output matrix
 */
extern void dataMatSparseMultiRkMat(void *A, lorads_sdp_dense *X, double *AX){
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;
    LORADS_ZERO(AX, double, X->rank * X->nRows);
    lorads_int row, col = 0;
    for (int i = 0; i < sparse->nTriMatElem; ++i){
        row = sparse->triMatRow[i];
        col = sparse->triMatCol[i];
        axpy(&X->rank, &sparse->triMatElem[i], &X->matElem[col], &sparse->nSDPCol, &AX[row], &sparse->nSDPCol);
        if (row != col){
            axpy(&X->rank, &sparse->triMatElem[i], &X->matElem[row], &sparse->nSDPCol, &AX[col], &sparse->nSDPCol);
        }
    }
    return;
}

/**
 * @brief Multiplies a sparse matrix with a vector
 * @param A Sparse matrix
 * @param x Input vector
 * @param y Output vector
 * @param n Dimension
 */
extern void dataMatSparseMV(void *A, double *x, double *y, lorads_int n){
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;
    assert(n == sparse->nSDPCol);
    LORADS_ZERO(y, double, n);
    lorads_int row, col = 0;
    lorads_int one = 1;
    for (int i = 0; i < sparse->nTriMatElem; ++i){
        row = sparse->triMatRow[i];
        col = sparse->triMatCol[i];
        axpy(&one, &sparse->triMatElem[i], &x[col], &sparse->nSDPCol, &y[row], &sparse->nSDPCol);
        if (row != col){
            axpy(&one, &sparse->triMatElem[i], &x[row], &sparse->nSDPCol, &y[col], &sparse->nSDPCol);
        }
    }
    return;
}

/**
 * @brief Computes the inner product between sparse matrices A and UV^T
 * @param n Matrix dimension
 * @param nnzA Number of non-zeros in matrix A
 * @param Ai Row indices of matrix A
 * @param Aj Column indices of matrix A
 * @param Ax Values of matrix A
 * @param nnzUVt Number of non-zeros in matrix UV^T
 * @param UVti Row indices of matrix UV^T
 * @param UVtj Column indices of matrix UV^T
 * @param UVtx Values of matrix UV^T
 * @param nnzIdx Index mapping array
 * @param res Pointer to store the result
 */
static void sparseAUV(lorads_int n, lorads_int nnzA, lorads_int *Ai, lorads_int *Aj, double *Ax,
                      lorads_int nnzUVt, lorads_int *UVti, lorads_int *UVtj, double *UVtx, lorads_int *nnzIdx, double *res){
    lorads_int incx = 1;
    lorads_int rowA, colA, rowUVt, colUVt = 0;
    if (nnzA == nnzUVt){
        res[0] += 2 * dot(&nnzA, Ax, &incx, UVtx, &incx);
        for (lorads_int i = 0; i < nnzA; ++i){
            rowA = Ai[i]; colA = Aj[i];
            rowUVt = UVti[i]; colUVt = UVtj[i];
            if (rowA == colA && rowA == rowUVt && rowUVt == colUVt){
                res[0] -= Ax[i] * UVtx[i];
            }
        }
    }else{
        double temp;
        assert(nnzUVt >  nnzA);
        lorads_int i, j, idx = 0;
        for (i = 0; i < nnzA; ++i){
            rowA = Ai[i];
            colA = Aj[i];
            idx = nnzIdx[i];
            temp = 2 * Ax[i] * UVtx[idx];
            res[0] += temp;
            if (rowA == colA){
                res[0] -= 0.5 * temp;
            }
        }
    }
}

/**
 * @brief Computes the inner product between dense matrices A and UV^T
 * @param nnzA Number of non-zeros in matrix A
 * @param Ai Row indices of matrix A
 * @param Aj Column indices of matrix A
 * @param Ax Values of matrix A
 * @param UVtx Values of matrix UV^T
 * @param nnzIdx Index mapping array
 * @param res Pointer to store the result
 */
static void denseAUV(lorads_int nnzA, const lorads_int *Ai, const lorads_int *Aj, double *Ax,
                     const double *UVtx, lorads_int *nnzIdx, double *res){
    lorads_int rowA, colA, idx = 0;
    double temp = 0.0;
    for (lorads_int i = 0; i < nnzA; ++i){
        rowA = Ai[i]; colA = Aj[i];
        idx = nnzIdx[i];
        temp = 2 * Ax[i] * UVtx[idx];
        res[0] += temp;
        if (rowA == colA){
            res[0] -= 0.5 * temp;
        }
    }
}

static void dataMatSparseMultiRkMatInnerProRkMat(void *A, lorads_sdp_dense *U, lorads_sdp_dense *V, double *res, void *UVtIn, sdp_coeff_type UVtType){
    /*
    res = <AU, V>
    UVt is (UVt + VUt) / 2 in fact
     */
    res[0] = 0.0;

    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;

    if (UVtType == SDP_COEFF_SPARSE){
        sdp_coeff_sparse *UVt = (sdp_coeff_sparse *)UVtIn;
        sparseAUV(sparse->nSDPCol, sparse->nTriMatElem, sparse->triMatRow, sparse->triMatCol, sparse->triMatElem,
                  UVt->nTriMatElem, UVt->triMatRow, UVt->triMatCol, UVt->triMatElem, sparse->nnzIdx2ResIdx, res);
    }else{
        sdp_coeff_dense *UVt = (sdp_coeff_dense *)UVtIn;
        denseAUV(sparse->nTriMatElem, sparse->triMatRow, sparse->triMatCol, sparse->triMatElem,
                 UVt->dsMatElem, sparse->nnzIdx2ResIdx, res);
    }
}

static void dataMatSparseAddDenseSDPCoeff(void *A, void *B, double weight){
    // B = B + A * weight
    sdp_coeff_sparse *sparse = (sdp_coeff_sparse *) A;
    sdp_coeff_dense *dense = (sdp_coeff_dense *) B;
    lorads_int n = dense->nSDPCol;
    if (sparse->nnzIdx2ResIdx == NULL){
        for (lorads_int i = 0; i < sparse->nTriMatElem; ++i){
            lorads_int row = sparse->triMatRow[i];
            lorads_int col = sparse->triMatCol[i];
            if (row >= col){
                dense->dsMatElem[(n * col -col*(col+1)/2 + row)] += weight * sparse->triMatElem[i];
            }
        }
    }else{
        lorads_int idx; lorads_int row; lorads_int col;
        for (lorads_int i = 0; i < sparse->nTriMatElem; ++i){
            row = sparse->triMatRow[i];
            col = sparse->triMatCol[i];
            assert(row >= col);
//            idx = dense->rowCol2NnzIdx[row][col];
            idx = sparse->nnzIdx2ResIdx[i];
            assert(idx != -1);
            dense->dsMatElem[idx] += weight * sparse->triMatElem[i];
        }
    }
}



static void dataMatSparseAddSparseSDPCoeff(void *A, void *B, double weight){
    // B = B + weight * A
    sdp_coeff_sparse *sparseA = (sdp_coeff_sparse *) A;
    sdp_coeff_sparse *sparseB = (sdp_coeff_sparse *) B;
    lorads_int nnzA = sparseA->nTriMatElem;
    lorads_int nnzB = sparseB->nTriMatElem;
    assert(nnzB >= nnzA);

    lorads_int rowA, colA; lorads_int idx;
    for (lorads_int i = 0; i < nnzA; ++i){
        rowA = sparseA->triMatRow[i];
        colA = sparseA->triMatCol[i];
        idx = sparseA->nnzIdx2ResIdx[i];
        sparseB->triMatElem[idx] += weight * sparseA->triMatElem[i];
    }
}

/**
 * @brief Adds a sparse matrix to another SDP coefficient matrix
 * @param A Source sparse matrix
 * @param B Target SDP coefficient matrix
 * @param weight Weight factor for the addition
 * @param B_type Type of the target matrix (dense or sparse)
 */
static void dataMatSparseAddSDPCoeff(void *A, void *B, double weight, sdp_coeff_type B_type){
    if (B_type == SDP_COEFF_DENSE){
        dataMatSparseAddDenseSDPCoeff(A, B, weight);
    }else if (B_type == SDP_COEFF_SPARSE){
        dataMatSparseAddSparseSDPCoeff(A, B, weight);
    }else{
        LORADS_ERROR_TRACE;
    }
}

/**
 * @brief Multiplies a dense matrix with a rank-k matrix
 * @param A Dense matrix
 * @param X Rank-k matrix
 * @param AX Output matrix
 * @details Computes AX = A * X where A is a dense matrix and X is a rank-k matrix
 */
extern void dataMatDenseMultiRkMat(void *A, lorads_sdp_dense *X, double *AX){
    /*
     AX = A * X, A is dense meanse result is dense
     */
    sdp_coeff_dense *dense = (sdp_coeff_dense *) A;
    double alpha = 1.0;
    double beta = 0.0;
    LORADS_ZERO(dense->fullMat, double, dense->nSDPCol * dense->nSDPCol);
    lorads_int idx = 0;
    for (lorads_int col = 0; col < dense->nSDPCol; ++col){
        for (lorads_int row = col; row < dense->nSDPCol; ++row){
            dense->fullMat[dense->nSDPCol * col + row] = dense->dsMatElem[idx];
            dense->fullMat[dense->nSDPCol * row + col] = dense->dsMatElem[idx];
            idx++;
        }
    }
    char side = 'L'; // C:= alpha * A * B + beta * C;
    char uplo = 'L';
    lorads_int m = dense->nSDPCol;
    lorads_int n = X->rank;
#ifdef UNDER_BLAS
    dsymm_(&side, &uplo, &m, &n, &alpha, dense->fullMat, &m, X->matElem, &m, &beta, AX, &m );
#else
    dsymm(&side, &uplo, &m, &n, &alpha, dense->fullMat, &m, X->matElem, &m, &beta, AX, &m );
#endif
}

/**
 * @brief Multiplies a dense matrix with a vector
 * @param A Dense matrix
 * @param x Input vector
 * @param y Output vector
 * @param n Dimension
 * @details Computes y = A * x where A is a dense matrix
 */
extern void dataMatDenseMV(void *A, double *x, double *y, lorads_int n){
    sdp_coeff_dense *dense = (sdp_coeff_dense *) A;
    double alpha = 1.0;
    double beta = 0.0;
    LORADS_ZERO(dense->fullMat, double, dense->nSDPCol * dense->nSDPCol);
    lorads_int idx = 0;
    for (lorads_int col = 0; col < dense->nSDPCol; ++col){
        for (lorads_int row = col; row < dense->nSDPCol; ++row){
            dense->fullMat[dense->nSDPCol * col + row] = dense->dsMatElem[idx];
            dense->fullMat[dense->nSDPCol * row + col] = dense->dsMatElem[idx];
            idx++;
        }
    }
    lorads_int m = dense->nSDPCol;
    assert(n == m);
    char side = 'L'; // C:= alpha * A * B + beta * C;
    char uplo = 'L';
    lorads_int k = 1;
#ifdef UNDER_BLAS
    dsymm_(&side, &uplo, &m, &k, &alpha, dense->fullMat, &m, x, &m, &beta, y, &m );
#else
    dsymm(&side, &uplo, &m, &k, &alpha, dense->fullMat, &m, x, &m, &beta, y, &m );
#endif
}

/**
 * @brief Computes inner product of a dense matrix with rank-k matrices
 * @param A Dense matrix
 * @param U First rank-k matrix
 * @param V Second rank-k matrix
 * @param res Pointer to store result
 * @param UVtIn UV^T matrix
 * @param UVtType Type of UV^T matrix
 */
static void dataMatDenseMultiRkMatInnerProRkMat(void *A, lorads_sdp_dense *U, lorads_sdp_dense *V, double *res, void *UVtIn, sdp_coeff_type UVtType){
    /* res = <AU, V>
     dense A, dense U, dense V
     */
    res[0] = 0.0;
    sdp_coeff_dense *dense = (sdp_coeff_dense *) A;
    lorads_int n = 0;
    lorads_int incx = 1;
    sdp_coeff_dense *UVt = (sdp_coeff_dense *)UVtIn;
    n = UVt->nSDPCol * (UVt->nSDPCol + 1) / 2;
    res[0] += 2 * dot(&n, dense->dsMatElem, &incx, UVt->dsMatElem, &incx);
    lorads_int idx = 0; lorads_int colNum = UVt->nSDPCol;
    for (lorads_int i = 0; i < UVt->nSDPCol; ++i){
        res[0] -= dense->dsMatElem[idx] * UVt->dsMatElem[idx];
        idx += colNum;
        colNum -= 1;
    }
}

/**
 * @brief Adds a dense matrix to another dense matrix
 * @param A Source dense matrix
 * @param B Target dense matrix
 * @param weight Weight factor for the addition
 */
static void dataMatDenseAddDenseSDPCoeff(void *A, void *B, double weight){
    // B += A * weight
    sdp_coeff_dense *denseA = (sdp_coeff_dense *) A;
    sdp_coeff_dense *denseB = (sdp_coeff_dense *) B;
    lorads_int n = denseA->nSDPCol * (denseA->nSDPCol + 1) / 2;
    lorads_int incx = 1;
    axpy(&n, &weight, denseA->dsMatElem, &incx, denseB->dsMatElem, &incx);
}

/**
 * @brief Adds a dense matrix to another SDP coefficient matrix
 * @param A Source dense matrix
 * @param B Target SDP coefficient matrix
 * @param weight Weight factor for the addition
 * @param B_type Type of the target matrix (must be dense)
 */
static void dataMatDenseAddSDPCoeff(void *A, void *B, double weight, sdp_coeff_type B_type){
    if (B_type == SDP_COEFF_DENSE){
        dataMatDenseAddDenseSDPCoeff(A, B, weight);
    }else{
        LORADS_ERROR_TRACE;
    }
}

/**
 * @brief Initializes the function pointers for an SDP coefficient structure
 * @param sdpCoeff SDP coefficient structure to initialize
 * @param dataType Type of coefficient (zero, sparse, or dense)
 * @details Sets up function pointers for various operations based on the coefficient type:
 * - create: Creates a new matrix of the specified type
 * - getnnz: Gets number of non-zero elements
 * - getmatnz: Gets sparsity pattern
 * - destroy: Destroys the matrix
 * - view: Prints matrix information
 * - mul_rk: Multiplies with rank-k matrix
 * - mv: Matrix-vector multiplication
 * - mul_inner_rk_double: Inner product with rank-k matrices
 * - add_sdp_coeff: Adds to another SDP coefficient
 * - nrm1/nrm2Square/nrmInf: Various norm computations
 * - zeros: Sets matrix to zero
 * - statNnz: Updates non-zero counter
 * - scaleData: Scales matrix by a factor
 * - collectNnzPos: Collects non-zero positions
 * - reConstructIndex: Reconstructs indices using dictionary
 */
static void sdpDataMatIChooseType(sdp_coeff *sdpCoeff, sdp_coeff_type dataType ) {
    sdpCoeff->dataType = dataType;
    switch (dataType) {
        case SDP_COEFF_ZERO:
            sdpCoeff->create = dataMatCreateZeroImpl;
            sdpCoeff->getnnz = dataMatGetNnzZeroImpl;
            sdpCoeff->getmatnz = dataMatGetZeroSparsityImpl;
            sdpCoeff->destroy = dataMatDestroyZeroImpl;
            sdpCoeff->view = dataMatViewZeroImpl;
            sdpCoeff->mul_rk = dataMatZeroMultiRkMat;
            sdpCoeff->mv = NULL;
            sdpCoeff->mul_inner_rk_double = dataMatZeroMultiRkMatInnerProRkMat;
            sdpCoeff->add_sdp_coeff = dataMatZeroAddDenseSDPCoeff;
            sdpCoeff->nrm1 = dataMatZeroNrm1;
            sdpCoeff->nrm2Square = dataMatZeroNrm2Square;
            sdpCoeff->nrmInf = dataMatZeroNrmInf;
            sdpCoeff->zeros = NULL;
            sdpCoeff->statNnz = dataMatZeroStatNnz;
            sdpCoeff->scaleData = dataMatZeroScale;
            sdpCoeff->collectNnzPos = dataMatZeroCollectNnzPos;
            sdpCoeff->reConstructIndex = dataMatReConstructIndex;
            break;
        case SDP_COEFF_SPARSE:
            sdpCoeff->create = dataMatCreateSparseImpl;
            sdpCoeff->getnnz = dataMatGetNnzSparseImpl;
            sdpCoeff->getmatnz = dataMatGetSparseSparsityImpl;
            sdpCoeff->destroy = dataMatDestroySparseImpl;
            sdpCoeff->view = dataMatViewSparseImpl;
            sdpCoeff->mul_rk = dataMatSparseMultiRkMat;
            sdpCoeff->mv = dataMatSparseMV; // for dual infeasibility
            sdpCoeff->mul_inner_rk_double = dataMatSparseMultiRkMatInnerProRkMat;
            sdpCoeff->add_sdp_coeff = dataMatSparseAddSDPCoeff;
            sdpCoeff->nrm1 = dataMatSparseNrm1;
            sdpCoeff->nrm2Square = dataMatSparseNrm2Square;
            sdpCoeff->nrmInf = dataMatSparseNrmInf;
            sdpCoeff->zeros = dataMatSparseZeros;
            sdpCoeff->statNnz = dataMatSparseStatNnz;
            sdpCoeff->scaleData = dataMatSparseScale;
            sdpCoeff->collectNnzPos = dataMatSparseCollectNnzPos;
            sdpCoeff->reConstructIndex = dataMatSparseReConstructIndex;
            break;
        case SDP_COEFF_DENSE:
            sdpCoeff->create = dataMatCreateDenseImpl;
            sdpCoeff->getnnz = dataMatGetNnzDenseImpl;
            sdpCoeff->getmatnz = dataMatGetDenseSparsityImpl;
            sdpCoeff->destroy = dataMatDestroyDenseImpl;
            sdpCoeff->view = dataMatViewDenseImpl;
            sdpCoeff->mul_rk = dataMatDenseMultiRkMat;
            sdpCoeff->mv = dataMatDenseMV; // for dual infeasibility
            sdpCoeff->mul_inner_rk_double = dataMatDenseMultiRkMatInnerProRkMat;
            sdpCoeff->add_sdp_coeff = dataMatDenseAddSDPCoeff;
            sdpCoeff->nrm1 = dataMatDenseNrm1;
            sdpCoeff->nrm2Square = dataMatDenseNrm2Square;
            sdpCoeff->nrmInf = dataMatDenseNrmInf;
            sdpCoeff->zeros = dataMatDenseZeros;
            sdpCoeff->statNnz = dataMatDenseStatNnz;
            sdpCoeff->scaleData = dataMatDenseScale;
            sdpCoeff->collectNnzPos = NULL; // not used
            sdpCoeff->reConstructIndex = dataMatDenseReConstructIndex; // not used
            break;
        default:
            assert(0);
            break;
    }
}

/**
 * @brief Creates a new SDP coefficient structure
 * @param psdpCoeff Pointer to store the created structure
 * @details Allocates memory for a new SDP coefficient structure
 */
extern void sdpDataMatCreate( sdp_coeff **psdpCoeff ) {
    if ( !psdpCoeff ) {
        LORADS_ERROR_TRACE;
    }
    sdp_coeff *sdpCoeff;
    LORADS_INIT(sdpCoeff, sdp_coeff, 1);
    LORADS_MEMCHECK(sdpCoeff);
    *psdpCoeff = sdpCoeff;
}

/**
 * @brief Sets the data for an SDP coefficient structure
 * @param sdpCoeff SDP coefficient structure
 * @param nSDPCol Number of columns
 * @param dataMatNnz Number of non-zero elements
 * @param dataMatIdx Index array
 * @param dataMatElem Element array
 * @details Automatically chooses between zero, sparse, and dense storage based on sparsity:
 * - Zero matrix if dataMatNnz is 0
 * - Dense matrix if more than 10% of elements are non-zero
 * - Sparse matrix otherwise
 */
extern void sdpDataMatSetData( sdp_coeff *sdpCoeff, lorads_int nSDPCol, lorads_int dataMatNnz, lorads_int *dataMatIdx, double *dataMatElem ) {
    // sdpCoeff  : data mat struct
    // nSDPCol   : data mat dimension
    // dataMatNnz: data mat non-zeros number
    sdpCoeff->nSDPCol = nSDPCol;

    /* At this stage, only sparse, zero and dense matrices are classified */
    if ( dataMatNnz == 0 ) {
        sdpDataMatIChooseType(sdpCoeff, SDP_COEFF_ZERO);
    } else if ( (double) dataMatNnz > 0.1 * (double)(nSDPCol * (nSDPCol + 1) / 2) ) {
        sdpDataMatIChooseType(sdpCoeff, SDP_COEFF_DENSE);
    } else {
        sdpDataMatIChooseType(sdpCoeff, SDP_COEFF_SPARSE);
    }
    /* Create data */
    sdpCoeff->create(&sdpCoeff->dataMat, nSDPCol,
                               dataMatNnz, dataMatIdx, dataMatElem);
}
