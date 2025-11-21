/**
 * @file lorads_sdp_data.h
 * @brief LORADS Semidefinite Programming Data Operations
 * @details This header defines the interface for handling Semidefinite Programming (SDP)
 * data structures in LORADS. The operations include:
 * - Creation and management of SDP coefficient matrices
 * - Matrix-vector and matrix-matrix operations
 * - Dense and sparse matrix operations
 * - Matrix scaling and zeroing
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_SDP_DATA_H
#define LORADS_SDP_DATA_H

#include "def_lorads_sdp_data.h"

/**
 * @brief Create a new SDP coefficient matrix
 * @param psdpCoeff Pointer to pointer to store the created SDP coefficient matrix
 */
extern void sdpDataMatCreate(sdp_coeff **psdpCoeff);

/**
 * @brief Set data for an SDP coefficient matrix
 * @param sdpCoeff Pointer to the SDP coefficient matrix
 * @param nSDPCol Number of SDP columns
 * @param dataMatNnz Number of non-zero elements
 * @param dataMatIdx Array of matrix indices
 * @param dataMatElem Array of matrix elements
 */
extern void sdpDataMatSetData(sdp_coeff *sdpCoeff, lorads_int nSDPCol, lorads_int dataMatNnz, lorads_int *dataMatIdx, double *dataMatElem);

/**
 * @brief Destroy an SDP coefficient matrix
 * @param psdpCoeff Pointer to pointer to the SDP coefficient matrix to destroy
 */
extern void sdpDataMatDestroy(sdp_coeff **psdpCoeff);

/**
 * @brief Multiply a dense matrix with a rank matrix
 * @param A Pointer to the input matrix
 * @param X Dense matrix
 * @param AX Output matrix
 */
extern void dataMatDenseMultiRkMat(void *A, lorads_sdp_dense *X, double *AX);

/**
 * @brief Compute hash function for matrix indices
 * @param row Row index
 * @param col Column index
 * @param size Hash table size
 * @return Hash value
 */
extern lorads_int hash_function(lorads_int row, lorads_int col, lorads_int size);

/**
 * @brief Perform dense matrix-vector multiplication
 * @param A Pointer to the input matrix
 * @param x Input vector
 * @param y Output vector
 * @param n Vector dimension
 */
extern void dataMatDenseMV(void *A, double *x, double *y, lorads_int n);

/**
 * @brief Multiply a sparse matrix with a rank matrix
 * @param A Pointer to the input matrix
 * @param X Dense matrix
 * @param AX Output matrix
 */
extern void dataMatSparseMultiRkMat(void *A, lorads_sdp_dense *X, double *AX);

/**
 * @brief Perform sparse matrix-vector multiplication
 * @param A Pointer to the input matrix
 * @param x Input vector
 * @param y Output vector
 * @param n Vector dimension
 */
extern void dataMatSparseMV(void *A, double *x, double *y, lorads_int n);

/**
 * @brief Zero out a sparse matrix
 * @param A Pointer to the matrix to zero
 */
extern void dataMatSparseZeros(void *A);

/**
 * @brief Scale a sparse matrix
 * @param A Pointer to the matrix to scale
 * @param scaleFactor Scaling factor
 */
extern void dataMatSparseScale(void *A, double scaleFactor);

/**
 * @brief Zero out a dense matrix
 * @param A Pointer to the matrix to zero
 */
extern void dataMatDenseZeros(void *A);

/**
 * @brief Scale a dense matrix
 * @param A Pointer to the matrix to scale
 * @param scaleFactor Scaling factor
 */
extern void dataMatDenseScale(void *A, double scaleFactor);

#endif