/**
 * @file lorads_elements.h
 * @brief LORADS Matrix Element Operations Interface
 * @details This header file defines the interface for basic matrix element operations
 * in LORADS. It provides functions for:
 * - Adding dense and sparse matrix elements to vectors
 * - Zeroing out dense and sparse matrix elements
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_ELEMENTS_H
#define LORADS_ELEMENTS_H

/**
 * @brief Add dense matrix elements to a vector
 * @param alpha Scaling factor for the addition
 * @param constrVal Pointer to the dense matrix elements
 * @param vec Target vector to add the elements to
 * @details Performs the operation: vec += alpha * constrVal
 */
extern void addDense(double *alpha, void *constrVal, double *vec);

/**
 * @brief Add sparse matrix elements to a vector
 * @param alpha Scaling factor for the addition
 * @param constrVal Pointer to the sparse matrix elements
 * @param vec Target vector to add the elements to
 * @details Performs the operation: vec += alpha * constrVal, where constrVal
 * contains only non-zero elements
 */
extern void addSparse(double *alpha, void *constrVal, double *vec);

/**
 * @brief Zero out dense matrix elements
 * @param constrVal Pointer to the dense matrix elements to zero
 * @details Sets all elements in the dense matrix to zero
 */
extern void zeroDense(void *constrVal);

/**
 * @brief Zero out sparse matrix elements
 * @param constrVal Pointer to the sparse matrix elements to zero
 * @details Sets all non-zero elements in the sparse matrix to zero
 */
extern void zeroSparse(void *constrVal);







#endif