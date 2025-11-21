/**
 * @file lorads_user_data.h
 * @brief LORADS User Data Interface
 * @details This header file defines the interface for managing user data structures in LORADS.
 * It provides functions for creating, initializing, and managing data structures for
 * different types of optimization cones (SDP, LP, bound).
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_USER_DATA_H
#define LORADS_USER_DATA_H


#include "lorads.h"
#include "def_lorads_user_data.h"
#include "lorads_utils.h"

/** @brief Forward declaration of user data structure */
typedef struct lorads_user_data user_data;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a new user data structure
 * @param pHdata Pointer to store the created user data structure
 * @details Allocates and initializes a new user data structure with zero values.
 * The structure is used to store optimization problem data for different cone types.
 */
extern void LUserDataCreate( user_data **pHdata );

/**
 * @brief Set cone data in user data structure
 * @param Hdata Pointer to user data structure
 * @param cone Type of cone (SDP, LP, bound)
 * @param nRow Number of rows
 * @param nCol Number of columns
 * @param coneMatBeg Column pointers for sparse matrix
 * @param coneMatIdx Row indices for sparse matrix
 * @param coneMatElem Matrix elements
 * @details Initializes the user data structure with cone-specific data and matrix information.
 */
extern void LUserDataSetConeData( user_data *Hdata, cone_type cone, lorads_int nRow, lorads_int nCol,
                                  lorads_int *coneMatBeg, lorads_int *coneMatIdx, double *coneMatElem);

/**
 * @brief Automatically choose appropriate cone type based on data characteristics
 * @param Hdata Pointer to user data structure
 * @return Selected cone type
 * @details Analyzes the data structure to determine the most appropriate cone type:
 * - Preserves existing bound, sparse SDP, or scalar bound types
 * - For dense SDP, checks sparsity to choose between dense and sparse representation
 * - For LP, checks if it represents bound constraints
 */
extern cone_type LUserDataChooseCone( user_data *Hdata );

/**
 * @brief Clear user data structure
 * @param Hdata Pointer to user data structure
 * @details Resets all fields in the user data structure to zero.
 */
extern void LUserDataClear( user_data *Hdata );

/**
 * @brief Destroy user data structure
 * @param pHdata Pointer to pointer of user data structure
 * @details Frees memory allocated for the user data structure and sets pointer to NULL.
 */
extern void LUserDataDestroy( user_data **pHdata );

/**
 * @brief Create array of SDP data structures
 * @param SDPDatas Pointer to store array of SDP data structures
 * @param nCones Number of SDP cones to create
 * @details Allocates memory for an array of user data structures for SDP cones.
 */
extern void LORADSCreateSDPDatas(user_data ***SDPDatas, lorads_int nCones);

/**
 * @brief Clear and free all user data structures
 * @param coneMatBeg Array of cone matrix column pointers
 * @param coneMatIdx Array of cone matrix row indices
 * @param coneMatElem Array of cone matrix elements
 * @param nBlks Number of blocks
 * @param BlkDims Array of block dimensions
 * @param rowRHS Array of right-hand side values
 * @param LpMatBeg LP matrix column pointers
 * @param LpMatIdx LP matrix row indices
 * @param LpMatElem LP matrix elements
 * @param SDPDatas Array of SDP data structures
 * @details Frees all memory allocated for optimization problem data structures.
 */
extern void LORADSClearUsrData(lorads_int **coneMatBeg, lorads_int **coneMatIdx, double **coneMatElem, lorads_int nBlks,
 lorads_int *BlkDims, double *rowRHS, lorads_int *LpMatBeg, lorads_int *LpMatIdx, double *LpMatElem, user_data **SDPDatas);

#ifdef __cplusplus
}
#endif

#endif /* LORADS_USER_DATA_H */
