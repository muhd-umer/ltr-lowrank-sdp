/**
 * @file def_lorads_user_data.h
 * @brief LORADS User Data Interface Definitions
 * @details This header file defines the data structures and interfaces for handling
 * user-provided conic optimization data in LORADS. It supports both SDP and LP cones
 * with different matrix representations.
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef DEF_LORADS_USER_DATA_H
#define DEF_LORADS_USER_DATA_H
#include "lorads_utils.h"
#include "lorads.h"
#include "def_lorads_sdp_conic.h"

/* Interface of user data */

/** 
 * @struct lorads_user_data
 * @brief LORADS user conic data structure for SDP and LP cones
 * @details This structure holds the data for different types of cones in the optimization problem.
 * The data representation varies depending on the cone type:
 * 
 * For SDP cones:
 * - Uses CSC (Compressed Sparse Column) representation
 * - Matrix size: [(n + 1) * n / 2] by [m + 1]
 * - Stores SDP matrix coefficients and objective coefficients
 * - Only lower triangular part is stored
 * 
 * For LP cones and bound cones:
 * - Uses CSC representation
 * - Matrix size: [n] by [m]
 * - Contains LP data
 */
struct lorads_user_data {
    /** @brief Type of the cone (SDP, LP, or bound) */
    cone_type cone;
    
    /** @brief Number of constraints in the cone */
    lorads_int     nConicRow;
    
    /** @brief For SDP cone: conic dimension, for LP cone: number of LP columns */
    lorads_int     nConicCol;
    
    /** @brief Array of column start indices in CSC format */
    lorads_int    *coneMatBeg;
    
    /** @brief Array of row indices in CSC format */
    lorads_int    *coneMatIdx;
    
    /** @brief Array of matrix elements in CSC format */
    double *coneMatElem;
    
    /** @brief Number of non-zero elements in the matrix */
    lorads_int nnz;
};

#endif /* def_lorads_user_data_h */
