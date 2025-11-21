/**
 * @file lorads_lp_conic.h
 * @brief LORADS Linear Programming Conic Interface
 * @details This header file defines the interface for handling Linear Programming (LP)
 * conic optimization problems in LORADS. It provides functions for:
 * - Setting up LP cones
 * - Managing LP cone data structures
 * - Handling LP cone operations
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_LP_CONIC
#define LORADS_LP_CONIC


#include "def_lorads_lp_conic.h"
#include "lorads.h"

/**
 * @brief Set up an LP cone with specified dimensions and data
 * @param lp_cone Pointer to the LP cone structure to initialize
 * @param nRows Number of rows in the LP cone
 * @param nLpCols Number of LP columns
 * @param lpMatBeg Array of matrix beginning indices
 * @param lpMatIdx Array of matrix indices
 * @param LpMatElem Array of matrix elements
 * @details Initializes an LP cone structure with the provided dimensions and data.
 * The function sets up the cone's internal data structures and prepares it for
 * optimization operations.
 */
extern void LORADSSetLpCone(lorads_lp_cone *lp_cone, lorads_int nRows,
                            lorads_int nLpCols, lorads_int *lpMatBeg,
                            lorads_int *lpMatIdx, double *LpMatElem);

#endif