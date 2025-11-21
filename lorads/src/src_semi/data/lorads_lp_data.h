/**
 * @file lorads_lp_data.h
 * @brief LORADS Linear Programming Data Interface
 * @details This header file defines the interface for handling Linear Programming (LP)
 * data structures in LORADS. It provides functions for:
 * - Managing LP coefficient matrices
 * - Handling different types of LP data representations
 * - Converting between data types
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_LP_DATA_H
#define LORADS_LP_DATA_H


#include "def_lorads_lp_data.h"

/**
 * @brief Choose and initialize the type of LP coefficient matrix
 * @param lpCoeff Pointer to the LP coefficient structure to initialize
 * @param dataType Type of LP coefficient matrix to create
 * @details Sets up the LP coefficient matrix with the specified type and initializes
 * appropriate function pointers for operations on that type of matrix
 */
extern void LPDataMatIChooseType(lp_coeff *lpCoeff, lp_coeff_type dataType);

#endif