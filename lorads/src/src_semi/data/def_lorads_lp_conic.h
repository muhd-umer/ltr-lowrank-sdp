/**
 * @file def_lorads_lp_conic.h
 * @brief Definitions for LP cone data structures in LORADS
 * @details This header file defines the data structures and functions for handling
 * LP (Linear Programming) cones in the LORADS solver. It includes structures for
 * storing LP cone data and function pointers for various operations.
 */

#ifndef DEF_LORADS_LP_CONIC
#define DEF_LORADS_LP_CONIC

#include "lorads.h"
#include "def_lorads_lp_data.h"
#include "def_lorads_elements.h"

/*
 Split an LP cone as
 ---------------------------------
          objMatElem
 ---------------------------------
 |          |          |          |
 | lp_coeff | lp_coeff | lp_coeff |
 |          |          |          |
 */
/**
 * @brief Structure representing an LP cone in the LORADS solver
 * @details This structure stores all necessary data for an LP cone, including:
 * - Objective coefficients
 * - Constraint coefficients
 * - Matrix dimensions and sparsity information
 */
typedef struct {

    lorads_int     nRow; // constraint number
    lorads_int     nCol; // dim of LP

    // obj coeff, full
    double  *objMatElem;

    // constraint coeff
    lorads_int *rowMatBeg;
    lorads_int *rowMatIdx;
    double *rowMatElem;

    lp_coeff **lpCol;
    double *nrm2Square;
} lorads_lp_cone_data;

/**
 * @brief Structure containing LP cone data and function pointers
 * @details This structure combines LP cone data with function pointers for various operations:
 * - Objective function calculations
 * - Cone data management
 * - Matrix operations
 * - Linear system solving
 */
typedef struct {
    lorads_int nCol;
    lorads_lp_cone_data *coneData;

    void (*coneObjNrm1)          (lorads_lp_cone_data *, double *, lorads_int);
    void (*coneObjNrm2Square)    (lorads_lp_cone_data *, double *, lorads_int);
    void (*destroyConeData)      (lorads_lp_cone_data **);
    void (*coneView)             (lorads_lp_cone_data * );
    void (*coneAUV)              (lorads_lp_cone_data *, lorads_lp_dense *, lorads_lp_dense *, double *, lorads_int);
    void (*coneAUV2)             (lorads_lp_cone_data *, double *, double *);
    void (*objAUV)               (lorads_lp_cone_data *, lorads_lp_dense *, lorads_lp_dense *, double *);
    void (*coneObjNrmInf)        (lorads_lp_cone_data *, double *, lorads_int);
    void (*lpDataWSum)           (lorads_lp_cone_data *, double *, double *, lorads_int);
    void (*objCoeffSum)          (lorads_lp_cone_data *, double *, lorads_int);
    void (*scalObj)              (lorads_lp_cone_data *, double);
}lorads_lp_cone;


#endif