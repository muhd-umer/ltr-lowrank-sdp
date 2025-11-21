/**
 * @file def_lorads_sdp_conic.h
 * @brief Definition of conic data structures for SDP (Semidefinite Programming) in LORADS solver
 * @details This header file defines the data structures and types used for handling
 * conic constraints in SDP problems, including feature detection, cone types, and their operations.
 */

#ifndef DEF_LORADS_SDP_CONIC
#define DEF_LORADS_SDP_CONIC

#include "def_lorads_elements.h"
#include "def_lorads_sdp_data.h"
#include "lorads.h"

/**
 * @brief Integer feature indices for problem characteristics
 * @details These macros define indices for various integer features used in problem analysis:
 * - Problem structure features (null objective, many cones, etc.)
 * - Dimension features (cone dimensions, number of rows, etc.)
 * - Matrix type counts (sparse, dense, zero matrices, etc.)
 */
#define INT_FEATURE_I_NULLOBJ 0      ///< Null objective feature index
#define INT_FEATURE_I_MANYCONES 1    ///< Many cones feature index
#define INT_FEATURE_I_NOPINTERIOR 2  ///< No primal interior feature index
#define INT_FEATURE_I_NODINTERIOR 3  ///< No dual interior feature index
#define INT_FEATURE_I_VERYDENSE 4    ///< Very dense problem feature index
#define INT_FEATURE_I_IMPTRACE 5     ///< Important trace feature index
#define INT_FEATURE_I_IMPYBOUND 6    ///< Important y-bound feature index
#define INT_FEATURE_N_SUMCONEDIMS 7  ///< Sum of cone dimensions feature index
#define INT_FEATURE_N_MAXCONEDIM 8   ///< Maximum cone dimension feature index
#define INT_FEATURE_N_CONES 9        ///< Number of cones feature index
#define INT_FEATURE_N_ROWS 10        ///< Number of rows feature index
#define INT_FEATURE_N_SPSDPCONES 11  ///< Number of sparse SDP cones feature index
#define INT_FEATURE_N_DSSDPCONES 12  ///< Number of dense SDP cones feature index
#define INT_FEATURE_N_LPCONES 13     ///< Number of LP cones feature index
#define INT_FEATURE_N_BNDCONES 14    ///< Number of bound cones feature index
#define INT_FEATURE_N_ZEORMATS 15    ///< Number of zero matrices feature index
#define INT_FEATURE_N_SPMATS 16      ///< Number of sparse matrices feature index
#define INT_FEATURE_N_DSMATS 17      ///< Number of dense matrices feature index
#define INT_FEATURE_N_SPR1MATS 18    ///< Number of sparse rank-1 matrices feature index
#define INT_FEATURE_N_DSR1MATS 19    ///< Number of dense rank-1 matrices feature index

/**
 * @brief Double feature indices for problem characteristics
 * @details These macros define indices for various floating-point features used in problem analysis:
 * - Norm features (Frobenius, one-norm, infinity-norm)
 * - Scaling factors
 * - Bounds and trace information
 */
#define DBL_FEATURE_OBJFRONORM 0     ///< Objective Frobenius norm feature index
#define DBL_FEATURE_OBJONENORM 1     ///< Objective one-norm feature index
#define DBL_FEATURE_RHSFRONORM 2     ///< RHS Frobenius norm feature index
#define DBL_FEATURE_RHSONENORM 3     ///< RHS one-norm feature index
#define DBL_FEATURE_RHSINFNORM 4     ///< RHS infinity-norm feature index
#define DBL_FEATURE_OBJSCALING 5     ///< Objective scaling feature index
#define DBL_FEATURE_RHSSCALING 6     ///< RHS scaling feature index
#define DBL_FEATURE_DATAFRONORM 7    ///< Data Frobenius norm feature index
#define DBL_FEATURE_DATAONENORM 8    ///< Data one-norm feature index
#define DBL_FEATURE_IMPYBOUNDUP 9    ///< Important y upper bound feature index
#define DBL_FEATURE_IMPYBOUNDLOW 10  ///< Important y lower bound feature index
#define DBL_FEATURE_IMPTRACEX 11     ///< Important trace x feature index

/**
 * @brief Enumeration of conic constraint types
 * @details Defines the different types of conic constraints supported by the solver:
 * - LP constraints: A' * y <= c
 * - Bound constraints: y <= u
 * - Scalar bound constraints
 * - Dense and sparse SDP constraints
 */
typedef enum {
    LORADS_CONETYPE_UNKNOWN,    ///< Unknown cone type (not implemented)
    LORADS_CONETYPE_LP,         ///< Linear programming cone (A' * y <= c)
    LORADS_CONETYPE_BOUND,      ///< Bound cone (y <= u)
    LORADS_CONETYPE_SCALAR_BOUND, ///< Scalar bound cone
    LORADS_CONETYPE_DENSE_SDP,  ///< Dense semidefinite programming cone
    LORADS_CONETYPE_SPARSE_SDP, ///< Sparse semidefinite programming cone
} cone_type;

/**
 * @brief Structure for SDP cone with polymorphic operations
 * @details Provides a unified interface for SDP cone operations, supporting different
 * storage formats and constraint types through function pointers.
 */
typedef struct{
    cone_type type;  ///< Type of the cone
    sdp_coeff *sdp_coeff_w_sum;  ///< Weighted sum of SDP coefficients
    sdp_coeff *sdp_obj_sum;      ///< Sum of SDP objective coefficients
    sdp_coeff *sdp_slack_var;    ///< Slack variables
    sdp_coeff *UVt_w_sum;        ///< Weighted sum of UV^T matrices
    sdp_coeff *UVt_obj_sum;      ///< Sum of objective UV^T matrices

    void  *usrData;  ///< User data pointer
    void  *coneData; ///< Cone-specific data pointer

    double sdp_coeff_w_sum_sp_ratio;  ///< Sparsity ratio of weighted sum

    lorads_int nConstr;  ///< Number of constraints

    /* Conic data interface */
    void         (*coneCreate)          ( void ** );  ///< Creates a new cone
    void         (*coneProcData)        ( void *, lorads_int, lorads_int, lorads_int *, lorads_int *, double * );  ///< Processes cone data
    void         (*conePresolveData)    ( void * );  ///< Presolves cone data
    void         (*coneDestroyData)     ( void ** );  ///< Destroys cone data

    void         (*coneAUV)             ( void *, lorads_sdp_dense *, lorads_sdp_dense *, double *, sdp_coeff *);  ///< Computes A*U*V^T
    void         (*objAUV)              ( void *, lorads_sdp_dense *, lorads_sdp_dense *, double *,  sdp_coeff *);  ///< Computes objective A*U*V^T
    void         (*coneObjNrm1)         (void *, double *);  ///< Computes L1 norm of objective
    void         (*coneObjNrm2Square)   (void *, double *);  ///< Computes squared L2 norm of objective
    void         (*coneObjNrmInf)       (void *, double *);  ///< Computes L inf norm of objective
    void         (*sdpDataWSum)         (void *, double *, sdp_coeff *);  ///< Computes weighted sum of SDP data
    void         (*addObjCoeff)         (void *, sdp_coeff *);  ///< Adds objective coefficient
    void         (*addObjCoeffRand)     (void *, sdp_coeff *);  ///< Adds random objective coefficient
    void         (*coneView)            ( void * );  ///< Prints cone information

    /* Feature detection */
    void         (*getstat)             ( void *, double *, lorads_int [20], double [20] );  ///< Gets cone statistics
    void         (*nnzStat)             ( void *, lorads_int *);  ///< Gets non-zero statistics
    void         (*nnzStatCoeff)        ( void *, double *, lorads_int *, lorads_int *);  ///< Gets coefficient non-zero statistics
    void         (*dataScale)           ( void *, double);  ///< Scales cone data
    void         (*objScale)            ( void *, double);  ///< Scales objective
    void         (*nnzStatAndDenseDetect)(void *, lorads_int *, bool *);  ///< Detects density and non-zero patterns
    void         (*collectNnzPos)       (void *, lorads_int *, lorads_int *);  ///< Collects non-zero positions
    void         (*reConstructIndex)    (void *, Dict *);  ///< Reconstructs indices using dictionary
}lorads_sdp_cone;

/**
 * @brief Structure for dense SDP block
 * @details Represents a dense SDP block with its coefficients and statistics.
 */
typedef struct {
    lorads_int   nRow;  ///< Number of rows
    lorads_int   nCol;  ///< Number of columns

    sdp_coeff **sdpRow;  ///< Array of SDP row coefficients
    sdp_coeff  *sdpObj;  ///< SDP objective coefficients

    lorads_int sdpConeStats[5];  ///< Statistics for different coefficient types
} lorads_cone_sdp_dense;

/**
 * @brief Structure for sparse SDP block
 * @details Represents a sparse SDP block with its coefficients and statistics.
 */
typedef struct {
    lorads_int   nRow;  ///< Total number of rows
    lorads_int   nCol;  ///< Number of columns

    lorads_int nRowElem;  ///< Number of non-zero rows
    lorads_int *rowIdx;   ///< Array of row indices
    sdp_coeff **sdpRow;   ///< Array of SDP row coefficients
    sdp_coeff  *sdpObj;   ///< SDP objective coefficients

    lorads_int sdpConeStats[5];  ///< Statistics for different coefficient types
} lorads_cone_sdp_sparse;

/**
 * @brief Structure for tuple of indices
 * @details Used for storing pairs of indices in various operations.
 */
typedef struct {
    lorads_int i;  ///< First index
    lorads_int j;  ///< Second index
} lorads_tuple;

#endif