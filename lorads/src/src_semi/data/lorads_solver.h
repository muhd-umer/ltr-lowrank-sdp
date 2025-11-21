/**
 * @file lorads_solver.h
 * @brief LORADS Solver Core Functions Header
 * @details This header file contains core functions for the LORADS solver, including:
 * - Solver initialization and cleanup
 * - Problem data management
 * - Variable initialization and destruction
 * - Rank determination and adjustment
 * - State management and transitions
 * - Result processing and output
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_SOLVER
#define LORADS_SOLVER

#include "def_lorads_solver.h"
#include "lorads.h"
#include "lorads_user_data.h"

/**
 * @brief Initialize the LORADS solver with problem dimensions
 * @param ASolver Pointer to solver instance
 * @param nRows Number of constraints
 * @param nCones Number of SDP cones
 * @param blkDims Array of block dimensions
 * @param nLpCols Number of LP columns
 */
extern void LORADSInitSolver(lorads_solver *ASolver, lorads_int nRows, lorads_int nCones, lorads_int *blkDims, lorads_int nLpCols);

/**
 * @brief Clean up and destroy solver resources
 * @param ASolver Pointer to solver instance
 */
extern void LORADSDestroySolver(lorads_solver *ASolver);

/**
 * @brief Set the dual objective coefficients
 * @param ASolver Pointer to solver instance
 * @param dObj Array of dual objective coefficients
 */
extern void LORADSSetDualObjective(lorads_solver *ASolver, double *dObj);

/**
 * @brief Initialize cone data for the solver
 * @param ASolver Pointer to solver instance
 * @param SDPDatas Array of user data structures
 * @param coneMatElem Array of cone matrix elements
 * @param coneMatBeg Array of cone matrix beginnings
 * @param coneMatIdx Array of cone matrix indices
 * @param BlkDims Array of block dimensions
 * @param nConstrs Number of constraints
 * @param nBlks Number of blocks
 * @param nLpCols Number of LP columns
 * @param LpMatBeg Array of LP matrix beginnings
 * @param LpMatIdx Array of LP matrix indices
 * @param LpMatElem Array of LP matrix elements
 */
extern void LORADSInitConeData(lorads_solver *ASolver, user_data **SDPDatas,
                               double **coneMatElem, lorads_int **coneMatBeg, lorads_int **coneMatIdx,
                               lorads_int *BlkDims, lorads_int nConstrs, lorads_int nBlks,
                               lorads_int nLpCols, lorads_int *LpMatBeg, lorads_int *LpMatIdx, double *LpMatElem);

/**
 * @brief Preprocess the problem data
 * @param ASolver Pointer to solver instance
 * @param BlkDims Array of block dimensions
 */
extern void LORADSPreprocess(lorads_solver *ASolver, lorads_int *BlkDims);

/**
 * @brief Clean up cone data
 * @param ASolver Pointer to solver instance
 */
extern void LORADSDestroyConeData(lorads_solver *ASolver);

/**
 * @brief Clean up preprocessing data
 * @param ASolver Pointer to solver instance
 */
extern void destroyPreprocess(lorads_solver *ASolver);

/**
 * @brief Determine initial ranks for SDP blocks
 * @param ASolver Pointer to solver instance
 * @param blkDims Array of block dimensions
 * @param timesRank Factor for rank estimation
 */
extern void LORADSDetermineRank(lorads_solver *ASolver, lorads_int *blkDims, double timesRank);

/**
 * @brief Detect sparsity patterns in SDP coefficients
 * @param ASolver Pointer to solver instance
 */
extern void detectSparsitySDPCoeff(lorads_solver *ASolver);

/**
 * @brief Initialize ALM variables
 * @param ASolver Pointer to solver instance
 * @param rankElem Array of rank elements
 * @param BlkDims Array of block dimensions
 * @param nBlks Number of blocks
 * @param nLpCols Number of LP columns
 * @param lbfgsHis Length of L-BFGS history
 */
extern void LORADSInitALMVars(lorads_solver *ASolver, lorads_int *rankElem, lorads_int *BlkDims, lorads_int nBlks, lorads_int nLpCols, lorads_int lbfgsHis);

/**
 * @brief Clean up ALM variables
 * @param ASolver Pointer to solver instance
 */
extern void LORADSDestroyALMVars(lorads_solver *ASolver);

/**
 * @brief Initialize ADMM variables
 * @param ASolver Pointer to solver instance
 * @param rankElem Array of rank elements
 * @param BlkDims Array of block dimensions
 * @param nBlks Number of blocks
 * @param nLpCols Number of LP columns
 */
extern void LORADSInitADMMVars(lorads_solver *ASolver, lorads_int *rankElem, lorads_int *BlkDims, lorads_int nBlks, lorads_int nLpCols);

/**
 * @brief Clean up ADMM variables
 * @param ASolver Pointer to solver instance
 */
extern void LORADSDestroyADMMVars(lorads_solver *ASolver);

/**
 * @brief Initialize function set
 * @param pfunc Pointer to function array
 * @param nLpCols Number of LP columns
 */
extern void LORADSInitFuncSet(lorads_func **pfunc, lorads_int nLpCols);

/**
 * @brief Check if all ranks are at maximum
 * @param asolver Pointer to solver instance
 * @param aug_factor Augmentation factor
 * @return 1 if all ranks are at maximum, 0 otherwise
 */
extern lorads_int CheckAllRankMax(lorads_solver *asolver, double aug_factor);

/**
 * @brief Augment ranks for SDP blocks
 * @param ASolver Pointer to solver instance
 * @param BlkDims Array of block dimensions
 * @param nBlks Number of blocks
 * @param aug_factor Augmentation factor
 * @return Status code indicating success or failure
 */
extern lorads_int AUG_RANK(lorads_solver *ASolver, lorads_int *BlkDims, lorads_int nBlks, double aug_factor);

/**
 * @brief End program and clean up resources
 * @param ASolver Pointer to solver instance
 */
extern void  LORADSEndProgram( lorads_solver *ASolver);

/**
 * @brief Print optimization results
 * @param pObj Primal objective value
 * @param dObj Dual objective value
 * @param constrVio Constraint violation
 * @param dualInfe Dual infeasibility
 * @param pdgap Primal-dual gap
 * @param constrVioInf Infinity norm of constraint violation
 * @param dualInfeInf Infinity norm of dual infeasibility
 */
extern void printRes(double pObj, double dObj, double constrVio, double dualInfe, double pdgap, double constrVioInf, double dualInfeInf);

/**
 * @brief Convert ALM state to ADMM state
 * @param ASolver Pointer to solver instance
 * @param params Solver parameters
 * @param alm_state ALM state
 * @param admm_state ADMM state
 */
extern void LORADS_ALMtoADMM(lorads_solver *ASolver, lorads_params *params, lorads_alm_state *alm_state, lorads_admm_state *admm_state);

/**
 * @brief Calculate dual infeasibility
 * @param ASolver Pointer to solver instance
 */
extern void calculate_dual_infeasibility_solver(lorads_solver *ASolver);

/**
 * @brief Scale dual variables
 * @param ASolver Pointer to solver instance
 * @param scaleTemp Temporary scaling array
 * @param scaleHis Scaling history array
 */
extern void objScale_dualvar(lorads_solver *ASolver, double *scaleTemp, double *scaleHis);

/**
 * @brief Calculate SDP constants
 * @param params Solver parameters
 * @param ASolver Pointer to solver instance
 * @param sdpConst Output SDP constants
 */
extern void cal_sdp_const(lorads_params *params, lorads_solver *ASolver, SDPConst *sdpConst);

/**
 * @brief Initialize solver state
 * @param params Solver parameters
 * @param ASolver Pointer to solver instance
 * @param alm_state_pointer ALM state
 * @param admm_state_pointer ADMM state
 * @param sdpConst SDP constants
 */
extern void initial_solver_state(lorads_params *params, lorads_solver *ASolver, lorads_alm_state *alm_state_pointer, lorads_admm_state *admm_state_pointer, SDPConst *sdpConst);

/**
 * @brief Perform reoptimization
 * @param params Solver parameters
 * @param ASolver Pointer to solver instance
 * @param alm_state_pointer ALM state
 * @param admm_state_pointer ADMM state
 * @param reopt_param Reoptimization parameter
 * @param reopt_alm_iter Number of ALM iterations for reoptimization
 * @param reopt_admm_iter Number of ADMM iterations for reoptimization
 * @param timeSolveStart Start time of solving
 * @param admm_bad_iter_flag Flag for bad ADMM iterations
 * @param reopt_level Level of reoptimization
 * @return Time taken for reoptimization
 */
extern double reopt(lorads_params *params, lorads_solver *ASolver, lorads_alm_state *alm_state_pointer, lorads_admm_state *admm_state_pointer, double *reopt_param, lorads_int *reopt_alm_iter, lorads_int *reopt_admm_iter, double timeSolveStart, int *admm_bad_iter_flag, int reopt_level);

/**
 * @brief Print problem information
 * @param ASolver Pointer to solver instance
 */
extern void printfProbInfo(lorads_solver *ASolver);

#endif