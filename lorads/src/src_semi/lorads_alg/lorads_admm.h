/**
 * @file lorads_admm.h
 * @brief LORADS ADMM Solver Header
 * @details This header file contains declarations for the Alternating Direction Method of Multipliers (ADMM)
 * implementation in LORADS. It includes functions for:
 * - ADMM optimization with and without reoptimization
 * - Objective value calculation
 * - Variable updates for both SDP and LP components
 * - Averaging operations for U and V matrices
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_ADMM_H
#define LORADS_ADMM_H

#include "lorads_solver.h"
#include "lorads.h"

/**
 * @brief Optimize using ADMM method
 * @param params Solver parameters
 * @param ASolver Pointer to solver instance
 * @param admm_iter_state ADMM iteration state
 * @param iter_celling Maximum number of iterations
 * @param timeSolveStart Start time of solving
 * @return Status code indicating success or failure
 */
extern lorads_int LORADSADMMOptimize(lorads_params *params, lorads_solver *ASolver, lorads_admm_state *admm_iter_state, lorads_int iter_celling, double timeSolveStart);

/**
 * @brief Optimize using ADMM method with reoptimization
 * @param params Solver parameters
 * @param ASolver Pointer to solver instance
 * @param admm_iter_state ADMM iteration state
 * @param iter_celling Maximum number of iterations
 * @param timeSolveStart Start time of solving
 * @return Status code indicating success or failure
 */
extern lorads_int LORADSADMMOptimize_reopt(lorads_params *params, lorads_solver *ASolver, lorads_admm_state *admm_iter_state, lorads_int iter_celling, double timeSolveStart);

/**
 * @brief Calculate objective value for ADMM using U and V matrices
 * @param ASolver Pointer to solver instance
 */
extern void LORADSCalObjUV_ADMM(lorads_solver *ASolver);

/**
 * @brief Calculate objective value for ADMM including LP variables
 * @param ASolver Pointer to solver instance
 */
extern void LORADSCalObjUV_ADMM_LP(lorads_solver *ASolver);

/**
 * @brief Average U and V matrices for an SDP cone
 * @param U First matrix
 * @param V Second matrix
 * @param UVavg Output averaged matrix
 */
extern void averageUV(lorads_sdp_dense *U, lorads_sdp_dense *V, lorads_sdp_dense *UVavg);

/**
 * @brief Average U and V matrices for LP variables
 * @param ulp First LP matrix
 * @param vlp Second LP matrix
 * @param uvavg Output averaged matrix
 */
extern void averageUVLP(lorads_lp_dense *ulp, lorads_lp_dense *vlp, lorads_lp_dense *uvavg);

/**
 * @brief Update SDP variables for one cone
 * @param ASolver Pointer to solver instance
 * @param updateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param iCone Cone index
 * @param rho Penalty parameter
 * @param CG_tol Conjugate gradient tolerance
 * @param CG_maxIter Maximum CG iterations
 */
extern void LORADSUpdateSDPVarOne(lorads_solver *ASolver, lorads_sdp_dense *updateVar, lorads_sdp_dense *noUpdateVar, lorads_int iCone, double rho, double CG_tol, lorads_int CG_maxIter);

/**
 * @brief Update SDP variables for one cone with positive S matrix
 * @param ASolver Pointer to solver instance
 * @param updateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param S Positive S matrix
 * @param iCone Cone index
 * @param rho Penalty parameter
 * @param CG_tol Conjugate gradient tolerance
 * @param CG_maxIter Maximum CG iterations
 */
extern void LORADSUpdateSDPVarOne_positive_S(lorads_solver *ASolver, lorads_sdp_dense *updateVar, lorads_sdp_dense *noUpdateVar, lorads_sdp_dense *S, lorads_int iCone, double rho, double CG_tol, lorads_int CG_maxIter);

/**
 * @brief Update SDP variables for one cone with negative S matrix
 * @param ASolver Pointer to solver instance
 * @param updateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param S Negative S matrix
 * @param iCone Cone index
 * @param rho Penalty parameter
 * @param CG_tol Conjugate gradient tolerance
 * @param CG_maxIter Maximum CG iterations
 */
extern void LORADSUpdateSDPVarOne_negative_S(lorads_solver *ASolver, lorads_sdp_dense *updateVar, lorads_sdp_dense *noUpdateVar, lorads_sdp_dense *S, lorads_int iCone, double rho, double CG_tol, lorads_int CG_maxIter);

/**
 * @brief Update LP variables for one column
 * @param ASolver Pointer to solver instance
 * @param UpdateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param iCol Column index
 * @param rho Penalty parameter
 */
extern void LORADSUpdateLPVarOne(lorads_solver *ASolver,  double *UpdateVar, double *noUpdateVar, lorads_int iCol, double rho);

/**
 * @brief Update LP variables for one column with positive S
 * @param ASolver Pointer to solver instance
 * @param UpdateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param iCol Column index
 * @param rho Penalty parameter
 * @param sLp Positive S vector
 */
extern void LORADSUpdateLPVarOne_positive_S(lorads_solver *ASolver,  double *UpdateVar, double *noUpdateVar, lorads_int iCol, double rho, double *sLp);

/**
 * @brief Update LP variables for one column with negative S
 * @param ASolver Pointer to solver instance
 * @param UpdateVar Variables to update
 * @param noUpdateVar Variables to keep unchanged
 * @param iCol Column index
 * @param rho Penalty parameter
 * @param sLp Negative S vector
 */
extern void LORADSUpdateLPVarOne_negative_S(lorads_solver *ASolver,  double *UpdateVar, double *noUpdateVar, lorads_int iCol, double rho, double *sLp);

#endif