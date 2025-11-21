/**
 * @file lorads_alg_common.h
 * @brief LORADS Common Algorithm Functions Header
 * @details This header file contains common functions used across different optimization algorithms
 * in LORADS. It includes functions for:
 * - Constraint value initialization and updates
 * - Variable updates for SDP and LP components
 * - Error calculation and monitoring
 * - Objective value computation
 * - Duality gap calculations
 * - Convergence checking
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_ALG_COMMON_H
#define LORADS_ALG_COMMON_H

#include "def_lorads_elements.h"
#include "lorads_solver.h"

/**
 * @brief Initialize constraint values for all SDP cones
 * @param ASolver Pointer to solver instance
 * @param uLpDummy Dummy LP matrix for U
 * @param vLpDummy Dummy LP matrix for V
 * @param U Array of SDP matrices for U
 * @param V Array of SDP matrices for V
 */
extern void LORADSInitConstrValAll(lorads_solver *ASolver, lorads_lp_dense *uLpDummy, lorads_lp_dense *vLpDummy, lorads_sdp_dense **U, lorads_sdp_dense **V);

/**
 * @brief Initialize constraint values for all components including LP
 * @param ASolver Pointer to solver instance
 * @param uLp LP matrix for U
 * @param vLp LP matrix for V
 * @param U Array of SDP matrices for U
 * @param V Array of SDP matrices for V
 */
extern void LORADSInitConstrValAllLP(lorads_solver *ASolver, lorads_lp_dense *uLp, lorads_lp_dense *vLp, lorads_sdp_dense **U, lorads_sdp_dense **V);

/**
 * @brief Initialize sum of constraint values for SDP cones
 * @param ASolver Pointer to solver instance
 */
extern void LORADSInitConstrValSum(lorads_solver *ASolver);

/**
 * @brief Initialize sum of constraint values including LP components
 * @param ASolver Pointer to solver instance
 */
extern void LORADSInitConstrValSumLP(lorads_solver *ASolver);

/**
 * @brief Update SDP variables for all cones
 * @param ASolver Pointer to solver instance
 * @param rho Penalty parameter
 * @param CG_tol Conjugate gradient tolerance
 * @param CG_maxIter Maximum CG iterations
 */
extern void LORADSUpdateSDPVar(lorads_solver *ASolver, double rho, double CG_tol, lorads_int CG_maxIter);

/**
 * @brief Update both SDP and LP variables
 * @param ASolver Pointer to solver instance
 * @param rho Penalty parameter
 * @param CG_tol Conjugate gradient tolerance
 * @param CG_maxIter Maximum CG iterations
 */
extern void LORADSUpdateSDPLPVar(lorads_solver *ASolver, double rho, double CG_tol, lorads_int CG_maxIter);

/**
 * @brief Compute UV^T for SDP coefficients
 * @param UVt_w_sum Output weighted sum of UV^T
 * @param U Input U matrix
 * @param V Input V matrix
 */
extern void LORADSUVt(sdp_coeff *UVt_w_sum, lorads_sdp_dense *U, lorads_sdp_dense *V);

/**
 * @brief Copy R matrices to V matrices for SDP cones
 * @param rLpDummy Dummy LP matrix for R
 * @param vlpDummy Dummy LP matrix for V
 * @param R Array of input R matrices
 * @param V Array of output V matrices
 * @param nCones Number of cones
 */
extern void copyRtoV(lorads_lp_dense *rLpDummy, lorads_lp_dense *vlpDummy, lorads_sdp_dense **R, lorads_sdp_dense **V, lorads_int nCones);

/**
 * @brief Copy R matrices to V matrices including LP components
 * @param rLp LP matrix for R
 * @param vlp LP matrix for V
 * @param R Array of input R matrices
 * @param V Array of output V matrices
 * @param nCones Number of cones
 */
extern void copyRtoVLP(lorads_lp_dense *rLp, lorads_lp_dense *vlp, lorads_sdp_dense **R, lorads_sdp_dense **V, lorads_int nCones);

/**
 * @brief Update DIMACS error metrics for ALM
 * @param ASolver Pointer to solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP vector r
 * @param r2 LP vector r2
 */
extern void LORADSUpdateDimacsErrorALM(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2);

/**
 * @brief Update DIMACS error metrics for ALM including LP
 * @param ASolver Pointer to solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP vector r
 * @param r2 LP vector r2
 */
extern void LORADSUpdateDimacsErrorALMLP(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2);

/**
 * @brief Update DIMACS error metrics for ADMM
 * @param ASolver Pointer to solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param u LP vector u
 * @param v LP vector v
 */
extern void LORADSUpdateDimacsErrorADMM(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, lorads_lp_dense *u, lorads_lp_dense *v);

/**
 * @brief Update DIMACS error metrics for ADMM including LP
 * @param ASolver Pointer to solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param u LP vector u
 * @param v LP vector v
 */
extern void LORADSUpdateDimacsErrorADMMLP(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, lorads_lp_dense *u, lorads_lp_dense *v);

/**
 * @brief Calculate infinity norm of objective
 * @param ASolver Pointer to solver instance
 */
extern void LORADSNrmInfObj(lorads_solver *ASolver);

/**
 * @brief Update dual variables
 * @param ASolver Pointer to solver instance
 * @param rho Penalty parameter
 */
extern void LORADSUpdateDualVar(lorads_solver *ASolver, double rho);

/**
 * @brief Calculate dual objective value
 * @param ASolver Pointer to solver instance
 */
extern void LORADSCalDualObj(lorads_solver *ASolver);

/**
 * @brief Calculate nuclear norm
 * @param ASolver Pointer to solver instance
 */
extern void LORADSNuclearNorm(lorads_solver *ASolver);

/**
 * @brief Check DIMACS error criteria for ALM
 * @param ASolver Pointer to solver instance
 */
extern void LORADSCheckDimacErrALMCriteria(lorads_solver *ASolver);

/**
 * @brief Calculate objective and constraint values for all SDP cones
 * @param ASolver Pointer to solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param UVobjVal Output objective value
 */
extern void LORADSObjConstrValAll(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, double *UVobjVal);

/**
 * @brief Calculate objective and constraint values including LP components
 * @param ASolver Pointer to solver instance
 * @param uLp LP matrix for U
 * @param vLp LP matrix for V
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param UVobjVal Output objective value
 */
extern void LORADSObjConstrValAllLP(lorads_solver *ASolver, lorads_lp_dense *uLp, lorads_lp_dense *vLp, lorads_sdp_dense **U, lorads_sdp_dense **V, double *UVobjVal);

/**
 * @brief Check solver status and convergence
 * @param ASolver Pointer to solver instance
 * @return Status code indicating convergence or error
 */
extern lorads_int LORADSCheckSolverStatus(lorads_solver *ASolver);

#endif