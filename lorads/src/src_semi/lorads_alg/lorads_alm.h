/**
 * @file lorads_alm.h
 * @brief Header file for Augmented Lagrangian Method (ALM) implementation in LORADS
 * @details This file declares functions for the ALM optimization algorithm, including
 * gradient calculations, line search, L-BFGS direction computation, and variable updates.
 */

#ifndef LORADS_ALM_H
#define LORADS_ALM_H


#include "def_lorads_elements.h"
#include "lorads_solver.h"

/**
 * @brief Calculates gradient for LP variables in ALM
 * @param ASolver Pointer to solver structure
 * @param rLp LP rank matrix
 * @param gradLp LP gradient matrix
 * @param R Array of SDP rank matrices
 * @param Grad Array of SDP gradient matrices
 * @param lagNormSquare Pointer to store Lagrangian norm squared
 * @param rho Penalty parameter
 */
extern void ALMCalGradLP(lorads_solver *ASolver, lorads_lp_dense *rLp, lorads_lp_dense *gradLp, lorads_sdp_dense **R, lorads_sdp_dense **Grad, double *lagNormSquare, double rho);

/**
 * @brief Sets gradient for LP variables in ALM
 * @param ASolver Pointer to solver structure
 * @param lp_cone LP cone structure
 * @param rLp LP rank matrix
 * @param Grad LP gradient matrix
 * @param iCol Column index
 * @param rho Penalty parameter
 */
extern void ALMSetGradLP(lorads_solver *ASolver, lorads_lp_cone *lp_cone, lorads_lp_dense *rLp, lorads_lp_dense *Grad, lorads_int iCol, double rho);

/**
 * @brief Calculates gradient for SDP variables in ALM
 * @param ASolver Pointer to solver structure
 * @param rLpDummy Dummy LP rank matrix
 * @param gradLpDummy Dummy LP gradient matrix
 * @param R Array of SDP rank matrices
 * @param Grad Array of SDP gradient matrices
 * @param lagNormSquare Pointer to store Lagrangian norm squared
 * @param rho Penalty parameter
 */
extern void ALMCalGrad(lorads_solver *ASolver, lorads_lp_dense *rLpDummy, lorads_lp_dense *gradLpDummy, lorads_sdp_dense **R, lorads_sdp_dense **Grad, double *lagNormSquare, double rho);

/**
 * @brief Sets gradient for SDP variables in ALM
 * @param ASolver Pointer to solver structure
 * @param ACone SDP cone structure
 * @param R SDP rank matrix
 * @param Grad SDP gradient matrix
 * @param iCone Cone index
 * @param rho Penalty parameter
 */
extern void ALMSetGrad(lorads_solver *ASolver, lorads_sdp_cone *ACone, lorads_sdp_dense *R, lorads_sdp_dense *Grad, lorads_int iCone, double rho);

/**
 * @brief Computes nth root of a number
 * @param base Base number
 * @param n Root order
 * @return nth root of base
 */
double LORADSnthroot(double base, lorads_int n);

/**
 * @brief Solves cubic equation
 * @param a Coefficient of x^3
 * @param b Coefficient of x^2
 * @param c Coefficient of x
 * @param d Constant term
 * @param res Array to store roots
 * @return Number of real roots
 */
extern lorads_int LORADScubic_equation(double a, double b, double c, double d, double *res);

/**
 * @brief Performs line search in ALM
 * @param rho Penalty parameter
 * @param n Dimension
 * @param lambd Array of lambda values
 * @param p1 First parameter
 * @param p2 Second parameter
 * @param q0 Array for q0 values
 * @param q1 Array for q1 values
 * @param q2 Array for q2 values
 * @param tau Step size
 * @return Status code
 */
extern lorads_int ALMLineSearch(double rho, lorads_int n, double *lambd, double p1, double p2, double *q0, double *q1, double *q2, double *tau);

/**
 * @brief Computes L-BFGS direction for SDP variables
 * @param params Solver parameters
 * @param ASolver Pointer to solver structure
 * @param head L-BFGS history node
 * @param gradLpDummy Dummy LP gradient matrix
 * @param dDummy Dummy LP direction matrix
 * @param Grad Array of SDP gradient matrices
 * @param D Array of SDP direction matrices
 * @param innerIter Inner iteration count
 */
extern void LBFGSDirection(lorads_params *params, lorads_solver *ASolver, lbfgs_node *head, lorads_lp_dense *gradLpDummy, lorads_lp_dense *dDummy, lorads_sdp_dense **Grad, lorads_sdp_dense **D, lorads_int innerIter);

/**
 * @brief Computes L-BFGS direction for LP variables
 * @param params Solver parameters
 * @param ASolver Pointer to solver structure
 * @param head L-BFGS history node
 * @param gradLp LP gradient matrix
 * @param d LP direction matrix
 * @param Grad Array of SDP gradient matrices
 * @param D Array of SDP direction matrices
 * @param innerIter Inner iteration count
 */
extern void LBFGSDirectionLP(lorads_params *params, lorads_solver *ASolver, lbfgs_node *head, lorads_lp_dense *gradLp, lorads_lp_dense *d, lorads_sdp_dense **Grad, lorads_sdp_dense **D, lorads_int innerIter);

/**
 * @brief Uses gradient as direction for SDP variables
 * @param ASolver Pointer to solver structure
 * @param dDummy Dummy LP direction matrix
 * @param gradLPDummy Dummy LP gradient matrix
 * @param D Array of SDP direction matrices
 * @param Grad Array of SDP gradient matrices
 */
extern void LBFGSDirectionUseGrad(lorads_solver *ASolver, lorads_lp_dense *dDummy, lorads_lp_dense *gradLPDummy, lorads_sdp_dense **D, lorads_sdp_dense **Grad);

/**
 * @brief Uses gradient as direction for LP variables
 * @param ASolver Pointer to solver structure
 * @param d LP direction matrix
 * @param gradLP LP gradient matrix
 * @param D Array of SDP direction matrices
 * @param Grad Array of SDP gradient matrices
 */
extern void LBFGSDirectionUseGradLP(lorads_solver *ASolver, lorads_lp_dense *d, lorads_lp_dense *gradLP, lorads_sdp_dense **D, lorads_sdp_dense **Grad);

/**
 * @brief Calculates q1, q2, and p12 for LP variables
 * @param ASolver Pointer to solver structure
 * @param rLp LP rank matrix
 * @param d LP direction matrix
 * @param R Array of SDP rank matrices
 * @param D Array of SDP direction matrices
 * @param q1 Pointer to store q1 value
 * @param q2 Pointer to store q2 value
 * @param p12 Pointer to store p12 value
 */
extern void ALMCalq12p12LP(lorads_solver *ASolver, lorads_lp_dense *rLp, lorads_lp_dense *d, lorads_sdp_dense **R, lorads_sdp_dense **D, double *q1, double *q2, double *p12);

/**
 * @brief Sets y as negative gradient for SDP variables
 * @param ASolver Pointer to solver structure
 * @param gradLpDummy Dummy LP gradient matrix
 * @param Grad Array of SDP gradient matrices
 */
extern void SetyAsNegGrad(lorads_solver *ASolver, lorads_lp_dense *gradLpDummy, lorads_sdp_dense **Grad);

/**
 * @brief Sets y as negative gradient for LP variables
 * @param ASolver Pointer to solver structure
 * @param gradLp LP gradient matrix
 * @param Grad Array of SDP gradient matrices
 */
extern void SetyAsNegGradLP(lorads_solver *ASolver, lorads_lp_dense *gradLp, lorads_sdp_dense **Grad);

/**
 * @brief Sets L-BFGS history for SDP variables
 * @param ASolver Pointer to solver structure
 * @param gradLpDummy Dummy LP gradient matrix
 * @param dDummy Dummy LP direction matrix
 * @param Grad Array of SDP gradient matrices
 * @param D Array of SDP direction matrices
 * @param tau Step size
 */
extern void setlbfgsHisTwo(lorads_solver *ASolver, lorads_lp_dense *gradLpDummy, lorads_lp_dense *dDummy, lorads_sdp_dense **Grad, lorads_sdp_dense **D, double tau);

/**
 * @brief Sets L-BFGS history for LP variables
 * @param ASolver Pointer to solver structure
 * @param gradLp LP gradient matrix
 * @param d LP direction matrix
 * @param Grad Array of SDP gradient matrices
 * @param D Array of SDP direction matrices
 * @param tau Step size
 */
extern void setlbfgsHisTwoLP(lorads_solver *ASolver, lorads_lp_dense *gradLp, lorads_lp_dense *d, lorads_sdp_dense **Grad, lorads_sdp_dense **D, double tau);

/**
 * @brief Updates variables in ALM for SDP variables
 * @param ASolver Pointer to solver structure
 * @param rLpDummy Dummy LP rank matrix
 * @param dDummy Dummy LP direction matrix
 * @param R Array of SDP rank matrices
 * @param D Array of SDP direction matrices
 * @param tau Step size
 */
extern void ALMupdateVar(lorads_solver *ASolver, lorads_lp_dense *rLpDummy, lorads_lp_dense *dDummy, lorads_sdp_dense **R, lorads_sdp_dense **D, double tau);

/**
 * @brief Updates variables in ALM for LP variables
 * @param ASolver Pointer to solver structure
 * @param rLp LP rank matrix
 * @param d LP direction matrix
 * @param R Array of SDP rank matrices
 * @param D Array of SDP direction matrices
 * @param tau Step size
 */
extern void ALMupdateVarLP(lorads_solver *ASolver, lorads_lp_dense *rLp, lorads_lp_dense *d, lorads_sdp_dense **R, lorads_sdp_dense **D, double tau);

/**
 * @brief Calculates q1, q2, and p12 for SDP variables
 * @param ASolver Pointer to solver structure
 * @param rLpDummy Dummy LP rank matrix
 * @param dDummy Dummy LP direction matrix
 * @param R Array of SDP rank matrices
 * @param D Array of SDP direction matrices
 * @param q1 Pointer to store q1 value
 * @param q2 Pointer to store q2 value
 * @param p12 Pointer to store p12 value
 */
extern void ALMCalq12p12(lorads_solver *ASolver, lorads_lp_dense *rLpDummy, lorads_lp_dense *dDummy, lorads_sdp_dense **R, lorads_sdp_dense **D, double *q1, double *q2, double *p12);

/**
 * @brief Calculates objective value for ALM
 * @param ASolver Pointer to solver structure
 */
extern void LORADSCalObjRR_ALM(lorads_solver *ASolver);

/**
 * @brief Calculates objective value for ALM with LP variables
 * @param ASolver Pointer to solver structure
 */
extern void LORADSCalObjRR_ALM_LP(lorads_solver *ASolver);

/**
 * @brief Performs reoptimization in ALM
 * @param params Solver parameters
 * @param ASolver Pointer to solver structure
 * @param alm_iter_state ALM iteration state
 * @param early_stop Whether to stop early
 * @param rho_update_factor Factor for updating penalty parameter
 * @param timeSolveStart Start time of solving
 * @return Status code
 */
extern lorads_int LORADS_ALMOptimize_reopt(lorads_params *params, lorads_solver *ASolver, lorads_alm_state *alm_iter_state, bool early_stop, double rho_update_factor, double timeSolveStart);

/**
 * @brief Performs ALM optimization
 * @param params Solver parameters
 * @param ASolver Pointer to solver structure
 * @param alm_iter_state ALM iteration state
 * @param rho_update_factor Factor for updating penalty parameter
 * @param timeSolveStart Start time of solving
 * @return Status code
 */
extern lorads_int LORADS_ALMOptimize(lorads_params *params, lorads_solver *ASolver, lorads_alm_state *alm_iter_state, double rho_update_factor, double timeSolveStart);

#endif