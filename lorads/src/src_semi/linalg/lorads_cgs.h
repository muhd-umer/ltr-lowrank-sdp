/**
 * @file lorads_cgs.h
 * @brief LORADS Conjugate Gradient Solver Interface
 * @details This header defines the interface for the conjugate gradient solver used in LORADS.
 * The solver is used for efficiently solving linear systems in both ALM and ADMM phases.
 * Key features include:
 * - Efficient handling of low-rank matrices
 * - Support for both SDP and LP components
 * - Adaptive tolerance control
 * - Preconditioning options
 * - Restart strategies
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_CG_H
#define LORADS_CG_H

#include "def_lorads_elements.h"
#include "def_lorads_cgs.h"
#include "lorads_solver.h"

/**
 * @struct admmCG
 * @brief Structure for ADMM conjugate gradient solver
 * @details Contains data needed for CG solves in ADMM:
 * - ACone: SDP cone data
 * - noUpdateVar: Fixed variable
 * - UpdateVarShell: Variable to update
 * - weight: Weight vector for preconditioning
 */
typedef struct
{
    lorads_sdp_cone *ACone;           ///< SDP cone data
    lorads_sdp_dense *noUpdateVar;    ///< Fixed variable
    lorads_sdp_dense *UpdateVarShell; ///< Variable to update
    double *weight;                   ///< Weight vector
} admmCG;

/**
 * @brief Create a new conjugate gradient solver
 * @param pCGSolver Pointer to store the new solver
 * @param blkDim Block dimension
 * @param rank Rank of the matrix
 * @param nConstr Number of constraints
 * @details Initializes a new CG solver with specified dimensions:
 * 1. Allocates memory for vectors
 * 2. Sets up initial state
 * 3. Configures solver parameters
 */
extern void LORADSCGSolverCreate(lorads_cg_linsys **pCGSolver, lorads_int blkDim, 
                                lorads_int rank, lorads_int nConstr);

/**
 * @brief Recreate a conjugate gradient solver
 * @param pCGSolver Pointer to solver to recreate
 * @param blkDim Block dimension
 * @param rank Rank of the matrix
 * @param nConstr Number of constraints
 * @details Reinitializes an existing CG solver with new dimensions:
 * 1. Frees old memory
 * 2. Allocates new memory
 * 3. Resets solver state
 */
extern void LORADSCGSolverReCreate(lorads_cg_linsys **pCGSolver, lorads_int blkDim, 
                                  lorads_int rank, lorads_int nConstr);

/**
 * @brief Solve a linear system using conjugate gradient
 * @param linSys Linear system to solve
 * @param x Solution vector
 * @param b Right-hand side vector
 * @param cg_tol Convergence tolerance
 * @param cg_maxIter Maximum iterations
 * @details Solves Ax = b using conjugate gradient method:
 * 1. Initialize vectors
 * 2. Compute initial residual
 * 3. Iterate until convergence
 * 4. Check convergence criteria
 */
extern void CGSolve(void *linSys, double *x, double *b, double cg_tol, 
                   lorads_int cg_maxIter);

/**
 * @brief Set data for conjugate gradient solver
 * @param cg CG solver instance
 * @param MMat Matrix data
 * @param Mvec Matrix-vector multiplication function
 * @details Sets up the matrix data and multiplication function:
 * 1. Store matrix data
 * 2. Set multiplication function
 * 3. Initialize solver state
 */
extern void CGSetData(lorads_cg_linsys *cg, void *MMat, 
                     void (*Mvec)(void *, double *, double *));

/**
 * @brief Clear conjugate gradient solver resources
 * @param pCGSolver Pointer to solver to clear
 * @details Frees all memory associated with the solver:
 * 1. Free vectors
 * 2. Free solver structure
 * 3. Reset pointers
 */
extern void CGSolverClear(void *pCGSolver);

/**
 * @brief Destroy conjugate gradient solver
 * @param ASolver Pointer to main solver
 * @details Cleans up all CG solvers in the main solver:
 * 1. Clear each cone's solver
 * 2. Free solver array
 * 3. Reset pointers
 */
extern void LORADSCGDestroy(lorads_solver *ASolver);

#endif //LORADS_CG_H
