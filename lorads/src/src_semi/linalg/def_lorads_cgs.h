/**
 * @file def_lorads_cgs.h
 * @brief LORADS Conjugate Gradient Solver Definitions
 * @details This header file defines the data structures and enumerations used in the
 * conjugate gradient solver implementation for LORADS. It includes definitions for
 * iteration status, linear system data structures, and solver parameters.
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef DEF_LORADS_CGS
#define DEF_LORADS_CGS

#include "def_lorads_elements.h"

/**
 * @brief Status codes for conjugate gradient iterations
 * @details Defines the possible outcomes of a conjugate gradient iteration:
 * - CG_ITER_STATUS_OK: Iteration completed successfully
 * - CG_ITER_STATUS_NUMERICAL: Numerical issues encountered
 * - CG_ITER_STATUS_MAXITER: Maximum iterations reached
 * - CG_ITER_STATUS_FAILED: General failure in iteration
 */
typedef enum {
    CG_ITER_STATUS_OK,        ///< Iteration completed successfully
    CG_ITER_STATUS_NUMERICAL, ///< Numerical issues encountered
    CG_ITER_STATUS_MAXITER,   ///< Maximum iterations reached
    CG_ITER_STATUS_FAILED     ///< General failure in iteration
} iter_status_lorads_cg;

/**
 * @brief Linear system data structure for conjugate gradient solver
 * @details This structure holds all necessary data for the conjugate gradient solver,
 * including problem dimensions, sparse matrix data, iteration vectors, and solver parameters.
 */
typedef struct {
    lorads_int nr;            ///< Dimension * rank
    lorads_int nnzRowOneCol;  ///< Number of non-zero rows in one column
    lorads_int nnzRow;        ///< Number of non-zero rows * rank
    lorads_int m;             ///< Number of non-zero constraints
    lorads_int *rowIdxInv;    ///< Inverse row indices for sparse row format
    lorads_int *rowIdx;       ///< Row indices for sparse row format

    lorads_sdp_dense **a;     ///< Dense matrix data (not allocated at creation)
    double *A;                ///< Sparse row format matrix data

    double *rIter;            ///< Residual vector r
    double *rIterNew;         ///< New residual vector r_new
    double *pIter;            ///< Search direction vector p
    double *qIter;            ///< Matrix-vector product q = A*p
    double *qIterNew;         ///< New matrix-vector product q_new
    double *QIter;            ///< Preconditioned matrix-vector product Q
    double *iterVec;          ///< Solution vector x

    /* Pre-conditioner */
    lorads_int useJacobi;     ///< Flag for using Jacobi preconditioner
//    double *JacobiPrecond;   ///< Jacobi preconditioner matrix (commented out)

    // Statistics
    double solveTime;         ///< Total solve time
    double cgDuration;        ///< Time spent in CG iterations
    double resiNorm;          ///< Residual norm
    lorads_int iter;          ///< Number of iterations performed

    iter_status_lorads_cg solStatus;  ///< Solution status

    // parameters
    lorads_int nRestartFreq;  ///< Frequency of CG restarts

    void  *MMat;              ///< Preconditioner matrix
    void (*Mvec)(void *, double *, double *);  ///< Preconditioner matrix-vector multiplication function

} lorads_cg_linsys;

#endif