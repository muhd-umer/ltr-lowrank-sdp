/**
 * @file lorads_cgs.c
 * @brief LORADS Conjugate Gradient Solver Implementation
 * @details This file implements the conjugate gradient solver for LORADS, providing
 * functions for creating, managing, and solving linear systems using the conjugate
 * gradient method. It includes support for preconditioning and various convergence
 * criteria.
 * 
 * @author LORADS Team
 * @date 2024
 */

#include "lorads.h"
#include "lorads_utils.h"
#include "lorads_vec_opts.h"
#include "def_lorads_cgs.h"
#include "lorads_solver.h"

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief Create a new conjugate gradient solver for a single cone
 * @param pCGSolver Pointer to store the created solver
 * @param blkDim Dimension of the cone (n)
 * @param rank Rank of the cone (r)
 * @param nConstr Number of constraints (m)
 * @details Initializes a new conjugate gradient solver with the specified dimensions
 * and allocates necessary memory for iteration vectors and solver parameters.
 */
extern void LORADSCGSolverCreate(lorads_cg_linsys **pCGSolver, lorads_int blkDim, lorads_int rank, lorads_int nConstr){
    /*  create a cg solver for one cone
        blkDim is the dimension of cone: n
        rank is the rank of cone: r
        nConstr is the number of constraints: m
     */
    LORADS_NULLCHECK(pCGSolver);

    lorads_cg_linsys *cg = NULL;
    LORADS_INIT(cg, lorads_cg_linsys, 1);
    LORADS_MEMCHECK(cg);

    lorads_int nr = blkDim * rank; // dim = n * r

    cg->nr = nr; // set dimension of liear system
    cg->m = nConstr;
    cg->iter = 0;

    LORADS_INIT(cg->rIter, double, nr);
    LORADS_INIT(cg->rIterNew, double, nr);
    LORADS_INIT(cg->pIter, double, nr);
    LORADS_INIT(cg->qIter, double, nr);
    LORADS_INIT(cg->qIterNew, double, nr);
    LORADS_INIT(cg->QIter, double, nr);

    cg->useJacobi = 0;

//    LORADS_INIT(cg->JacobiPrecond, double, nr);

    /* Set parameters */
    cg->nRestartFreq = 20;
    cg->solveTime = 0.0;

    cg->a = NULL;
    *pCGSolver = cg;
}

/**
 * @brief Recreate a conjugate gradient solver with new dimensions
 * @param pCGSolver Pointer to the solver to recreate
 * @param blkDim New dimension of the cone
 * @param rank New rank of the cone
 * @param nConstr New number of constraints
 * @details Reinitializes an existing solver with new dimensions, reallocating
 * all necessary vectors and resetting solver parameters.
 */
extern void LORADSCGSolverReCreate(lorads_cg_linsys **pCGSolver, lorads_int blkDim, lorads_int rank, lorads_int nConstr){
    lorads_cg_linsys *cg = *pCGSolver;
    lorads_int nr = blkDim * rank;
    cg->nr = nr; // set dimension of liear system
    cg->m = nConstr;
    cg->iter = 0;
    LORADS_FREE(cg->rIter);
    LORADS_INIT(cg->rIter, double, nr);
    LORADS_FREE(cg->rIterNew);
    LORADS_INIT(cg->rIterNew, double, nr);
    LORADS_FREE(cg->pIter);
    LORADS_INIT(cg->pIter, double, nr);
    LORADS_FREE(cg->qIter);
    LORADS_INIT(cg->qIter, double, nr);
    LORADS_FREE(cg->qIterNew);
    LORADS_INIT(cg->qIterNew, double, nr);
    LORADS_FREE(cg->QIter);
    LORADS_INIT(cg->QIter, double, nr);
//    LORADS_FREE(cg->JacobiPrecond);
//    LORADS_INIT(cg->JacobiPrecond, double, nr);
}

/**
 * @brief Set matrix-vector multiplication function for the solver
 * @param cg Pointer to the conjugate gradient solver
 * @param MMat Pointer to the matrix data
 * @param Mvec Function pointer for matrix-vector multiplication
 * @details Sets the matrix data and multiplication function used in the solver.
 */
extern void CGSetData(lorads_cg_linsys *cg, void *MMat, void (*Mvec) (void *, double *, double *)){
//    if (cg->MMat || cg->Mvec){
//        return;
//    }
    cg->MMat = MMat;
    cg->Mvec = Mvec;
}

/**
 * @brief Solve a linear system using conjugate gradient method
 * @param linSys Pointer to the linear system solver
 * @param x Solution vector (input/output)
 * @param b Right-hand side vector
 * @param cg_tol Convergence tolerance
 * @param cg_maxIter Maximum number of iterations
 * @details Implements the conjugate gradient algorithm with:
 * - Automatic restart based on frequency
 * - Convergence checking using relative residual norm
 * - Numerical stability checks
 * - Performance statistics collection
 */
extern void CGSolve(void *linSys, double *x, double *b, double cg_tol, lorads_int cg_maxIter){

    lorads_cg_linsys *cg = (lorads_cg_linsys *)linSys;
    cg->solStatus = CG_ITER_STATUS_MAXITER; // Initialize to max iter

    /* CG working scalers */
    double qTr = 0.0; // for alpha, beta
    double qTrNew = 0.0; // for beta 
    double pTQ = 0.0; // for alpha
    double alpha = 0.0;
    double beta = 0.0;
    double resiNorm = 0.0; // for stopping 
    double bNorm = 0.0; // for rescale

    double *r = cg->rIter;
    double *p = cg->pIter;
    double *q = cg->qIter;
    double *qNew = cg->qIterNew;
    double *Q= cg->QIter;
    lorads_int nr = cg->nr;
    lorads_int m = cg->m;

    /* Get restart frequency */
    lorads_int nRestartFreq = LORADS_MAX(cg->nRestartFreq, 20);
    // set constant
    lorads_int incx = 1;
    double one = 1.0;
    double minusOne = -1.0;

    // start solving
    double cgStartTime = LUtilGetTimeStamp();
    double cgDuration = 0.0;

    /* Compute initial b norm */
    bNorm = nrm1(&nr, b, &incx);
    //    if ( bNorm > 1e-8 ) {
    //        scaleFactor = one/bNorm;
    //        // b = b / bNorm;
    //        scal(&nr, &scaleFactor, b, &incx);
    //        // a = a / bNorm;
    //        double sqrtScale = sqrt(scaleFactor);
    //        // lorads_int dim = nr * m;
    //        // scal(&dim, &sqrtScale, a, &incx);
    //        for (lorads_int i = 0; i < cg->m; ++i){
    //            a[i]->scal(a[i]->rk_mat, sqrtScale);
    //        }
    //    }

#ifdef LORADS_CG_DEBUG
    lorads_int dim = nr * m;
    double aNorm;
    aNorm = nrm2(&dim, a, &incx);
    lorads_printf("a norm after rescale %f\n", aNorm);
#endif

    // change a and initial solution
    // double *x= updateVar->matElem;

#ifdef LORADS_CG_DEBUG
    lorads_int Adim = nr * nr;
    double ANorm;
    ANorm = nrm2(&Adim, a, &incx);
    lorads_printf("a norm after solver Init %f\n", ANorm);
#endif
//    lorads_rk_mat_dense *shell;
//    LORADS_INIT(shell, lorads_rk_mat_dense, 1);
    /* Compute initial residual */
//    linSysProduct(ACone, weight, noUpdateVar, shell, updateVar->matElem, r);
    cg->Mvec(cg->MMat, x, r);
    // r = b - r
    axpy(&nr, &minusOne, b, &incx, r, &incx);
    scal(&nr, &minusOne, r, &incx);


    /* Compute initial residual norm */
    resiNorm = nrm2(&nr, r, &incx);
    if ( resiNorm / bNorm < cg_tol) {
        cg->solStatus = CG_ITER_STATUS_OK;
        goto exit_cleanup;
    }

    double zero = 0.0;
    // p = r;
    // LORADS_MEMCPY(p, r, double, nr);
    scal(&nr, &zero, p, &incx);
    axpy(&nr, &one, r, &incx, p, &incx);
    // q = r;
    // LORADS_MEMCPY(q, r, double, nr);
    scal(&nr, &zero, q, &incx);
    axpy(&nr, &one, r, &incx, q, &incx);
    // Initialize qTr
    qTr = dot(&nr, q, &incx, r, &incx);
    cg->iter = 0;
    for (lorads_int k = 0; k < cg_maxIter; ++k){
        cg->iter += 1;
#ifdef LORADS_CG_DEBUG
        lorads_printf("CGIter %d, normResi: %f\n", k, resiNorm);
#endif
//        linSysProduct(ACone, weight, noUpdateVar, shell, p, Q);
        cg->Mvec(cg->MMat, p, Q);
        qTr = dot(&nr, q, &incx, r, &incx);
        pTQ = dot(&nr, p, &incx, Q, &incx);
        alpha = qTr / pTQ;
        // x = x + alpha * p;
        axpy(&nr, &alpha, p, &incx, x, &incx);
        // r = r - alpha * Q;
        double negAlpha = -alpha;
        axpy(&nr, &negAlpha, Q, &incx, r, &incx);
        resiNorm = nrm2(&nr, r, &incx);
        cg->resiNorm = resiNorm;
        if ( resiNorm / bNorm < cg_tol ) {
            cg->solStatus = CG_ITER_STATUS_OK;
            goto exit_cleanup;
        }
        if (k%nRestartFreq == 0){
            // r = b - a * x;
            // linSysProduct(ACone, weight, noUpdateVar, shell, updateVar->matElem, r);
            cg->Mvec(cg->MMat, x, r);
            axpy(&nr, &minusOne, b, &incx, r, &incx);
            scal(&nr, &minusOne, r, &incx);
            // p = r;
            // LORADS_MEMCPY(p, r, double, nr);
            scal(&nr, &zero, p, &incx);
            axpy(&nr, &one, r, &incx, p, &incx);
            // q = r;
            // LORADS_MEMCPY(q, r, double, nr);
            scal(&nr, &zero, q, &incx);
            axpy(&nr, &one, r, &incx, q, &incx);
            // Initialize qTr
            qTr = dot(&nr, q, &incx, r, &incx);
        }
        // qNew = r
        // LORADS_MEMCPY(qNew, r, double, nr);
        scal(&nr, &zero, qNew, &incx);
        axpy(&nr, &one, r, &incx, qNew, &incx);

        qTrNew = dot(&nr, qNew, &incx, r, &incx);
        beta = qTrNew / qTr;
        // p = rNew + beta * p;
        scal(&nr, &beta, p, &incx);
        axpy(&nr, &one, r, &incx, p, &incx);

        // update qTr
        qTr = qTrNew;
        // update q
        // LORADS_MEMCPY(q, qNew, double, nr);
        scal(&nr, &zero, q, &incx);
        axpy(&nr, &one, qNew, &incx, q, &incx);

        if ( resiNorm != resiNorm ) {
            cg->solStatus = CG_ITER_STATUS_NUMERICAL;
            LORADS_ERROR_TRACE;
        }
    }

    exit_cleanup:
    /* Collect solution time statistics */
    cg->cgDuration = LUtilGetTimeStamp() - cgStartTime;
    cg->solveTime += cg->cgDuration;
}

/**
 * @brief Clear and free resources used by a conjugate gradient solver
 * @param pCGSolver Pointer to the solver to clear
 * @details Frees all allocated memory and resets solver state.
 */
extern void CGSolverClear(void *pCGSolver){
    if ( !pCGSolver ) {
        return;
    }
    lorads_cg_linsys *cg = (lorads_cg_linsys *) pCGSolver;
    LORADS_FREE(cg->rIter);
    LORADS_FREE(cg->rIterNew);
    LORADS_FREE(cg->pIter);
    LORADS_FREE(cg->qIter);
    LORADS_FREE(cg->qIterNew);
    LORADS_FREE(cg->QIter);
//    LORADS_FREE(cg->JacobiPrecond);
    LORADS_FREE(cg);
}

/**
 * @brief Destroy all conjugate gradient solvers in a LORADS solver
 * @param ASolver Pointer to the LORADS solver
 * @details Frees all conjugate gradient solvers and their associated resources
 * for all cones in the solver.
 */
extern void LORADSCGDestroy(lorads_solver *ASolver)
{
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone)
    {
        CGSolverClear(ASolver->CGLinsys[iCone]);
    }
    LORADS_FREE(ASolver->CGLinsys);
}