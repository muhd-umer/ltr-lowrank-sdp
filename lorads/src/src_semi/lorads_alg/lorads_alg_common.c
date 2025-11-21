/**
 * @file lorads_alg_common.c
 * @brief LORADS Common Algorithm Functions
 * @details This file implements common functions used across different LORADS algorithms,
 * including constraint handling, variable updates, and convergence monitoring.
 * 
 * @author LORADS Team
 * @date 2024
 */

#include <math.h>
#include "lorads_solver.h"
#include "lorads_vec_opts.h"
#include "lorads_dense_opts.h"
#include "lorads_admm.h"

/**
 * @brief Get value pointer from constraint vector
 * @param constrVal Constraint vector
 * @param type Vector type (dense or sparse)
 * @param res Pointer to store result
 * @details Extracts the value pointer from either dense or sparse constraint vector
 */
extern void valRes(void *constrVal, vecType type, double **res)
{
    if (type == LORADS_DENSE_VEC){
        dense_vec *dense = (dense_vec *)constrVal;
        *res = dense->val;
    }
    else if (type == LORADS_SPARSE_VEC){
        sparse_vec *sparse = (sparse_vec *)constrVal;
        *res = sparse->val;
    }
}

/**
 * @brief Compute UV^T matrix product
 * @param UVt_w_sum Output coefficient matrix
 * @param U First input matrix
 * @param V Second input matrix
 * @details Computes (UV^T + VU^T)/2 for symmetric matrices, handling both sparse and dense formats
 */
extern void LORADSUVt(sdp_coeff *UVt_w_sum, lorads_sdp_dense *U, lorads_sdp_dense *V){
    /* sdp_coeff_w_sum is the sum of all sdp data in the cone,
     It has two data type
     - sparse: means most all sdp data coeff is sparse
     - dense:  there exist a sdp data coeff is dense

     calculate (UVt + VUt) / 2
     */
    if (UVt_w_sum->dataType == SDP_COEFF_SPARSE){
        sdp_coeff_sparse *sparse = (sdp_coeff_sparse *)UVt_w_sum->dataMat;
        // method1:
        lorads_int row = 0;
        lorads_int col = 0;
        lorads_int incx = 1;
        lorads_int incy = 1;
        for (lorads_int i = 0; i < sparse->nTriMatElem; ++i)
        {
            row = sparse->triMatRow[i];
            col = sparse->triMatCol[i];
            incx = U->nRows;
            incy = V->nRows;
            if (row != col){
                sparse->triMatElem[i] = 0.5 * dot(&(U->rank), &U->matElem[row], &incx, &V->matElem[col], &incy);
                sparse->triMatElem[i] += 0.5 *  dot(&(U->rank), &U->matElem[col], &incx, &V->matElem[row], &incy);
            }else{
                sparse->triMatElem[i] = dot(&(U->rank), &U->matElem[row], &incx, &V->matElem[col], &incy);
            }
        }
    }
    else if (UVt_w_sum->dataType == SDP_COEFF_DENSE){
        sdp_coeff_dense *dense = (sdp_coeff_dense *)UVt_w_sum->dataMat;
//        double *fullDataMat;
//        LORADS_INIT(fullDataMat, double, dense->nSDPCol * dense->nSDPCol);
        LORADS_ZERO(dense->fullMat, double, dense->nSDPCol * dense->nSDPCol);
//        printf("rank: %lld\n", U->rank);
        // alpha = 0.5, beta = 0.0;
        fds_syr2k(ACharConstantUploLow, 'N', U->nRows, U->rank, 0.5, U->matElem, V->matElem, 0.0, dense->fullMat);
        lorads_int idx = 0;
        lorads_int row = 0;
        for (lorads_int col = 0; col < dense->nSDPCol; ++col)
        {
            LORADS_MEMCPY(&dense->dsMatElem[idx], &dense->fullMat[dense->nSDPCol * col + row], double, dense->nSDPCol - col);
            row++;
            idx += (dense->nSDPCol - col);
        }
//        LORADS_FREE(fullDataMat);
    }
}

/**
 * @brief Initialize constraint values for a single cone
 * @param ACone SDP cone
 * @param U First matrix
 * @param V Second matrix
 * @param constrVal Output constraint values
 * @details Computes and stores constraint values for a single SDP cone
 */
extern void LORADSInitConstrVal(lorads_sdp_cone *ACone, lorads_sdp_dense *U, lorads_sdp_dense *V, double *constrVal)
{
    LORADSUVt(ACone->sdp_coeff_w_sum, U, V);
    // Calculate Constraint Value for one cone
    ACone->coneAUV(ACone->coneData, U, V, constrVal, ACone->sdp_coeff_w_sum);
}

/**
 * @brief Initialize constraint values for all cones
 * @param ASolver LORADS solver instance
 * @param uLpDummy LP variable (unused)
 * @param vLpDummy LP variable (unused)
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @details Computes and stores constraint values for all SDP cones
 */
extern void LORADSInitConstrValAll(lorads_solver *ASolver, lorads_lp_dense *uLpDummy, lorads_lp_dense *vLpDummy, lorads_sdp_dense **U, lorads_sdp_dense **V){
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone){
        double *constrVal;
        valRes(ASolver->var->constrVal[iCone]->data, ASolver->var->constrVal[iCone]->type, &constrVal);
        LORADSInitConstrVal(ASolver->SDPCones[iCone], U[iCone], V[iCone], constrVal);
    }
}

/**
 * @brief Initialize constraint values for all cones including LP
 * @param ASolver LORADS solver instance
 * @param uLp LP variable
 * @param vLp LP variable
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @details Computes and stores constraint values for all SDP cones and LP variables
 */
extern void LORADSInitConstrValAllLP(lorads_solver *ASolver, lorads_lp_dense *uLp, lorads_lp_dense *vLp, lorads_sdp_dense **U, lorads_sdp_dense **V)
{
    lorads_lp_cone *lpCone = ASolver->lpCone;
    LORADSInitConstrValAll(ASolver, uLp, vLp, U, V);
    for (lorads_int iCol = 0; iCol < lpCone->nCol; ++iCol){
        double *constrVal;
        valRes(ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValLP[iCol]->type, &constrVal);
        lpCone->coneAUV(lpCone->coneData, uLp, vLp, constrVal, iCol);
    }
}

/**
 * @brief Initialize constraint values and objective value for a single cone
 * @param ACone SDP cone
 * @param U First matrix
 * @param V Second matrix
 * @param constrVal Output constraint values
 * @param UVobjVal Output objective value
 * @details Computes both constraint values and objective value for a single SDP cone
 */
extern void LORADSInitConstrValObjVal(lorads_sdp_cone *ACone, lorads_sdp_dense *U, lorads_sdp_dense *V, double *constrVal, double *UVobjVal){
    LORADSUVt(ACone->sdp_obj_sum, U, V);
    // Calculate Constraint Value for one cone
    ACone->objAUV(ACone->coneData, U, V, UVobjVal, ACone->sdp_obj_sum);
//    printf("objVal: %f\n", UVobjVal[0]);
    ACone->coneAUV(ACone->coneData, U, V, constrVal, ACone->sdp_obj_sum);
}

/**
 * @brief Compute objective and constraint values for all cones
 * @param ASolver LORADS solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param UVobjVal Output objective value
 * @details Computes objective and constraint values for all SDP cones
 */
extern void LORADSObjConstrValAll(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, double *UVobjVal){
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone){
        ASolver->var->constrVal[iCone]->zero(ASolver->var->constrVal[iCone]->data);
        double *constrVal;
        valRes(ASolver->var->constrVal[iCone]->data, ASolver->var->constrVal[iCone]->type, &constrVal);
        LORADSInitConstrValObjVal(ASolver->SDPCones[iCone], U[iCone], V[iCone], constrVal, UVobjVal);
    }
}

/**
 * @brief Compute objective and constraint values for all cones including LP
 * @param ASolver LORADS solver instance
 * @param uLp LP variable
 * @param vLp LP variable
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param UVobjVal Output objective value
 * @details Computes objective and constraint values for all SDP cones and LP variables
 */
extern void LORADSObjConstrValAllLP(lorads_solver *ASolver, lorads_lp_dense *uLp, lorads_lp_dense *vLp, lorads_sdp_dense **U, lorads_sdp_dense **V, double *UVobjVal)
{
    ASolver->lpCone->objAUV(ASolver->lpCone->coneData, uLp, vLp, UVobjVal);
    for (lorads_int iCol = 0; iCol < uLp->nCols; ++iCol)
    {
        double *constrVal;
        valRes(ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValLP[iCol]->type, &constrVal);
        ASolver->lpCone->coneAUV(ASolver->lpCone->coneData, uLp, vLp, constrVal, iCol);
    }
    LORADSObjConstrValAll(ASolver, U, V, UVobjVal);
}

/**
 * @brief Update constraint values for a single cone
 * @param ACone SDP cone
 * @param U First matrix
 * @param V Second matrix
 * @param constrVal Constraint vector to update
 * @details Updates constraint values for a single SDP cone
 */
extern void LORADSUpdateConstrVal(lorads_sdp_cone *ACone, lorads_sdp_dense *U, lorads_sdp_dense *V, lorads_vec *constrVal)
{
    // constrVal = Acal(UVt)
    double *constrValRes;
    valRes(constrVal->data, constrVal->type, &constrValRes);
    LORADSInitConstrVal(ACone, U, V, constrValRes);
}

/**
 * @brief Initialize sum of constraint values
 * @param ASolver LORADS solver instance
 * @details Computes sum of constraint values across all SDP cones
 */
extern void LORADSInitConstrValSum(lorads_solver *ASolver)
{
    LORADS_ZERO(ASolver->var->constrValSum, double, ASolver->nRows);
    double alpha = 1.0;
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone)
    {
        ASolver->var->constrVal[iCone]->add(&alpha, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);
    }
}

/**
 * @brief Initialize sum of constraint values including LP
 * @param ASolver LORADS solver instance
 * @details Computes sum of constraint values across all SDP cones and LP variables
 */
extern void LORADSInitConstrValSumLP(lorads_solver *ASolver)
{
    LORADS_ZERO(ASolver->var->constrValSum, double, ASolver->nRows);
    double alpha = 1.0;
    for (lorads_int iCol = 0; iCol < ASolver->nLpCols; ++iCol)
    {
        ASolver->var->constrValLP[iCol]->add(&alpha, ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValSum);
    }

    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone)
    {
        ASolver->var->constrVal[iCone]->add(&alpha, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);
    }
}

/**
 * @brief Copy R matrices to V matrices
 * @param rLpDummy LP variable (unused)
 * @param vlpDummy LP variable (unused)
 * @param R Array of R matrices
 * @param V Array of V matrices
 * @param nCones Number of cones
 * @details Copies contents of R matrices to V matrices
 */
extern void copyRtoV(lorads_lp_dense *rLpDummy, lorads_lp_dense *vlpDummy, lorads_sdp_dense **R, lorads_sdp_dense **V, lorads_int nCones)
{
    lorads_int n = 0;
    for (lorads_int iCone = 0; iCone < nCones; ++iCone)
    {
        n = R[iCone]->rank * R[iCone]->nRows;
        LORADS_MEMCPY(V[iCone]->matElem, R[iCone]->matElem, double, n);
    }
}

/**
 * @brief Copy R matrices to V matrices including LP
 * @param rLp LP variable
 * @param vlp LP variable
 * @param R Array of R matrices
 * @param V Array of V matrices
 * @param nCones Number of cones
 * @details Copies contents of R matrices to V matrices and LP variables
 */
extern void copyRtoVLP(lorads_lp_dense *rLp, lorads_lp_dense *vlp, lorads_sdp_dense **R, lorads_sdp_dense **V, lorads_int nCones)
{
    lorads_int n = 0;
    LORADS_MEMCPY(vlp->matElem, rLp->matElem, double, vlp->nCols);
    for (lorads_int iCone = 0; iCone < nCones; ++iCone)
    {
        n = R[iCone]->rank * R[iCone]->nRows;
        LORADS_MEMCPY(V[iCone]->matElem, R[iCone]->matElem, double, n);
    }
}

/**
 * @brief Update SDP variables
 * @param ASolver LORADS solver instance
 * @param rho Penalty parameter
 * @param CG_tol CG tolerance
 * @param CG_maxIter Maximum CG iterations
 * @details Updates SDP variables using conjugate gradient method
 */
extern void LORADSUpdateSDPVar(lorads_solver *ASolver, double rho, double CG_tol, lorads_int CG_maxIter){
    double minusOne = -1.0;
    double one = 1.0;
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone)
    {
        // update U
#ifdef DUAL_U_V
        LORADSUpdateSDPVarOne_positive_S(ASolver, ASolver->var->U[iCone], ASolver->var->V[iCone], ASolver->var->S[iCone], iCone, rho, CG_tol, CG_maxIter);
#else
        LORADSUpdateSDPVarOne(ASolver, ASolver->var->U[iCone], ASolver->var->V[iCone], iCone, rho, CG_tol, CG_maxIter);
#endif
        // ASolver->constrValSum - ASolver->constrVal[iCone]
        ASolver->var->constrVal[iCone]->add(&minusOne, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);
        // update ASolver->constrVal[iCone]
        LORADSUpdateConstrVal(ASolver->SDPCones[iCone], ASolver->var->U[iCone], ASolver->var->V[iCone], ASolver->var->constrVal[iCone]);
        // update ASolver->constrValSum
        ASolver->var->constrVal[iCone]->add(&one, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);

        // update V
#ifdef DUAL_U_V
        LORADSUpdateSDPVarOne_negative_S(ASolver, ASolver->var->V[iCone], ASolver->var->U[iCone], ASolver->var->S[iCone], iCone, rho, CG_tol, CG_maxIter);
#else
        LORADSUpdateSDPVarOne(ASolver, ASolver->var->V[iCone], ASolver->var->U[iCone], iCone, rho, CG_tol, CG_maxIter);
#endif
        ASolver->var->constrVal[iCone]->add(&minusOne, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);
        LORADSUpdateConstrVal(ASolver->SDPCones[iCone], ASolver->var->U[iCone], ASolver->var->V[iCone], ASolver->var->constrVal[iCone]);
        ASolver->var->constrVal[iCone]->add(&one, ASolver->var->constrVal[iCone]->data, ASolver->var->constrValSum);
    }
}

/**
 * @brief Update LP constraint values
 * @param lp_cone LP cone
 * @param uLp LP variable
 * @param vLp LP variable
 * @param constrVal Constraint vector to update
 * @param iCol Column index
 * @details Updates constraint values for LP variables
 */
extern void LORADSUpdateConstrValLP(lorads_lp_cone *lp_cone, lorads_lp_dense *uLp, lorads_lp_dense *vLp, lorads_vec *constrVal, lorads_int iCol)
{
    double *constrValRes;
    valRes(constrVal->data, constrVal->type, &constrValRes);
    lp_cone->coneAUV(lp_cone->coneData, uLp, vLp, constrValRes, iCol);
}

/**
 * @brief Update SDP and LP variables
 * @param ASolver LORADS solver instance
 * @param rho Penalty parameter
 * @param CG_tol CG tolerance
 * @param CG_maxIter Maximum CG iterations
 * @details Updates both SDP and LP variables using conjugate gradient method
 */
extern void LORADSUpdateSDPLPVar(lorads_solver *ASolver, double rho, double CG_tol, lorads_int CG_maxIter){
    double minusOne = -1.0;
    double one = 1.0;
    LORADSUpdateSDPVar(ASolver, rho, CG_tol, CG_maxIter);
    for (lorads_int iCol = 0; iCol < ASolver->nLpCols; ++iCol){
#ifdef DUAL_U_V
        LORADSUpdateLPVarOne_positive_S(ASolver, &ASolver->var->uLp->matElem[iCol], &ASolver->var->vLp->matElem[iCol],  iCol, rho, ASolver->var->sLp->matElem);
#else
        LORADSUpdateLPVarOne(ASolver, &ASolver->var->uLp->matElem[iCol], &ASolver->var->vLp->matElem[iCol],  iCol, rho);
#endif
        ASolver->var->constrValLP[iCol]->add(&minusOne, ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValSum);
        LORADSUpdateConstrValLP(ASolver->lpCone, ASolver->var->uLp, ASolver->var->vLp, ASolver->var->constrValLP[iCol], iCol);
        ASolver->var->constrValLP[iCol]->add(&one, ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValSum);

#ifdef DUAL_U_V
        LORADSUpdateLPVarOne_negative_S(ASolver, &ASolver->var->vLp->matElem[iCol], &ASolver->var->uLp->matElem[iCol],   iCol, rho, ASolver->var->sLp->matElem);
#else
        LORADSUpdateLPVarOne(ASolver, &ASolver->var->vLp->matElem[iCol], &ASolver->var->uLp->matElem[iCol],   iCol, rho);
#endif
        ASolver->var->constrValLP[iCol]->add(&minusOne, ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValSum);
        LORADSUpdateConstrValLP(ASolver->lpCone, ASolver->var->uLp, ASolver->var->vLp, ASolver->var->constrValLP[iCol], iCol);
        ASolver->var->constrValLP[iCol]->add(&one, ASolver->var->constrValLP[iCol]->data, ASolver->var->constrValSum);
    }
}

/**
 * @brief Compute primal infeasibility
 * @param ASolver LORADS solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP variable
 * @param r2 LP variable
 * @details Computes primal infeasibility measure
 */
extern void primalInfeasibility(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2){
    LORADSInitConstrValAll(ASolver, r, r2, R, R2);
    LORADSInitConstrValSum(ASolver);
    double one = 1.0;
    double minusOne = -1.0;
    axpbyAddition(&ASolver->nRows, &(one), ASolver->rowRHS, &(minusOne), ASolver->var->constrValSum, ASolver->constrVio);
    lorads_int incx = 1;
    ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] = nrm2(&ASolver->nRows, ASolver->constrVio, &incx) / (1 + ASolver->bRHSNrm1);
}

/**
 * @brief Compute primal infeasibility including LP
 * @param ASolver LORADS solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP variable
 * @param r2 LP variable
 * @details Computes primal infeasibility measure including LP variables
 */
extern void primalInfeasibilityLP(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2){
    LORADSInitConstrValAllLP(ASolver, r, r2, R, R2);
    LORADSInitConstrValSumLP(ASolver);
    double one = 1.0;
    double minusOne = -1.0;
    axpbyAddition(&ASolver->nRows, &(one), ASolver->rowRHS, &(minusOne), ASolver->var->constrValSum, ASolver->constrVio);
    lorads_int incx = 1;
    ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] = nrm2(&ASolver->nRows, ASolver->constrVio, &incx) / (1 + ASolver->bRHSNrm1);
}

/**
 * @brief Update DIMACS error for ALM
 * @param ASolver LORADS solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP variable
 * @param r2 LP variable
 * @details Updates DIMACS error measures for ALM algorithm
 */
extern void LORADSUpdateDimacsErrorALM(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2){
    primalInfeasibility(ASolver, R, R2, r, r2);
    double gap = (ASolver->pObjVal - ASolver->dObjVal);
    ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP] = LORADS_ABS(gap) / (1 + LORADS_ABS(ASolver->pObjVal) + LORADS_ABS(ASolver->dObjVal));
}

/**
 * @brief Update DIMACS error for ALM including LP
 * @param ASolver LORADS solver instance
 * @param R Array of R matrices
 * @param R2 Array of R2 matrices
 * @param r LP variable
 * @param r2 LP variable
 * @details Updates DIMACS error measures for ALM algorithm including LP variables
 */
extern void LORADSUpdateDimacsErrorALMLP(lorads_solver *ASolver, lorads_sdp_dense **R, lorads_sdp_dense **R2, lorads_lp_dense *r, lorads_lp_dense *r2){
    primalInfeasibilityLP(ASolver, R, R2, r, r2);
    double gap = (ASolver->pObjVal - ASolver->dObjVal);
    ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP] = LORADS_ABS(gap) / (1 + LORADS_ABS(ASolver->pObjVal) + LORADS_ABS(ASolver->dObjVal));
}

/**
 * @brief Update DIMACS error for ADMM
 * @param ASolver LORADS solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param u LP variable
 * @param v LP variable
 * @details Updates DIMACS error measures for ADMM algorithm
 */
extern void LORADSUpdateDimacsErrorADMM(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, lorads_lp_dense *u, lorads_lp_dense *v){
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone){
        averageUV(U[iCone], V[iCone], ASolver->var->R[iCone]);
    }
    averageUVLP(u, v, ASolver->var->rLp);
    primalInfeasibility(ASolver, ASolver->var->R, ASolver->var->R, ASolver->var->rLp, ASolver->var->rLp);
    double gap = (ASolver->pObjVal - ASolver->dObjVal);
    ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP] = LORADS_ABS(gap) / (1 + LORADS_ABS(ASolver->pObjVal) + LORADS_ABS(ASolver->dObjVal));
}

/**
 * @brief Update DIMACS error for ADMM including LP
 * @param ASolver LORADS solver instance
 * @param U Array of U matrices
 * @param V Array of V matrices
 * @param u LP variable
 * @param v LP variable
 * @details Updates DIMACS error measures for ADMM algorithm including LP variables
 */
extern void LORADSUpdateDimacsErrorADMMLP(lorads_solver *ASolver, lorads_sdp_dense **U, lorads_sdp_dense **V, lorads_lp_dense *u, lorads_lp_dense *v){
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone){
        averageUV(U[iCone], V[iCone], ASolver->var->R[iCone]);
    }
    averageUVLP(u, v, ASolver->var->rLp);
    primalInfeasibilityLP(ASolver, ASolver->var->R, ASolver->var->R, ASolver->var->rLp, ASolver->var->rLp);
    double gap = (ASolver->pObjVal - ASolver->dObjVal);
    ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP] = LORADS_ABS(gap) / (1 + LORADS_ABS(ASolver->pObjVal) + LORADS_ABS(ASolver->dObjVal));
}

/**
 * @brief Compute infinity norm of objective
 * @param ASolver LORADS solver instance
 * @details Computes infinity norm of objective function coefficients
 */
extern void LORADSNrmInfObj(lorads_solver *ASolver)
{
    ASolver->cObjNrmInf = 0.0;
    if (ASolver->nLpCols > 0)
    {
        lorads_lp_cone *lpCone = ASolver->lpCone;
        lpCone->coneObjNrmInf(lpCone->coneData, &ASolver->cObjNrmInf, ASolver->nLpCols);
    }
    for (lorads_int iCone = 0; iCone < ASolver->nCones; ++iCone)
    {
        double temp = 0.0;
        lorads_sdp_cone *ACone = ASolver->SDPCones[iCone];
        ACone->coneObjNrmInf(ACone->coneData, &temp);
        ASolver->cObjNrmInf = LORADS_MAX(ASolver->cObjNrmInf, temp);
    }
}

/**
 * @brief Update dual variables
 * @param ASolver LORADS solver instance
 * @param rho Penalty parameter
 * @details Updates dual variables using ADMM update rule
 */
extern void LORADSUpdateDualVar(lorads_solver *ASolver, double rho)
{
    double *dualVar = ASolver->var->dualVar;
    double minusRho = -rho;
    double *b = ASolver->rowRHS;
    double *constrSum = ASolver->var->constrValSum;
    lorads_int n = ASolver->nRows;

    // lambda = lambda + rho * b
    lorads_int incx = 1;
    axpy(&(n), &(rho), b, &(incx), dualVar, &(incx));
    // lambda = lambda - rho * constrVal
    axpy(&(n), &(minusRho), constrSum, &(incx), dualVar, &(incx));
}

/**
 * @brief Calculate dual objective value
 * @param ASolver LORADS solver instance
 * @details Computes dual objective value
 */
extern void LORADSCalDualObj(lorads_solver *ASolver)
{
    lorads_int n = ASolver->nRows;
    lorads_int one = 1;
    ASolver->dObjVal = dot(&n, ASolver->rowRHS, &one, ASolver->var->dualVar, &one);
    ASolver->dObjVal /= ASolver->scaleObjHis;
}

/**
 * @brief Check solver status
 * @param ASolver LORADS solver instance
 * @return Return code indicating solver status
 * @details Checks if solver has reached optimal solution
 */
extern lorads_int LORADSCheckSolverStatus(lorads_solver *ASolver){
    lorads_int retcode = LORADS_RETCODE_OK;
    if (ASolver->AStatus == LORADS_PRIMAL_OPTIMAL){
        retcode = LORADS_RETCODE_EXIT;
    }
    return retcode;
}