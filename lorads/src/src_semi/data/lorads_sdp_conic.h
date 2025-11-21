/**
 * @file lorads_sdp_conic.h
 * @brief LORADS SDP Conic Interface
 * @details This header file defines the interface for handling Semidefinite Programming (SDP)
 * conic optimization problems in LORADS. It includes functions for:
 * - Setting up and managing SDP cones
 * - Processing and presolving cone data
 * - Handling dense and sparse matrix operations
 * - Computing matrix norms and inner products
 * - Managing ARPACK eigenvalue computations
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_SDP_CONIC
#define LORADS_SDP_CONIC


#include "def_lorads_sdp_conic.h"

/**
 * @brief Set up a cone in the LORADS solver
 * @param ASolver Pointer to the LORADS solver
 * @param iCone Index of the cone
 * @param userCone User-defined cone data
 */
extern void LORADSSetCone(lorads_solver *ASolver, lorads_int iCone, void *userCone);

/**
 * @brief Process data for a cone
 * @param ACone Pointer to the SDP cone structure
 */
extern void AConeProcData(lorads_sdp_cone *ACone );

/**
 * @brief Presolve data for a cone
 * @param ACone Pointer to the SDP cone structure
 * @param Dim Dimension of the cone
 */
extern void AConePresolveData( lorads_sdp_cone *ACone, lorads_int Dim);

/**
 * @brief Sum up SDP data across all cones
 * @param ASolver Pointer to the LORADS solver
 */
extern void LORADSSumSDPData(lorads_solver *ASolver);

/**
 * @brief Destroy auxiliary sparse matrix data
 * @param pA Pointer to pointer to the auxiliary matrix
 */
extern void destroyForAuxiSparse(void **pA);

/**
 * @brief Detect sparsity pattern in dense cone data
 * @param sdp_coeff_w_sum_pointer Pointer to the SDP coefficient matrix
 */
extern void AConeDenseDetectSparsity(sdp_coeff **sdp_coeff_w_sum_pointer);

/**
 * @brief Detect sparsity pattern in the sum of SDP matrices
 * @param ASolver Pointer to the LORADS solver
 */
extern void LORADSDetectSparsityOfSumSDP(lorads_solver *ASolver);

/**
 * @brief Destroy presolve data for a cone
 * @param ACone Pointer to the SDP cone structure
 */
extern void AConeDestroyPresolveData(lorads_sdp_cone *ACone);

/**
 * @brief Clear a dense cone implementation
 * @param cone Pointer to the cone structure
 */
extern void sdpDenseConeClearImpl( void *cone );

/**
 * @brief Destroy a dense cone implementation
 * @param pCone Pointer to pointer to the cone structure
 */
extern void sdpDenseConeDestroyImpl( void **pCone );

/**
 * @brief Detect features of a dense cone
 * @param cone Pointer to the cone structure
 * @param rowRHS Right-hand side values
 * @param coneIntFeatures Array to store integer features
 * @param coneDblFeatures Array to store double features
 */
extern void sdpDenseConeFeatureDetectImpl( void *cone, double *rowRHS, lorads_int coneIntFeatures[20], double coneDblFeatures[20] );

/**
 * @brief View information about a dense cone
 * @param cone Pointer to the cone structure
 */
extern void sdpDenseConeViewImpl( void *cone );

/**
 * @brief Compute AUV product for a dense cone
 * @param coneIn Pointer to the cone structure
 * @param U First matrix
 * @param V Second matrix
 * @param constrVal Constraint values
 * @param UVt Product matrix
 */
extern void sdpDenseConeAUVImpl( void *coneIn, lorads_sdp_dense *U, lorads_sdp_dense *V, double *constrVal, sdp_coeff *UVt);

/**
 * @brief Compute objective AUV product for a dense cone
 * @param coneIn Pointer to the cone structure
 * @param U First matrix
 * @param V Second matrix
 * @param pObj Objective value
 * @param UVt Product matrix
 */
extern void sdpDenseObjAUVImpl( void *coneIn, lorads_sdp_dense *U, lorads_sdp_dense *V, double *pObj, sdp_coeff *UVt);

/**
 * @brief Compute 1-norm of objective for a dense cone
 * @param cone Pointer to the cone structure
 * @param objConeNrm1 Pointer to store the norm value
 */
extern void sdpDenseConeObjNrm1(void *cone, double *objConeNrm1);

/**
 * @brief Compute squared 2-norm of objective for a dense cone
 * @param coneIn Pointer to the cone structure
 * @param objConeNrm2Square Pointer to store the norm value
 */
extern void sdpDenseConeObjNrm2Square(void *coneIn, double *objConeNrm2Square);

/**
 * @brief Compute infinity norm of objective for a dense cone
 * @param coneIn Pointer to the cone structure
 * @param objConeNrmInf Pointer to store the norm value
 */
extern void sdpDenseConeObjNrmInf(void *coneIn, double *objConeNrmInf);

/**
 * @brief Add objective coefficients to a dense cone
 * @param cone Pointer to the cone structure
 * @param w_sum Sum of weights
 */
extern void sdpDenseConeAddObjCoeff(void *cone, sdp_coeff *w_sum);

/**
 * @brief Add random objective coefficients to a dense cone
 * @param cone Pointer to the cone structure
 * @param w_sum Sum of weights
 */
extern void sdpDenseConeAddObjCoeffRand(void *cone, sdp_coeff *w_sum);

/**
 * @brief Scale data in a dense cone
 * @param coneIn Pointer to the cone structure
 * @param scaleFactorSDPData Scaling factor
 */
extern void sdpDenseConeDataScale(void *coneIn, double scaleFactorSDPData);

/**
 * @brief Compute non-zero statistics for a dense cone
 * @param coneIn Pointer to the cone structure
 * @param stat Pointer to store statistics
 */
extern void sdpDenseConeNnzStat(void *coneIn, lorads_int *stat);

/**
 * @brief Compute non-zero statistics for coefficients in a dense cone
 * @param coneIn Pointer to the cone structure
 * @param stat Pointer to store coefficient statistics
 * @param nnzStat Pointer to store non-zero statistics
 * @param eleStat Pointer to store element statistics
 */
extern void sdpDenseConeNnzStatCoeff(void *coneIn, double *stat, lorads_int *nnzStat, lorads_int *eleStat);

/* Declaration of ARPACK functions */
/**
 * @brief ARPACK function for computing eigenvalues
 * @param ido Reverse communication flag
 * @param bmat Matrix type
 * @param n Dimension of the problem
 * @param which Which eigenvalues to compute
 * @param nev Number of eigenvalues to compute
 * @param tol Convergence tolerance
 * @param resid Residual vector
 * @param ncv Number of Lanczos vectors
 * @param v Lanczos basis vectors
 * @param ldv Leading dimension of v
 * @param iparam Integer parameters
 * @param ipntr Integer pointers
 * @param workd Double precision work array
 * @param workl Double precision work array
 * @param lworkl Length of workl
 * @param info Information flag
 */
extern void dsaupd_(int *ido, char *bmat,  lorads_int *n, char *which,  int *nev, double *tol, double *resid,
                     int *ncv, double *v,  lorads_int *ldv,  int *iparam,  int *ipntr, double *workd,
                    double *workl,  int *lworkl,  int *info);

/**
 * @brief ARPACK function for computing eigenvectors
 * @param rvec Whether to compute eigenvectors
 * @param HowMny How many eigenvectors to compute
 * @param select Selection array
 * @param d Eigenvalues
 * @param z Eigenvectors
 * @param ldz Leading dimension of z
 * @param sigma Shift value
 * @param bmat Matrix type
 * @param n Dimension of the problem
 * @param which Which eigenvalues to compute
 * @param nev Number of eigenvalues to compute
 * @param tol Convergence tolerance
 * @param resid Residual vector
 * @param ncv Number of Lanczos vectors
 * @param v Lanczos basis vectors
 * @param ldv Leading dimension of v
 * @param iparam Integer parameters
 * @param ipntr Integer pointers
 * @param workd Double precision work array
 * @param workl Double precision work array
 * @param lworkl Length of workl
 * @param info Information flag
 */
extern void dseupd_(int *rvec, char *HowMny,  int *select, double *d, double *z,  lorads_int *ldz, double *sigma,
                    char *bmat,  lorads_int *n, char *which,  int *nev, double *tol, double *resid,
                     int *ncv, double *v,  lorads_int *ldv,  int *iparam,  int *ipntr, double *workd,
                    double *workl,  int *lworkl,  int *info);

/**
 * @brief Check for dual infeasibility
 * @param matvec Matrix-vector multiplication function
 * @param M Matrix data
 * @param res Result vector
 * @param n Dimension of the problem
 * @return 1 if dual infeasible, 0 otherwise
 */
int dual_infeasible(void (*matvec) (void *M, double *x, double *y, lorads_int n), void *M, double *res, lorads_int n);

#endif