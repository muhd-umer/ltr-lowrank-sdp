/**
 * @file lorads.h
 * @brief LORADS Main Header File
 * @details This header file contains core definitions, types, and structures for the LORADS
 * (Low-Rank ADMM Splitting) solver. It includes:
 * - Type definitions and macros
 * - Return codes and status enums
 * - Solver parameters structure
 * - Build information
 * - Constants and thresholds
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_H
#define LORADS_H

/* Macro for MATLAB */
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define calloc mxCalloc
#define printf mexPrintf
#define free mxFree
#else
#endif

/** @brief Threshold for determining when to use sparse cone representation */
#define LORADS_SPARSE_CONE_THRESHOLD  (0.3)

/** @brief Enable Conjugate Gradient method */
#define USE_CG

/** @brief Default relative tolerance for oracle rank eigenvalue cutoff */
#define LORADS_ORACLE_EPS (1e-6)

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief Integer type definitions based on platform
 * @details Defines lorads_int as either 32-bit or 64-bit integer depending on platform
 */
#ifdef MAC_INT64
#define lorads_int     int64_t
#endif

#ifdef UNIX_INT64
#define lorads_int     int64_t
#endif

#ifdef INT32
#define lorads_int     int
#endif

/**
 * @brief Return codes for LORADS functions
 * @details Enumeration of possible return values from LORADS functions
 */
typedef enum {
    LORADS_RETCODE_OK,        ///< Operation completed successfully
    LORADS_RETCODE_FAILED,    ///< Operation failed
    LORADS_RETCODE_MEMORY,    ///< Memory allocation failed
    LORADS_RETCODE_EXIT,      ///< Operation exited early
    LORADS_RETCODE_RANK,      ///< Rank-related error occurred
} lorads_retcode;

/**
 * @brief Solver status codes
 * @details Enumeration of possible solver termination states
 */
typedef enum{
    LORADS_UNKNOWN,              ///< Unknown status
    LORADS_PRIMAL_DUAL_OPTIMAL,  ///< Both primal and dual optimal
    LORADS_PRIMAL_OPTIMAL,       ///< Only primal optimal
    LORADS_MAXITER,              ///< Maximum iterations reached
    LORADS_TIME_LIMIT,           ///< Time limit reached
}lorads_status;

/**
 * @brief Oracle rank computation methods
 */
typedef enum{
    LORADS_ORACLE_RANK_GRAM = 0,  ///< Use Gram matrix eigenvalues
    LORADS_ORACLE_RANK_NAIVE = 1  ///< Use full matrix eigen decomposition
} lorads_oracle_rank_method;

/** @brief Infinity value used in the solver */
#define LORADS_INFINITY          1e+30

/**
 * @brief ALM (Augmented Lagrangian Method) difficulty levels
 * @details Character constants representing different levels of problem difficulty
 */
#define EASY                   ('e')  ///< Easy problem
#define MEDIUM                 ('m')  ///< Medium difficulty
#define HARD                   ('h')  ///< Hard problem
#define SUPER                  ('s')  ///< Super hard problem

/**
 * @brief Return code constants
 * @details Integer constants for different return conditions
 */
#define RET_CODE_OK            (0)    ///< Successful operation
#define RET_CODE_TIME_OUT      (1)    ///< Time limit exceeded
#define RET_CODE_NUM_ERR       (4)    ///< Numerical error occurred
#define RET_CODE_BAD_ITER      (8)    ///< Bad iteration detected

/** @brief Maximum number of ALM sub-iterations */
extern int MAX_ALM_SUB_ITER;

/**
 * @brief Build date information
 * @details Constants defining the build date of the library
 */
#define BUILD_DATE_YEAR         (2024)  ///< Build year
#define BUILD_DATE_MONTH        (08)    ///< Build month
#define BUILD_DATE_DAY          (27)    ///< Build day

/** @brief Default termination tolerance */
#define LORADS_TERMINATION_TOL (1e-5)

#include <getopt.h>
#include <stdbool.h>

/**
 * @brief Solver parameters structure
 * @details Contains all configurable parameters for the LORADS solver
 */
typedef struct{
    char *fname;                ///< Input file name
    double initRho;             ///< Initial penalty parameter
    double rhoMax;              ///< Maximum penalty parameter
    double rhoCellingALM;       ///< ALM penalty parameter ceiling
    double rhoCellingADMM;      ///< ADMM penalty parameter ceiling
    lorads_int maxALMIter;      ///< Maximum ALM iterations
    lorads_int maxADMMIter;     ///< Maximum ADMM iterations
    double timesLogRank;        ///< Factor for rank estimation
    lorads_int fixedRank;       ///< Fixed rank override for all cones (-1 to disable)
    lorads_int rhoFreq;         ///< Frequency of penalty updates
    double rhoFactor;           ///< Penalty increase factor
    double ALMRhoFactor;        ///< ALM penalty increase factor
    double phase1Tol;           ///< Phase 1 convergence tolerance
    double phase2Tol;           ///< Phase 2 convergence tolerance
    double timeSecLimit;        ///< Time limit in seconds
    double heuristicFactor;     ///< Factor for heuristic adjustments
    lorads_int lbfgsListLength; ///< L-BFGS history length
    double endTauTol;           ///< Final tau tolerance
    double endALMSubTol;        ///< Final ALM subproblem tolerance
    bool l2Rescaling;           ///< Whether to use L2 rescaling
    lorads_int reoptLevel;      ///< Level of reoptimization
    lorads_int dyrankLevel;     ///< Level of dynamic rank adjustment
    bool highAccMode;           ///< Whether to use high accuracy mode
    lorads_oracle_rank_method oracleRankMethod; ///< Oracle rank computation method
} lorads_params;

/** @brief Disable penalty method */
#define NO_PENALTY_METHOD

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* lorads_h */
