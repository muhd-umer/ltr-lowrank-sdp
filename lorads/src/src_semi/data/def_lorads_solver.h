/**
 * @file def_lorads_solver.h
 * @brief Core solver data structures and state definitions for LORADS
 * @details This header file defines the main data structures used in the LORADS solver,
 * including variables, solver state, and function interfaces for both ALM and ADMM algorithms.
 */

#ifndef DEF_LORADS_SOLVER
#define DEF_LORADS_SOLVER

#include <stdbool.h>
#include <stdio.h>
#include <limits.h>
#include "def_lorads_lp_conic.h"
#include "def_lorads_sdp_conic.h"
#include "def_lorads_elements.h"
#include "def_lorads_cgs.h"
#include "def_lorads_lbfgs.h"
#include "lorads.h"

/**
 * @brief Logging context for dynamic trajectory capture
 */
typedef struct{
    FILE *trajectory_fp;
    FILE *log_fp;
    char problem_name[PATH_MAX];
    char input_file_path[PATH_MAX];
    char trajectory_path[PATH_MAX * 2];
    char log_path[PATH_MAX * 2];
    char json_path[PATH_MAX * 2];
    double solve_start_time;
    int naive_fallback_warned;

    /* Trajectory storage for JSON output */
    lorads_int *phase1_curr_rank;
    lorads_int *phase1_oracle_rank;
    lorads_int phase1_count;
    lorads_int phase1_capacity;

    lorads_int *phase2_curr_rank;
    lorads_int *phase2_oracle_rank;
    lorads_int phase2_count;
    lorads_int phase2_capacity;
} lorads_logging_ctx;

/**
 * @brief Structure containing all variables used in the LORADS solver
 * @details This structure holds all the variables needed for both SDP and LP cones,
 * including ADMM variables, ALM variables, and auxiliary variables for optimization.
 */
typedef struct{
    /* Variables SDPCone */
    lorads_sdp_dense **U;    // admm variable, and lbfgs descent direction D
    lorads_sdp_dense **V;    // admm variable only
#ifdef DUAL_U_V
    lorads_sdp_dense **S;    // admm dual variable
#endif
    lorads_sdp_dense **R;    // average variable for storage and ALM variable
    lorads_sdp_dense **Grad; // grad of R

    /* Variable LPCone */
    lorads_lp_dense *rLp;
    lorads_lp_dense *uLp;
    lorads_lp_dense *vLp;
#ifdef DUAL_U_V
    lorads_lp_dense *sLp;
#endif
    lorads_lp_dense *gradLp;

    /* ALM lbfgs and ADMM Variables */
    double *dualVar;

    /* Auxiliary variable */
    lorads_vec **constrVal; // constraint violation [iCone][iConstr]
    double *constrValSum;        // constraint violation [iConstr]
    double *ARDSum;              // q1 in line search of ALM
    double *ADDSum;              // q2 in line search of ALM
    double **bLinSys;            // for solving linear system, bInLinSys[iCone]
    lorads_int *rankElem;               // all rank
    double *M1temp;              // M1 for solving linear system
    double *bestDualVar;
    lorads_sdp_dense **M2temp; // M2 for solving linear system
    double *Dtemp;
    lorads_vec **constrValLP;
}lorads_variable;

/**
 * @brief Main solver structure containing all problem data and state
 * @details This structure holds all the data needed for the LORADS solver:
 * - Problem data (constraints, cones)
 * - Optimization variables
 * - Convergence monitoring
 * - Algorithm state and parameters
 */
typedef struct{
    /* User data */
    lorads_int nRows;      // constraint number
    double *rowRHS; // b of Ax = b

    /* Cones */
    lorads_int nCones; // sdp cones block number
    lorads_sdp_cone **SDPCones;
    lorads_cg_linsys **CGLinsys;

    // variable
    lorads_variable *var;

    /* Auxiliary variable LPCone (SDPCone but rank is 1, dim is 1) */
    lorads_lp_cone *lpCone;
    lorads_int nLpCols;

    /* ALM lbfgs Variables */
    lorads_int hisRecT;
    lbfgs_node *lbfgsHis; // record difference of primal variable history and gradient history, lbfgsHis[iCone]

    /* Monitor */
    lorads_int nIterCount;
    double cgTime;
    lorads_int cgIter;
    lorads_int checkSolTimes;
//    double traceSum;

    /* Convergence criterion */
    double pObjVal;
    double dObjVal;
    double pInfeas;
    double dInfeas;
    double cObjNrm1;
    double cObjNrm2;
    double cObjNrmInf;
    double bRHSNrm1;
    double bRHSNrmInf;
    double bRHSNrm2;
    double *dimacError;
    double *constrVio;

    /* Starting time */
    double dTimeBegin;

    // scale
    double cScaleFactor;
    double bScaleFactor;

    // check exit alm
    lorads_int *rank_max;
    double *sparsitySDPCoeff;
    double overallSparse;
    lorads_int nnzSDPCoeffSum;
    lorads_int SDPCoeffSum;
    lorads_status AStatus;

    lorads_oracle_rank_method oracleMethod;
    double oracleEpsilon;
    lorads_logging_ctx log_ctx;
    double scaleObjHis;
} lorads_solver;

/**
 * @brief Structure containing function pointers for ALM and ADMM operations
 * @details This structure provides a unified interface for various optimization operations:
 * - Constraint value initialization and updates
 * - Gradient calculations
 * - L-BFGS direction computation
 * - Variable updates
 * - Objective function calculations
 * - Convergence monitoring
 */
typedef struct {
    void (*InitConstrValAll)    (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **);
    void (*InitConstrValSum)    (lorads_solver *);
    void (*ALMCalGrad)          (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **, double *, double);
    void (*LBFGSDirection)      (lorads_params *, lorads_solver *, lbfgs_node *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **, lorads_int);
    void (*LBFGSDirUseGrad)     (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **);
    void (*copyRtoV)            (lorads_lp_dense *rlp, lorads_lp_dense *vlp, lorads_sdp_dense **, lorads_sdp_dense **, lorads_int);
    void (*ALMCalq12p12)        (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **, double *, double *, double *);
    void (*setAsNegGrad)        (lorads_solver *, lorads_lp_dense *, lorads_sdp_dense **);
    void (*ALMupdateVar)        (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **, double);
    void (*setlbfgsHisTwo)      (lorads_solver *, lorads_lp_dense *, lorads_lp_dense *, lorads_sdp_dense **, lorads_sdp_dense **, double);
    void (*updateDimacsALM)     (lorads_solver *, lorads_sdp_dense **, lorads_sdp_dense **, lorads_lp_dense *, lorads_lp_dense *);
    void (*updateDimacsADMM)    (lorads_solver *, lorads_sdp_dense **, lorads_sdp_dense **, lorads_lp_dense *, lorads_lp_dense *);
    void (*calObj_admm)         (lorads_solver *);
    void (*calObj_alm)          (lorads_solver *);
    void (*updateDimac)         (lorads_solver *);

    void (*admmUpdateVar)       (lorads_solver *, double, double, lorads_int);
}lorads_func;

/**
 * @brief Structure containing ALM algorithm state
 * @details This structure tracks the state of the Augmented Lagrangian Method:
 * - Iteration counters
 * - Penalty parameter
 * - Primal and dual infeasibilities
 * - Objective values
 * - Step size
 */
typedef struct{
    bool is_rank_updated;
    lorads_int outerIter;
    lorads_int innerIter;
    double rho;
    double l_inf_primal_infeasibility;
    double l_1_primal_infeasibility;
    double l_2_primal_infeasibility;
    double primal_dual_gap;
    double primal_objective_value;
    double dual_objective_value;
    double l_inf_dual_infeasibility;
    double l_1_dual_infeasibility;
    double l_2_dual_infeasibility;
    double tau;
}lorads_alm_state;

/**
 * @brief Structure containing ADMM algorithm state
 * @details This structure tracks the state of the Alternating Direction Method of Multipliers:
 * - Iteration counters
 * - Penalty parameter
 * - Primal and dual infeasibilities
 * - Objective values
 * - Convergence metrics
 */
typedef struct{
    lorads_int iter;
    lorads_int nBlks;
    lorads_int cg_iter;
    double rho;
    double l_1_dual_infeasibility;
    double l_inf_dual_infeasibility;
    double l_1_primal_infeasibility;
    double l_inf_primal_infeasibility;
    double l_2_primal_infeasibility;
    double l_2_dual_infeasibility;
    double primal_objective_value;
    double dual_objective_value;
    double primal_dual_gap;
}lorads_admm_state;

/**
 * @brief Structure containing SDP problem constants
 * @details This structure stores various norm values for the objective and constraint data:
 * - L1, L2, and L inf norms for objective coefficients
 * - L1, L2, and L inf norms for right-hand side values
 */
typedef struct{
    double l_1_norm_c;
    double l_2_norm_c;
    double l_inf_norm_c;
    double l_1_norm_b;
    double l_2_norm_b;
    double l_inf_norm_b;
}SDPConst;

#endif
