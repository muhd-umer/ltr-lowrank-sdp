/**
 * @file main.c
 * @brief LORADS Main Program
 * @details This file contains the main program for the LORADS (Low-Rank ADMM Splitting) solver
 * for Semidefinite Programming. The program:
 * - Parses command line arguments
 * - Reads problem data
 * - Initializes solver
 * - Runs optimization
 * - Outputs results
 * 
 * @author LORADS Team
 * @date 2024
 */

#include <stdio.h>
#include <math.h>

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

// Include necessary header files for LORADS components
#include "lorads_file_io.h"
#include "def_lorads_user_data.h"
#include "lorads_user_data.h"
#include "lorads_utils.h"
#include "def_lorads_solver.h"
#include "lorads_solver.h"
#include "lorads_alm.h"
#include "lorads_admm.h"
#include "lorads_alg_common.h"
#include "lorads_logging.h"

/**
 * @brief Initialize default parameters for the solver
 * @param params Pointer to solver parameters structure
 * @details These parameters control the behavior of ALM and ADMM algorithms:
 * - initRho: Initial penalty parameter for ALM/ADMM
 * - rhoMax: Maximum allowed penalty parameter
 * - rhoCellingALM/ADMM: Upper bounds for penalty parameters
 * - maxALMIter/maxADMMIter: Maximum iterations for each phase
 * - timesLogRank: Factor for rank estimation, initial rank = log(n) * timesLogRank
 * - rhoFreq: Frequency of penalty parameter updates
 * - rhoFactor/ALMRhoFactor: Factors for increasing penalty parameters
 * - phase1Tol/phase2Tol: Convergence tolerances for each phase
 * - timeSecLimit: Maximum runtime in seconds
 * - heuristicFactor: Factor for heuristic adjustments, between phase1 and phase2
 * - lbfgsListLength: Length of L-BFGS history
 * - endTauTol/endALMSubTol: Final tolerances for tau (ALM subproblem's step size) and ALM subproblems
 * - l2Rescaling: Whether to use L2 rescaling
 * - reoptLevel: Level of reoptimization, 0-3, larger means more reoptimization
 * - dyrankLevel: Level of dynamic rank adjustment, 0-3, larger means more dynamic rank adjustment
 * - highAccMode: Whether to use high accuracy mode, 0-1, 1 means high accuracy mode
 */
void initCommandLineArgs(lorads_params *params)
{
    params->fname = "NULL";                    ///< Input file name
    params->logFile = NULL;                   ///< Optional log file path (NULL to disable)
    params->jsonFile = NULL;                  ///< Optional JSON output file path (NULL to disable)
    params->initRho = 0.0;                     ///< Initial penalty parameter
    params->rhoMax = 5000.0;                   ///< Maximum penalty parameter
    params->rhoCellingALM = 1e+8;             ///< Upper bound for ALM penalty parameter
    params->rhoCellingADMM = params->rhoMax * 200;  ///< Upper bound for ADMM penalty parameter
    params->maxALMIter = 200;                  ///< Maximum ALM iterations
    params->maxADMMIter = 10000;              ///< Maximum ADMM iterations
    params->timesLogRank = 2.0;               ///< Factor for rank estimation
    params->fixedRank = -1;                   ///< Fixed rank override (-1 disables)
    params->initRank = -1;                    ///< Initial rank for dynamic adjustment (-1 disables)
    params->rhoFreq = 5;                      ///< Frequency of penalty parameter updates
    params->rhoFactor = 1.2;                  ///< Factor for increasing penalty parameter
    params->ALMRhoFactor = 2.0;               ///< Factor for increasing ALM penalty parameter
    params->rankUpdateFactor = 1.5;           ///< Factor for increasing rank during dynamic adjustment
    params->phase1Tol = 1e-3;                 ///< Tolerance for phase 1 (coarse optimization)
    params->phase2Tol = 1e-5;                 ///< Tolerance for phase 2 (fine optimization)
    params->timeSecLimit = 3600.0;            ///< Time limit in seconds
    params->heuristicFactor = 1.0;            ///< Factor for heuristic adjustments
    params->lbfgsListLength = 2;              ///< Length of L-BFGS history
    params->endTauTol = 1e-16;                ///< Final tolerance for tau
    params->endALMSubTol = 1e-10;             ///< Final tolerance for ALM subproblems
    params->l2Rescaling = false;              ///< Whether to use L2 rescaling
    params->reoptLevel = 2;                   ///< Level of reoptimization
    params->dyrankLevel = 2;                  ///< Level of dynamic rank adjustment
    params->highAccMode = false;              ///< Whether to use high accuracy mode
    params->oracleRankMethod = LORADS_ORACLE_RANK_GRAM; ///< Oracle rank computation method
}

/**
 * @brief Print summary information about the problem dimensions
 * @param nConstrs Number of constraints
 * @param nBlks Number of SDP blocks
 * @param nLpCols Number of LP columns
 * @details Prints the dimensions of the optimization problem, including:
 * - Number of constraints
 * - Number of SDP blocks
 * - Number of LP columns
 */
static void print_summary_info(lorads_int nConstrs, lorads_int nBlks, lorads_int nLpCols){
#ifdef INT32
    printf("nConstrs = %d, sdp nBlks = %d, lp Cols = %d\n", nConstrs, nBlks, nLpCols);
#endif
#ifdef MAC_INT64
    printf("nConstrs = %lld, sdp nBlks = %lld, lp Cols = %lld\n", nConstrs, nBlks, nLpCols);
#endif
#ifdef UNIX_INT64
    printf("nConstrs = %ld, sdp nBlks = %ld, lp Cols = %ld\n", nConstrs, nBlks, nLpCols);
#endif
}

/**
 * @brief Command line options structure for LORADS solver
 * @details Defines all available command line options for configuring the LORADS solver.
 * Each option is defined with:
 * - name: Long option name
 * - has_arg: Whether the option requires an argument
 * - flag: Pointer to flag variable (0 for direct return)
 * - val: Value to return when option is found
 * 
 * The options control various aspects of the solver including:
 * - Penalty parameters and their update rules
 * - Iteration limits and convergence criteria
 * - Rank estimation and adjustment
 * - Algorithm behavior and accuracy settings
 */
static struct option long_options[] = {
        {"logfile", required_argument, 0, 1025},           ///< Optional log file path
        {"jsonfile", required_argument, 0, 1026},          ///< Optional JSON output file path
        {"initRho", required_argument, 0, 1000},           ///< Initial penalty parameter for ALM/ADMM
        {"rhoMax", required_argument, 0, 1001},            ///< Maximum allowed penalty parameter
        {"rhoCellingALM", required_argument, 0, 1002},     ///< Upper bound for ALM penalty parameter
        {"rhoCellingADMM", required_argument, 0, 1003},    ///< Upper bound for ADMM penalty parameter
        {"maxALMIter", required_argument, 0, 1004},        ///< Maximum number of ALM iterations
        {"maxADMMIter", required_argument, 0, 1005},       ///< Maximum number of ADMM iterations
        {"timesLogRank", required_argument, 0, 1006},      ///< Factor for rank estimation (initial rank = log(n) * timesLogRank)
        {"fixedRank", required_argument, 0, 1022},         ///< Fixed rank override for all cones
        {"initRank", required_argument, 0, 1023},          ///< Initial rank for dynamic adjustment
        {"rhoFreq", required_argument, 0, 1007},           ///< Frequency of penalty parameter updates
        {"rhoFactor", required_argument, 0, 1008},         ///< Factor for increasing penalty parameter
        {"ALMRhoFactor", required_argument, 0, 1009},      ///< Factor for increasing ALM penalty parameter
        {"rankUpdateFactor", required_argument, 0, 1024},  ///< Factor for increasing rank during dynamic adjustment
        {"phase1Tol", required_argument, 0, 1010},         ///< Convergence tolerance for phase 1 (coarse optimization)
        {"phase2Tol", required_argument, 0, 1011},         ///< Convergence tolerance for phase 2 (fine optimization)
        {"timeSecLimit", required_argument, 0, 1012},      ///< Maximum runtime in seconds
        {"heuristicFactor", required_argument, 0, 1013},   ///< Factor for heuristic adjustments between phases
        {"lbfgsListLength", required_argument, 0, 1014},   ///< Length of L-BFGS history for direction computation
        {"endTauTol", required_argument, 0, 1015},         ///< Final tolerance for tau (ALM subproblem's step size)
        {"endALMSubTol", required_argument, 0, 1016},      ///< Final tolerance for ALM subproblems
        {"l2Rescaling", required_argument, 0, 1017},       ///< Enable/disable L2 rescaling (0/1)
        {"reoptLevel", required_argument, 0, 1018},        ///< Level of reoptimization (0-3, higher means more reoptimization)
        {"dyrankLevel", required_argument, 0, 1019},       ///< Level of dynamic rank adjustment (0-3, higher means more adjustment)
        {"highAccMode", required_argument, 0, 1020},       ///< Enable/disable high accuracy mode (0/1)
        {"oracleRankNaive", no_argument, 0, 1021},         ///< Use naive oracle rank computation
        {0, 0, 0, 0}                                      ///< End of options array
};

/**
 * @brief Print the input parameters for the solver
 * @param params Solver parameters structure
 * @details Prints all solver parameters in a formatted table, including:
 * - File name
 * - Penalty parameters
 * - Iteration limits
 * - Convergence tolerances
 * - Algorithm settings
 */
static void printInput(lorads_params params){
    printf("Input parameters:\n");
    printf("----------------------------------------------\n");
#ifdef INT32
    printf("fname = %s\n", params.fname);
    printf("initRho = %f\n", params.initRho);
    printf("rhoMax = %f\n", params.rhoMax);
    printf("rhoCellingALM = %f\n", params.rhoCellingALM);
    printf("rhoCellingADMM = %f\n", params.rhoCellingADMM);
    printf("maxALMIter = %d\n", params.maxALMIter);
    printf("maxADMMIter = %d\n", params.maxADMMIter);
    printf("timesLogRank = %f\n", params.timesLogRank);
    printf("fixedRank = %d\n", params.fixedRank);
    printf("initRank = %d\n", params.initRank);
    printf("rhoFreq = %d\n", params.rhoFreq);
    printf("rhoFactor = %f\n", params.rhoFactor);
    printf("ALMRhoFactor = %f\n", params.ALMRhoFactor);
    printf("rankUpdateFactor = %f\n", params.rankUpdateFactor);
    printf("phase1Tol = %f\n", params.phase1Tol);
    printf("phase2Tol = %f\n", params.phase2Tol);
    printf("timeSecLimit = %f\n", params.timeSecLimit);
    printf("heuristicFactor = %f\n", params.heuristicFactor);
    printf("lbfgsListLength = %d\n", params.lbfgsListLength);
    printf("endTauTol = %f\n", params.endTauTol);
    printf("endALMSubTol = %f\n", params.endALMSubTol);
    printf("l2Rescaling = %d\n", params.l2Rescaling);
    printf("reoptLevel = %d\n", params.reoptLevel);
    printf("dyrankLevel = %d\n", params.dyrankLevel);
    printf("highAccMode = %d\n", params.highAccMode);
    printf("oracleRankMethod = %d\n", params.oracleRankMethod);
#endif
#ifdef MAC_INT64
    printf("fname = %s\n", params.fname);
    printf("initRho = %f\n", params.initRho);
    printf("rhoMax = %f\n", params.rhoMax);
    printf("rhoCellingALM = %f\n", params.rhoCellingALM);
    printf("rhoCellingADMM = %f\n", params.rhoCellingADMM);
    printf("maxALMIter = %lld\n", params.maxALMIter);
    printf("maxADMMIter = %lld\n", params.maxADMMIter);
    printf("timesLogRank = %f\n", params.timesLogRank);
    printf("fixedRank = %lld\n", params.fixedRank);
    printf("initRank = %lld\n", params.initRank);
    printf("rhoFreq = %lld\n", params.rhoFreq);
    printf("rhoFactor = %f\n", params.rhoFactor);
    printf("ALMRhoFactor = %f\n", params.ALMRhoFactor);
    printf("rankUpdateFactor = %f\n", params.rankUpdateFactor);
    printf("phase1Tol = %f\n", params.phase1Tol);
    printf("phase2Tol = %f\n", params.phase2Tol);
    printf("timeSecLimit = %f\n", params.timeSecLimit);
    printf("heuristicFactor = %f\n", params.heuristicFactor);
    printf("lbfgsListLength = %lld\n", params.lbfgsListLength);
    printf("endTauTol = %f\n", params.endTauTol);
    printf("endALMSubTol = %f\n", params.endALMSubTol);
    printf("l2Rescaling = %d\n", params.l2Rescaling);
    printf("reoptLevel = %lld\n", params.reoptLevel);
    printf("dyrankLevel = %lld\n", params.dyrankLevel);
    printf("highAccMode = %d\n", params.highAccMode);
    printf("oracleRankMethod = %d\n", params.oracleRankMethod);
#endif
#ifdef UNIX_INT64
    printf("fname = %s\n", params.fname);
    printf("initRho = %f\n", params.initRho);
    printf("rhoMax = %f\n", params.rhoMax);
    printf("rhoCellingALM = %f\n", params.rhoCellingALM);
    printf("rhoCellingADMM = %f\n", params.rhoCellingADMM);
    printf("maxALMIter = %ld\n", params.maxALMIter);
    printf("maxADMMIter = %ld\n", params.maxADMMIter);
    printf("timesLogRank = %f\n", params.timesLogRank);
    printf("fixedRank = %ld\n", params.fixedRank);
    printf("initRank = %ld\n", params.initRank);
    printf("rhoFreq = %ld\n", params.rhoFreq);
    printf("rhoFactor = %f\n", params.rhoFactor);
    printf("ALMRhoFactor = %f\n", params.ALMRhoFactor);
    printf("rankUpdateFactor = %f\n", params.rankUpdateFactor);
    printf("phase1Tol = %f\n", params.phase1Tol);
    printf("phase2Tol = %f\n", params.phase2Tol);
    printf("timeSecLimit = %f\n", params.timeSecLimit);
    printf("heuristicFactor = %f\n", params.heuristicFactor);
    printf("lbfgsListLength = %ld\n", params.lbfgsListLength);
    printf("endTauTol = %f\n", params.endTauTol);
    printf("endALMSubTol = %f\n", params.endALMSubTol);
    printf("l2Rescaling = %d\n", params.l2Rescaling);
    printf("reoptLevel = %ld\n", params.reoptLevel);
    printf("dyrankLevel = %ld\n", params.dyrankLevel);
    printf("highAccMode = %d\n", params.highAccMode);
    printf("oracleRankMethod = %d\n", params.oracleRankMethod);
#endif
    printf("----------------------------------------------\n");
}

int main(int argc, char *argv[]) {
    lorads_retcode retcode;
    lorads_params params;
    initCommandLineArgs(&params);

    int opt, long_index = 0;
    params.fname = argv[1];

    while ((opt = getopt_long(argc, argv, "r:", long_options, &long_index)) != -1){
        switch(opt){
            case 1025:
                params.logFile = optarg;
                break;
            case 1026:
                params.jsonFile = optarg;
                break;
            case 1000:
                params.initRho = atof(optarg);
                break;
            case 1001:
                params.rhoMax = atof(optarg);
                break;
            case 1002:
                params.rhoCellingALM = atof(optarg);
                break;
            case 1003:
                params.rhoCellingADMM = atof(optarg);
                break;
            case 1004:
                params.maxALMIter = atoi(optarg);
                break;
            case 1005:
                params.maxADMMIter = atoi(optarg);
                break;
            case 1006:
                params.timesLogRank = atof(optarg);
                break;
            case 1022:
                params.fixedRank = atoi(optarg);
                break;
            case 1023:
                params.initRank = atoi(optarg);
                break;
            case 1007:
                params.rhoFreq = atoi(optarg);
                break;
            case 1008:
                params.rhoFactor = atof(optarg);
                break;
            case 1009:
                params.ALMRhoFactor = atof(optarg);
                break;
            case 1024:
                params.rankUpdateFactor = atof(optarg);
                break;
            case 1010:
                params.phase1Tol = atof(optarg);
                break;
            case 1011:
                params.phase2Tol = atof(optarg);
                break;
            case 1012:
                params.timeSecLimit = atof(optarg);
                break;
            case 1013:
                params.heuristicFactor = atof(optarg);
                break;
            case 1014:
                params.lbfgsListLength = atoi(optarg);
                break;
            case 1015:
                params.endTauTol = atof(optarg);
                break;
            case 1016:
                params.endALMSubTol = atof(optarg);
                break;
            case 1017:
                params.l2Rescaling = atoi(optarg);
                break;
            case 1018:
                params.reoptLevel = atoi(optarg);
                break;
            case 1019:
                params.dyrankLevel = atoi(optarg);
                break;
            case 1020:
                params.highAccMode = atoi(optarg);
                break;
            case 1021:
                params.oracleRankMethod = LORADS_ORACLE_RANK_NAIVE;
                break;
        }
    }

    params.rhoCellingADMM = params.rhoMax * 200;


    printf("-----------------------------------------------------------\n");
    printf("  L         OOO      RRRR       A      DDDD       SSS \n");
    printf("  L        O   O     R   R     A A     D   D     S    \n");
    printf("  L        O   O     RRRR     AAAAA    D   D      SSS \n");
    printf("  L        O   O     R  R     A   A    D   D         S\n");
    printf("  LLLLL     OOO      R   R    A   A    DDDD       SSS \n");
    printf("-----------------------------------------------------------\n");


    printInput(params);
    lorads_int nConstrs = 0;
    lorads_int nBlks = 0;
    lorads_int *BlkDims = NULL;
    double *rowRHS = NULL;
    lorads_int **coneMatBeg = NULL;
    lorads_int **coneMatIdx = NULL;
    double **coneMatElem = NULL;
    lorads_int nLpCols = 0;
    lorads_int *LpMatBeg = NULL;
    lorads_int *LpMatIdx = NULL;
    double *LpMatElem = NULL;
    lorads_int nCols = 0;
    lorads_int nElem = 0;
    user_data **SDPDatas = NULL;
    LUtilStartCtrlCCheck();

    double timeStart = LUtilGetTimeStamp();
    retcode = LReadSDPA(params.fname, &nConstrs, &nBlks, &BlkDims, &rowRHS, &coneMatBeg,
                        &coneMatIdx, &coneMatElem, &nCols, &nLpCols, &LpMatBeg,
                        &LpMatIdx, &LpMatElem, &nElem);
    if (retcode != LORADS_RETCODE_OK) {
        goto freeUsrData;
    }
    printf("Reading SDPA file in %f seconds \n", LUtilGetTimeStamp() - timeStart);
    print_summary_info(nConstrs, nBlks, nLpCols);

    lorads_solver *ASolver = NULL;
    double timeSolveStart = LUtilGetTimeStamp();
    double solvingTime = 0.0;
    // create a empty solver
    LORADS_INIT(ASolver, lorads_solver, 1);
    LORADS_INIT(ASolver->var, lorads_variable, 1);
    LORADSInitSolver(ASolver, nConstrs, nBlks, BlkDims, nLpCols);

    // // allocate memory for SDPDatas
    LORADS_INIT(SDPDatas, user_data *, nBlks);
    LORADS_MEMCHECK(SDPDatas);
    LORADSSetDualObjective(ASolver, rowRHS);
    LORADSInitConeData(ASolver, SDPDatas, coneMatElem, coneMatBeg, coneMatIdx, BlkDims, nConstrs, nBlks, nLpCols,
                       LpMatBeg, LpMatIdx, LpMatElem);
    LORADSPreprocess(ASolver, BlkDims);

    LORADSDetermineRank(ASolver, BlkDims, params.timesLogRank, params.fixedRank, params.initRank);
//    detectSparsitySDPCoeff(ASolver);

    // ALM allocate
    LORADSInitALMVars(ASolver, ASolver->var->rankElem, BlkDims, nBlks, nLpCols, params.lbfgsListLength);
    ASolver->hisRecT = params.lbfgsListLength;
    // ADMM allocate
    LORADSInitADMMVars(ASolver, ASolver->var->rankElem, BlkDims, nBlks, nLpCols);

    // initial state
    SDPConst sdpConst;
    lorads_alm_state alm_state_pointer;
    lorads_admm_state admm_state_pointer;
    initial_solver_state(&params, ASolver, &alm_state_pointer, &admm_state_pointer, &sdpConst);
    lorads_logging_init(ASolver, &params, timeSolveStart);
//    printfProbInfo(ASolver);
    double reopt_param = 5;
    lorads_int reopt_alm_iter = 3;
    lorads_int reopt_admm_iter = 50;
    double reopt_time_new = LUtilGetTimeStamp();
    lorads_int alm_reopt_min_iter = 3;
    lorads_int admm_reopt_min_iter = 50;


    double initial_solving_time = 0.0;
    double all_time = 0.0;
    double all_dual_infea = 0.0;
    int admm_bad_iter_flag;
    admm_bad_iter_flag = 0;


    CPU_OPTIMIZE:
    if (params.highAccMode){
        alm_reopt_min_iter = 3;
        admm_reopt_min_iter = 1000;
    }else{
        alm_reopt_min_iter = 3;
        admm_reopt_min_iter = 50;
    }
    ASolver->AStatus = LORADS_UNKNOWN;
    printf("-----------------------------------------------------------------------\n");
    printf("Start solving by ALM and ADMM\n");
    printf("-----------------------------------------------------------------------\n");
    double all_time_start = LUtilGetTimeStamp();
    double time_start = LUtilGetTimeStamp();
    LORADS_ALMOptimize(&params, ASolver, &alm_state_pointer, params.maxALMIter, timeSolveStart);
    if (LUtilGetTimeStamp() - timeSolveStart > params.timeSecLimit){
        printf("Time limit reached\n");
        ASolver->AStatus = LORADS_TIME_LIMIT;
        goto END_SOLVING;
    }
    LORADS_ALMtoADMM(ASolver, &params, &alm_state_pointer, &admm_state_pointer);
    if (LORADSADMMOptimize(&params, ASolver, &admm_state_pointer, params.maxADMMIter, timeSolveStart) == RET_CODE_BAD_ITER){
        admm_bad_iter_flag = 1;
    }

    // if (admm_state_pointer.primal_dual_gap >= params.phase1Tol)
    // {
    //     params.phase1Tol *= 0.1;
    //     params.phase1Tol = LORADS_MAX(params.phase1Tol, params.phase2Tol);
    // }
    double time_end = LUtilGetTimeStamp();
    initial_solving_time = time_end - time_start;
    all_time += initial_solving_time;
//    double timeBeforeCalDualInfeasibility = LUtilGetTimeStamp();
//    PRINT_AFTER_INI_SOLVING:
//    calculate_dual_infeasibility_solver(ASolver);
//    double timeAfterCalDualInfeasibility = LUtilGetTimeStamp();
//    printf("Calculate dual infeasibility in %f seconds \n", timeAfterCalDualInfeasibility - timeBeforeCalDualInfeasibility);
//    if (LUtilGetTimeStamp() - timeSolveStart > params.timeSecLimit){
//        printf("Time limit reached\n");
//        ASolver->AStatus = LORADS_TIME_LIMIT;
//        goto END_SOLVING;
//    }
//    printf("-----------------------------------------------------------------------\n");
//    printf("After initial solving results:\n");
//    printRes(ASolver->pObjVal, ASolver->dObjVal,
//             ASolver->dimacError[ LORADS_DIMAC_ERROR_CONSTRVIO_L1],
//             ASolver->dimacError[ LORADS_DIMAC_ERROR_DUALFEASIBLE_L1],
//             ASolver->dimacError[ LORADS_DIMAC_ERROR_PDGAP],
//             ASolver->dimacError[ LORADS_DIMAC_ERROR_CONSTRVIO_L1] * (1 + ASolver->bRHSNrm1) / (1 + ASolver->bRHSNrmInf),
//             ASolver->dimacError[ LORADS_DIMAC_ERROR_DUALFEASIBLE_L1] * (1 + ASolver->cObjNrm1) / (1 + ASolver->cObjNrmInf));
//    printf("-----------------------------------------------------------------------\n");
    double timeBeforeReopt = LUtilGetTimeStamp();
    double reopt_time = LUtilGetTimeStamp();
    int cnt = 0;
    if (params.reoptLevel >= 1) {
        while ((alm_state_pointer.primal_dual_gap > params.phase2Tol ||
                alm_state_pointer.l_1_primal_infeasibility > params.phase2Tol) &&
               (admm_state_pointer.primal_dual_gap > params.phase2Tol ||
                admm_state_pointer.l_1_primal_infeasibility > params.phase2Tol)){
            if (cnt >= 1) {
                break;
            }
            printf("******  reopt parameter:%.3f\n", reopt_param);
            time_start = LUtilGetTimeStamp();
            reopt(&params, ASolver, &alm_state_pointer, &admm_state_pointer, &reopt_param,
                                    &alm_reopt_min_iter, &admm_reopt_min_iter, timeSolveStart, &admm_bad_iter_flag, 1);
            time_end = LUtilGetTimeStamp();
            all_time += (time_end - time_start);
//            printf("reopt round %d in %f seconds \n", cnt, LUtilGetTimeStamp() - reopt_time);
            cnt += 1;
            if (LUtilGetTimeStamp() - timeSolveStart > params.timeSecLimit){
                printf("Time limit reached\n");
                ASolver->AStatus = LORADS_TIME_LIMIT;
                goto END_SOLVING;
            }
        }
    }
    time_start = LUtilGetTimeStamp();
    calculate_dual_infeasibility_solver(ASolver);
    time_end = LUtilGetTimeStamp();
    all_dual_infea += (time_end - time_start);
    all_time += (time_end - time_start);
    admm_state_pointer.l_1_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1];
    admm_state_pointer.l_inf_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1] * (1 + ASolver->cObjNrm1) / (1 + ASolver->cObjNrmInf);
    admm_state_pointer.l_2_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1] * (1 + ASolver->cObjNrm1) / (1 + ASolver->cObjNrm2);
    admm_state_pointer.primal_dual_gap = ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP];
    admm_state_pointer.l_1_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1];
    admm_state_pointer.l_inf_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] * (1 + ASolver->bRHSNrm1) / (1 + ASolver->bRHSNrmInf);
    admm_state_pointer.l_2_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] * (1 + ASolver->bRHSNrm1) / (1 + ASolver->bRHSNrm2);
    printf("-----------------------------------------------------------------------\n");
    printf("Dual infeasibility: l_1 = %f, l_inf = %f, l_2 = %f\n", admm_state_pointer.l_1_dual_infeasibility, admm_state_pointer.l_inf_dual_infeasibility, admm_state_pointer.l_2_dual_infeasibility);
    printf("-----------------------------------------------------------------------\n");
    if (params.reoptLevel >= 2){
        int dual_cnt = 0;
        while(admm_state_pointer.l_1_dual_infeasibility > params.phase2Tol || admm_state_pointer.primal_dual_gap > params.phase2Tol || admm_state_pointer.l_1_primal_infeasibility > params.phase2Tol){
            if (dual_cnt >= 2){
                break;
            }
            // if (admm_state_pointer.primal_dual_gap >= params.phase2Tol * 1e1)
            // {
            //     params.phase1Tol *= 0.1;
            //     params.phase1Tol = LORADS_MAX(params.phase1Tol, params.phase2Tol);
            // }
            if (!params.highAccMode && admm_state_pointer.l_1_dual_infeasibility <= 5 * params.phase2Tol && admm_state_pointer.primal_dual_gap <= 5 * params.phase2Tol && admm_state_pointer.l_1_primal_infeasibility <= 1 * params.phase2Tol)
            {
                break;
            }
//            if (dual_cnt >= 1){
//                double one = 1.0;
//                reopt(&params, ASolver, &alm_state_pointer, &admm_state_pointer, &one, &reopt_alm_iter, &reopt_admm_iter);
//            }else{
            printf("******  reopt parameter:%.3f\n", reopt_param);
            time_start = LUtilGetTimeStamp();
            reopt(&params, ASolver, &alm_state_pointer, &admm_state_pointer, &reopt_param, &reopt_alm_iter, &reopt_admm_iter, timeSolveStart, &admm_bad_iter_flag, 2);


            if (ASolver->nLpCols > 0){
                averageUVLP(ASolver->var->uLp, ASolver->var->vLp, ASolver->var->rLp);
            }
            for (lorads_int iCone = 0; iCone <  ASolver->nCones; iCone++){
                averageUV(ASolver->var->U[iCone], ASolver->var->V[iCone], ASolver->var->R[iCone]);
            }
            if (ASolver->nLpCols > 0){
                copyRtoVLP(ASolver->var->rLp, ASolver->var->vLp, ASolver->var->R, ASolver->var->V, ASolver->nCones);
            }else{
                copyRtoV(ASolver->var->rLp, ASolver->var->vLp, ASolver->var->R, ASolver->var->V, ASolver->nCones);
            }

            time_end = LUtilGetTimeStamp();
            all_time += (time_end - time_start);

            DUAL_INFEASIBILITY:
            time_start = LUtilGetTimeStamp();
            calculate_dual_infeasibility_solver(ASolver);
            time_end = LUtilGetTimeStamp();
            all_dual_infea += (time_end - time_start);
            all_time += (time_end - time_start);
            admm_state_pointer.l_1_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1];
            admm_state_pointer.l_inf_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1] * (1 + ASolver->cObjNrm1) / (1 + ASolver->cObjNrmInf);
            admm_state_pointer.l_2_dual_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_DUALFEASIBLE_L1] * (1 + ASolver->cObjNrm1) / (1 + ASolver->cObjNrm2);
            admm_state_pointer.primal_dual_gap = ASolver->dimacError[LORADS_DIMAC_ERROR_PDGAP];
            admm_state_pointer.l_1_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1];
            admm_state_pointer.l_inf_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] * (1 + ASolver->bRHSNrm1) / (1 + ASolver->bRHSNrmInf);
            admm_state_pointer.l_2_primal_infeasibility = ASolver->dimacError[LORADS_DIMAC_ERROR_CONSTRVIO_L1] * (1 + ASolver->bRHSNrm1) / (1 + ASolver->bRHSNrm2);
            printf("-----------------------------------------------------------------------\n");
            printf("reopt %d:Dual infeasibility: l_1 = %f, l_inf = %f, l_2 = %f\n", dual_cnt, admm_state_pointer.l_1_dual_infeasibility, admm_state_pointer.l_inf_dual_infeasibility, admm_state_pointer.l_2_dual_infeasibility);
            printf("-----------------------------------------------------------------------\n");
            dual_cnt += 1;
            if (LUtilGetTimeStamp() - timeSolveStart > params.timeSecLimit){
                printf("Time limit reached\n");
                ASolver->AStatus = LORADS_TIME_LIMIT;
                goto END_SOLVING;
            }
        }
    }
    PRINTRES:
    if (admm_state_pointer.l_1_dual_infeasibility <= 5 * params.phase2Tol && admm_state_pointer.primal_dual_gap <= 5 * params.phase2Tol && admm_state_pointer.l_1_primal_infeasibility <= 1 * params.phase2Tol)
    {
            ASolver->AStatus = LORADS_PRIMAL_DUAL_OPTIMAL;
    }
    else if (admm_state_pointer.primal_dual_gap <= 5 * params.phase2Tol && admm_state_pointer.l_1_primal_infeasibility <= 1 * params.phase2Tol)
    {
            ASolver->AStatus = LORADS_PRIMAL_OPTIMAL;
        }else{
            ASolver->AStatus = LORADS_MAXITER;
        }
    END_SOLVING:
        /* Compute final oracle rank and write JSON output */
        lorads_int final_oracle_rank = lorads_compute_oracle_rank(ASolver, 2);
        if (final_oracle_rank < 0)
        {
            final_oracle_rank = 0;
        }
        lorads_write_json_output(ASolver, final_oracle_rank, admm_state_pointer.primal_objective_value, admm_state_pointer.dual_objective_value, admm_state_pointer.l_1_primal_infeasibility, admm_state_pointer.l_inf_primal_infeasibility, admm_state_pointer.primal_dual_gap, all_time, params.rhoMax, params.heuristicFactor);

        lorads_logging_close(ASolver);
        LORADSEndProgram(ASolver);
    double all_time_end = LUtilGetTimeStamp();
    all_time = all_time_end - all_time_start;
    exit_cleanup:
    if (ASolver->AStatus != LORADS_TIME_LIMIT){
        printf("initial solving: %f\n", initial_solving_time);
        printf("all_time - all_dual_infea: %f\n", all_time - all_dual_infea);
        printf("all_dual_infea: %f\n", all_dual_infea);
        printf("all_time: %f\n", all_time);
    }else{
        printf("Time limit reached :%f\n", params.timeSecLimit);
        printf("initial solving: %f\n", initial_solving_time);
        printf("all_time - all_dual_infea: %f\n", all_time - all_dual_infea);
        printf("all_dual_infea: %f\n", all_dual_infea);
        printf("all_time: %f\n", all_time);
    }

    freeUVS:
        // free memory
        LORADSDestroyADMMVars(ASolver);
        LORADSDestroyALMVars(ASolver);
        LORADS_FREE(ASolver->sparsitySDPCoeff);
        LORADS_FREE(ASolver->var->rankElem);
        destroyPreprocess(ASolver);
        LORADSDestroyConeData(ASolver);
    freeSolver:
        LORADSDestroySolver(ASolver);
        LORADS_FREE(ASolver);
    freeUsrData:
        // clear user data
        LORADSClearUsrData(coneMatBeg, coneMatIdx, coneMatElem, nBlks, BlkDims, rowRHS, LpMatBeg, LpMatIdx, LpMatElem, SDPDatas);
    return 0;
}
