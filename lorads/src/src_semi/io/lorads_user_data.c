/**
 * @file lorads_user_data.c
 * @brief LORADS User Data Implementation
 * @details This file implements functions for managing user data structures in LORADS,
 * including creation, initialization, and cleanup of data structures for different
 * types of optimization cones (SDP, LP, bound).
 * 
 * @author LORADS Team
 * @date 2024
 */

#include "def_lorads_user_data.h"
#include "lorads_user_data.h"
#include "lorads_utils.h"
#include "lorads_sparse_opts.h"


#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief Check if LP data implies bound constraint on y
 * @param Ldata Pointer to user data structure
 * @return 1 if data represents bound constraints, 0 otherwise
 * @details Determines if the LP data structure represents simple bound constraints
 * by checking if each column contains at most one variable.
 */
static lorads_int LUserDataICheckLpBound( user_data *Ldata ) {
    if ( Ldata->cone != LORADS_CONETYPE_LP ) {
        return 0;
    }
    
    /* If each column holds at most one variable, then it is bound */
    for ( lorads_int i = 0; i < Ldata->nConicRow; ++i ) {
        if ( Ldata->coneMatBeg[i + 1] - Ldata->coneMatBeg[i] >= 2 ) {
            return 0;
        }
    }
    
    return 1;
}

/**
 * @brief Create a new user data structure
 * @param pLdata Pointer to store the created user data structure
 * @details Allocates and initializes a new user data structure with zero values.
 * The structure is used to store optimization problem data for different cone types.
 */
extern void LUserDataCreate( user_data **pLdata ) {
    
    if ( !pLdata ) {
        LORADS_ERROR_TRACE;
    }
    
    user_data *Ldata = NULL;
    LORADS_INIT(Ldata, user_data, 1);
    LORADS_MEMCHECK(Ldata);
    
    LORADS_ZERO(Ldata, user_data, 1);
    *pLdata = Ldata;
}

/**
 * @brief Set cone data in user data structure
 * @param Ldata Pointer to user data structure
 * @param cone Type of cone (SDP, LP, bound)
 * @param nRow Number of rows
 * @param nCol Number of columns
 * @param coneMatBeg Column pointers for sparse matrix
 * @param coneMatIdx Row indices for sparse matrix
 * @param coneMatElem Matrix elements
 * @details Initializes the user data structure with cone-specific data and matrix information.
 */
extern void LUserDataSetConeData( user_data *Ldata, cone_type cone, lorads_int nRow, lorads_int nCol,
                                  lorads_int *coneMatBeg, lorads_int *coneMatIdx, double *coneMatElem ) {
    
    Ldata->cone = cone;
    Ldata->nConicRow = nRow;
    Ldata->nConicCol = nCol;
    Ldata->coneMatBeg = coneMatBeg;
    Ldata->coneMatIdx = coneMatIdx;
    Ldata->coneMatElem = coneMatElem;
    
    return;
}

/**
 * @brief Automatically choose appropriate cone type based on data characteristics
 * @param Ldata Pointer to user data structure
 * @return Selected cone type
 * @details Analyzes the data structure to determine the most appropriate cone type:
 * - Preserves existing bound, sparse SDP, or scalar bound types
 * - For dense SDP, checks sparsity to choose between dense and sparse representation
 * - For LP, checks if it represents bound constraints
 */
extern cone_type LUserDataChooseCone( user_data *Ldata ) {
        
    /* Automatic choice between different cone types*/
    if ( Ldata->cone == LORADS_CONETYPE_BOUND ||
         Ldata->cone == LORADS_CONETYPE_SPARSE_SDP || Ldata->cone == LORADS_CONETYPE_SCALAR_BOUND ) {
        
        return Ldata->cone;
        
    } else if ( Ldata->cone == LORADS_CONETYPE_DENSE_SDP ) {
        
        lorads_int nzSDPCoeffs = csp_nnz_cols(Ldata->nConicRow, &Ldata->coneMatBeg[1]);
        return ( nzSDPCoeffs > LORADS_SPARSE_CONE_THRESHOLD * Ldata->nConicRow ) ? \
                LORADS_CONETYPE_DENSE_SDP : LORADS_CONETYPE_SPARSE_SDP;
        
    } else if ( Ldata->cone == LORADS_CONETYPE_LP ) {
        
        if ( LUserDataICheckLpBound(Ldata) ) {
            return LORADS_CONETYPE_BOUND;
        } else {
            return LORADS_CONETYPE_LP;
        }
        
    }
    
    return LORADS_CONETYPE_UNKNOWN;
}

/**
 * @brief Clear user data structure
 * @param Ldata Pointer to user data structure
 * @details Resets all fields in the user data structure to zero.
 */
extern void LUserDataClear( user_data *Ldata ) {
    
    if ( !Ldata ) {
        return;
    }
    
    LORADS_ZERO(Ldata, user_data, 1);
    
    return;
}

/**
 * @brief Destroy user data structure
 * @param pLdata Pointer to pointer of user data structure
 * @details Frees memory allocated for the user data structure and sets pointer to NULL.
 */
extern void LUserDataDestroy( user_data **pLdata ) {
    
    if ( !pLdata ) {
        return;
    }
    
    LUserDataClear(*pLdata);
    LORADS_FREE(*pLdata);
    
    return;
}

/**
 * @brief Create array of SDP data structures
 * @param SDPDatas Pointer to store array of SDP data structures
 * @param nCones Number of SDP cones to create
 * @details Allocates memory for an array of user data structures for SDP cones.
 */
extern void LORADSCreateSDPDatas(user_data ***SDPDatas, lorads_int nCones){
    user_data **SDPDatasTemp;
    LORADS_INIT(SDPDatasTemp, user_data *, nCones);
    *SDPDatas = SDPDatasTemp;
}

/**
 * @brief Clear and free all user data structures
 * @param coneMatBeg Array of cone matrix column pointers
 * @param coneMatIdx Array of cone matrix row indices
 * @param coneMatElem Array of cone matrix elements
 * @param nBlks Number of blocks
 * @param BlkDims Array of block dimensions
 * @param rowRHS Array of right-hand side values
 * @param LpMatBeg LP matrix column pointers
 * @param LpMatIdx LP matrix row indices
 * @param LpMatElem LP matrix elements
 * @param SDPDatas Array of SDP data structures
 * @details Frees all memory allocated for optimization problem data structures.
 */
extern void LORADSClearUsrData(lorads_int **coneMatBeg, lorads_int **coneMatIdx, double **coneMatElem, lorads_int nBlks,
                               lorads_int *BlkDims, double *rowRHS, lorads_int *LpMatBeg, lorads_int *LpMatIdx, double *LpMatElem, user_data **SDPDatas){
    for (lorads_int iBlk = 0; iBlk < nBlks; ++iBlk)
    {
        LORADS_FREE(coneMatBeg[iBlk]);
        LORADS_FREE(coneMatIdx[iBlk]);
        LORADS_FREE(coneMatElem[iBlk]);
    }
    LORADS_FREE(BlkDims);
    LORADS_FREE(rowRHS);
    LORADS_FREE(LpMatBeg);
    LORADS_FREE(LpMatIdx);
    LORADS_FREE(LpMatElem);
    LORADS_FREE(coneMatBeg);
    LORADS_FREE(coneMatIdx);
    LORADS_FREE(coneMatElem);
    LORADS_FREE(SDPDatas);
}