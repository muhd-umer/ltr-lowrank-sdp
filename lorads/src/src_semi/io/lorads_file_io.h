/**
 * @file lorads_file_io.h
 * @brief LORADS File I/O Interface Definitions
 * @details This header file defines the interface for reading and writing SDPA format files
 * in the LORADS system. It provides functions for parsing SDPA files and handling
 * semidefinite programming data structures.
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef LORADS_FILE_IO_H
#define LORADS_FILE_IO_H

/** @file hdsdp\_file\_io.h
 * HDSDP input and output for SDPA format
 */
#ifndef LORADS_FILE_READER_H
#define LORADS_FILE_READER_H
#include "lorads.h"
#include "lorads_utils.h"

/**
 * @brief Read and parse an SDPA format file
 * @param fname Input file name
 * @param pnConstrs Pointer to store number of constraints (row dimension of matrix A, dimension of b)
 * @param pnBlks Pointer to store number of semidefinite blocks
 * @param pblkDims Pointer to store dimensions of semidefinite blocks
 * @param prowRHS Pointer to store dual coefficients (b in Ax=b)
 * @param pconeMatBeg Pointer to store cone matrix column pointers
 * @param pconeMatIdx Pointer to store cone matrix row indices
 * @param pconeMatElem Pointer to store cone matrix elements
 * @param pnCols Pointer to store dimension of all variables except LP variables
 * @param pnLPCols Pointer to store dimension of LP variables
 * @param pLpMatBeg Pointer to store LP matrix column pointers
 * @param pLpMatIdx Pointer to store LP matrix row indices
 * @param pLpMatElem Pointer to store LP matrix elements
 * @param pnElems Pointer to store number of non-zero elements in A (excluding LP variables)
 * @return LORADS_RETCODE_OK on success, error code otherwise
 * 
 * @note The sum of pnCols and pnLPCols equals the column dimension of matrix A,
 * which is also the dimension of x and c in the optimization problem.
 */
lorads_retcode LReadSDPA(
    char *fname, // filename
    lorads_int *pnConstrs, // constralorads_int number, row number of matrix A, the dimension of b
    lorads_int *pnBlks, // block number for semidefinite variable
    lorads_int **pblkDims, // block dimension for semidefinite variable
    double **prowRHS, // dual coefficient, b in Ax=b
    lorads_int ***pconeMatBeg, // cone matrix begin idx
    lorads_int ***pconeMatIdx, // cone matrix idx
    double ***pconeMatElem, // cone matrix element
    lorads_int *pnCols, // dimension of all variables other than LP variables
    lorads_int *pnLPCols, // dimension of LP variables
    // pnCols + pnLPCols = column number of matrix A = the dimension of x = the dimension of c
    lorads_int **pLpMatBeg,
    lorads_int **pLpMatIdx,
    double **pLpMatElem,
    lorads_int *pnElems // number of elements in A other than LP variables
    );

#endif /* LORADS_FILE_READER_H */

#endif /* LORADS_FILE_IO_H */
