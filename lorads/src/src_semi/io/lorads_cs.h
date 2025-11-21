/**
 * @file lorads_cs.h
 * @brief LORADS CSparse Interface Definitions
 * @details This header file defines the interface for sparse matrix operations in LORADS,
 * based on the CXSparse/CSparse library. It provides data structures and functions for
 * handling sparse matrices in both triplet and compressed column formats.
 * 
 * @note This is based on CXSparse/Include/cs.h, which is a superset of CSparse.
 * The interface maintains compatibility with CSparse while providing additional features
 * like support for complex matrices and both lorads_int and long versions.
 * 
 * @author LORADS Team
 * @date 2024
 */

/* ========================================================================== */
/* CXSparse/Include/cs.h file */
/* ========================================================================== */

/* This is the CXSparse/Include/cs.h file.  It has the same name (cs.h) as
   the CSparse/Include/cs.h file.  The 'make install' for SuiteSparse installs
   CXSparse, and this file, instead of CSparse.  The two packages have the same
   cs.h include filename, because CXSparse is a superset of CSparse.  Any user
   program that uses CSparse can rely on CXSparse instead, with no change to the
   user code.  The #include "cs.h" line will work for both versions, in user
   code, and the function names and user-visible typedefs from CSparse all
   appear in CXSparse.  For experimenting and changing the package itself, I
   recommend using CSparse since it's simpler and easier to modify.  For
   using the package in production codes, I recommend CXSparse since it has
   more features (support for complex matrices, and both lorads_int and long
   versions).
 */

/* ========================================================================== */

#ifndef _LORADS_CS_
#define _LORADS_CS_

#ifdef __cplusplus
extern "C" {
#endif
#include "lorads.h"

/**
 * @struct lorads_cs_sparse
 * @brief Sparse matrix structure in compressed column or triplet format
 * @details This structure represents a sparse matrix in either compressed column
 * format (CSC) or triplet format. The format is determined by the nz field.
 */
typedef struct lorads_cs_sparse {
    /** @brief Maximum number of entries that can be stored */
    lorads_int nzmax;
    
    /** @brief Number of rows in the matrix */
    lorads_int m;
    
    /** @brief Number of columns in the matrix */
    lorads_int n;
    
    /** @brief Column pointers (size n+1) or column indices (size nzmax) */
    lorads_int *p;
    
    /** @brief Row indices, size nzmax */
    lorads_int *i;
    
    /** @brief Numerical values, size nzmax */
    double *x;
    
    /** @brief Number of entries in triplet matrix, -1 for compressed-column format */
    lorads_int nz;
} dcs ; //dcs alias

/**
 * @brief Add an entry to a triplet matrix
 * @param T Pointer to triplet matrix
 * @param i Row index
 * @param j Column index
 * @param x Value to add
 * @return 1 if successful, 0 otherwise
 */
int dcs_entry (dcs *T, lorads_int i, lorads_int j, double x) ;

/**
 * @brief Convert a triplet matrix to compressed column format
 * @param T Pointer to triplet matrix
 * @return Pointer to compressed column matrix or NULL if conversion fails
 */
dcs *dcs_compress (const dcs *T) ;

/**
 * @brief Compute the 1-norm of a sparse matrix
 * @param A Pointer to sparse matrix
 * @return Matrix 1-norm or -1 if input is invalid
 */
double dcs_norm (const dcs *A) ;

/**
 * @brief Print a sparse matrix
 * @param A Pointer to sparse matrix
 * @param brief Whether to print in brief format
 * @return 1 if successful, 0 otherwise
 */
int dcs_print (const dcs *A, int brief) ;

/* utilities */

/**
 * @brief Allocate and initialize memory
 * @param n Number of elements
 * @param size Size of each element
 * @return Pointer to allocated and zeroed memory
 */
void *dcs_calloc (lorads_int n, size_t size) ;

/**
 * @brief Free allocated memory
 * @param p Pointer to memory to free
 * @return NULL
 */
void *dcs_free (void *p) ;

/**
 * @brief Reallocate memory
 * @param p Pointer to existing memory
 * @param n New number of elements
 * @param size Size of each element
 * @param ok Pointer to store success status
 * @return Pointer to reallocated memory
 */
void *dcs_realloc (void *p, lorads_int n, size_t size, int *ok) ;

/**
 * @brief Allocate a sparse matrix
 * @param m Number of rows
 * @param n Number of columns
 * @param nzmax Maximum number of entries
 * @param values Whether to allocate space for values
 * @param t Whether to create a triplet matrix
 * @return Pointer to allocated sparse matrix
 */
dcs *dcs_spalloc (lorads_int m, lorads_int n, lorads_int nzmax, lorads_int values, lorads_int t) ;

/**
 * @brief Free a sparse matrix
 * @param A Pointer to sparse matrix to free
 * @return NULL
 */
dcs *dcs_spfree (dcs *A) ;

/**
 * @brief Reallocate a sparse matrix to hold more entries
 * @param A Pointer to sparse matrix
 * @param nzmax New maximum number of entries
 * @return 1 if successful, 0 otherwise
 */
int dcs_sprealloc (dcs *A, lorads_int nzmax) ;

/**
 * @brief Allocate memory
 * @param n Number of elements
 * @param size Size of each element
 * @return Pointer to allocated memory
 */
void *dcs_malloc (lorads_int n, size_t size) ;

/* utilities */

/**
 * @brief Compute cumulative sum of an array
 * @param p Output array for cumulative sums
 * @param c Input array of counts
 * @param n Length of arrays
 * @return Sum of all elements in c
 */
double dcs_cumsum (lorads_int *p, lorads_int *c, lorads_int n) ;

/**
 * @brief Free workspace and return sparse matrix result
 * @param C Pointer to sparse matrix result
 * @param w Workspace pointer to free
 * @param x Additional workspace pointer to free
 * @param ok Success status
 * @return Pointer to result matrix or NULL if operation failed
 */
dcs *dcs_done (dcs *C, void *w, void *x, int ok) ;

/**
 * @brief Free workspace and return integer array result
 * @param p Pointer to integer array result
 * @param C Pointer to sparse matrix
 * @param w Workspace pointer to free
 * @param ok Success status
 * @return Pointer to result array or NULL if operation failed
 */
lorads_int *dcs_idone (lorads_int *p, dcs *C, void *w, int ok) ;

/**
 * @brief Check if matrix is in compressed column format
 * @param A Pointer to sparse matrix
 */
#define IS_CSC(A) (A && (A->nz == -1))

/**
 * @brief Check if matrix is in triplet format
 * @param A Pointer to sparse matrix
 */
#define IS_TRIPLET(A) (A && (A->nz >= 0))

#ifdef __cplusplus
}
#endif

#endif