/**
 * @file lorads_cs.c
 * @brief LORADS CSparse Implementation
 * @details This file implements CSparse routines for handling sparse matrices in LORADS.
 * The implementation is based on Tim Davis's SuiteSparse library and provides:
 * - Memory management for sparse matrices
 * - Sparse matrix creation and manipulation
 * - Matrix compression and conversion
 * - Matrix norm calculations
 * - Matrix printing utilities
 * 
 * @author LORADS Team
 * @date 2024
 */

#include "lorads_utils.h"
#include "lorads_cs.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief Allocate memory for CSparse operations
 * @param n Number of elements to allocate
 * @param size Size of each element in bytes
 * @return Pointer to allocated memory
 * @details Wrapper for malloc with size validation
 */
void *dcs_malloc (lorads_int n, size_t size) {
    
    return (malloc (LORADS_MAX (n, 1) * size)) ;
}

/**
 * @brief Allocate and initialize memory for CSparse operations
 * @param n Number of elements to allocate
 * @param size Size of each element in bytes
 * @return Pointer to allocated and zeroed memory
 * @details Wrapper for calloc with size validation
 */
void *dcs_calloc (lorads_int n, size_t size) {
    
    return (calloc (LORADS_MAX (n,1), size)) ;
}

/**
 * @brief Free memory allocated for CSparse operations
 * @param p Pointer to memory to free
 * @return NULL
 * @details Wrapper for free that handles NULL pointers
 */
void *dcs_free (void *p) {
    
    if (p) free (p) ;       /* free p if it is not already NULL */
    return (NULL) ;         /* return NULL to simplify the use of ddcs_free */
}

/**
 * @brief Reallocate memory for CSparse operations
 * @param p Pointer to existing memory
 * @param n New number of elements
 * @param size Size of each element in bytes
 * @param ok Pointer to store success status
 * @return Pointer to reallocated memory or original pointer if reallocation fails
 * @details Wrapper for realloc with size validation and error checking
 */
void *dcs_realloc (void *p, lorads_int n, size_t size, int *ok) {
    
    void *pnew ;
    pnew = realloc (p, LORADS_MAX (n,1) * size) ; /* realloc the block */
    *ok = (pnew != NULL) ;                  /* realloc fails if pnew is NULL */
    return ((*ok) ? pnew : p) ;             /* return original p if failure */
}

/**
 * @brief Allocate a sparse matrix
 * @param m Number of rows
 * @param n Number of columns
 * @param nzmax Maximum number of entries
 * @param values Whether to allocate space for values
 * @param triplet Whether to create a triplet matrix
 * @return Pointer to allocated sparse matrix or NULL if allocation fails
 * @details Creates either a triplet or compressed column format matrix
 */
dcs *dcs_spalloc (lorads_int m, lorads_int n, lorads_int nzmax, lorads_int values, lorads_int triplet) {
    
    dcs *A = dcs_calloc (1, sizeof (dcs)) ;    /* allocate the dcs struct */
    if (!A) return (NULL) ;                 /* out of memory */
    A->m = m ;                              /* define dimensions and nzmax */
    A->n = n ;
    A->nzmax = nzmax = LORADS_MAX (nzmax, 1) ;
    A->nz = triplet ? 0 : -1 ;              /* allocate triplet or comp.col */
    A->p = dcs_malloc (triplet ? nzmax : n+1, sizeof (lorads_int)) ;
    A->i = dcs_malloc (nzmax, sizeof (lorads_int)) ;
    A->x = values ? dcs_malloc (nzmax, sizeof (double)) : NULL ;
    return ((!A->p || !A->i || (values && !A->x)) ? dcs_spfree (A) : A) ;
}

/**
 * @brief Reallocate a sparse matrix to hold more entries
 * @param A Pointer to sparse matrix
 * @param nzmax New maximum number of entries
 * @return 1 if successful, 0 otherwise
 * @details Increases the capacity of the matrix while preserving its contents
 */
int dcs_sprealloc (dcs *A, lorads_int nzmax) {
    
    int ok, oki, okj = 1, okx = 1 ;
    if (!A) return (0) ;
    if (nzmax <= 0) nzmax = IS_CSC(A) ? (A->p [A->n]) : A->nz ;
    nzmax = LORADS_MAX (nzmax, 1) ;
    A->i = dcs_realloc (A->i, nzmax, sizeof (lorads_int), &oki) ;
    if (IS_TRIPLET (A)) A->p = dcs_realloc (A->p, nzmax, sizeof (lorads_int), &okj) ;
    if (A->x) A->x = dcs_realloc (A->x, nzmax, sizeof (double), &okx) ;
    ok = (oki && okj && okx) ;
    if (ok) A->nzmax = nzmax ;
    return (ok) ;
}

/**
 * @brief Free a sparse matrix
 * @param A Pointer to sparse matrix to free
 * @return NULL
 * @details Frees all memory associated with the sparse matrix
 */
dcs *dcs_spfree (dcs *A) {
    
    if (!A) return (NULL) ;     /* do nothing if A already NULL */
    dcs_free (A->p) ;
    dcs_free (A->i) ;
    dcs_free (A->x) ;
    return ((dcs *) dcs_free (A)) ;   /* free the dcs struct and return NULL */
}

/**
 * @brief Free workspace and return sparse matrix result
 * @param C Pointer to sparse matrix result
 * @param w Workspace pointer to free
 * @param x Additional workspace pointer to free
 * @param ok Success status
 * @return Pointer to result matrix or NULL if operation failed
 * @details Helper function to clean up workspace and handle result
 */
dcs *dcs_done (dcs *C, void *w, void *x, int ok) {
    
    dcs_free (w) ;                       /* free workspace */
    dcs_free (x) ;
    return (ok ? C : dcs_spfree (C)) ;   /* return result if OK, else free it */
}

/**
 * @brief Compute cumulative sum of an array
 * @param p Output array for cumulative sums
 * @param c Input array of counts
 * @param n Length of arrays
 * @return Sum of all elements in c
 * @details Computes p[i] = sum(c[0..i-1]) and copies p[0..n-1] back into c[0..n-1]
 */
double dcs_cumsum (lorads_int *p, lorads_int *c, lorads_int n) {
    
    lorads_int i, nz = 0 ;
    double nz2 = 0 ;
    if (!p || !c) return (-1) ;     /* check inputs */
    for (i = 0 ; i < n ; i++) {
        p [i] = nz ;
        nz += c [i] ;
        nz2 += c [i] ;              /* also in double to avoid lorads_int overflow */
        c [i] = p [i] ;             /* also copy p[0..n-1] back into c[0..n-1]*/
    }
    p [n] = nz ;
    return (nz2) ;                  /* return sum (c [0..n-1]) */
}

/**
 * @brief Add an entry to a triplet matrix
 * @param T Pointer to triplet matrix
 * @param i Row index
 * @param j Column index
 * @param x Value to add
 * @return 1 if successful, 0 otherwise
 * @details Adds a new entry to the triplet matrix, reallocating if necessary
 */
int dcs_entry (dcs *T, lorads_int i, lorads_int j, double x) {
    
    if (!IS_TRIPLET (T) || i < 0 || j < 0) return (0) ;     /* check inputs */
    if (T->nz >= T->nzmax && !dcs_sprealloc (T,2*(T->nzmax))){
        printf("wrong\n");
        return (0);
    }
    if (T->x) T->x [T->nz] = x ;
    T->i [T->nz] = i ;
    T->p [T->nz++] = j ;
    T->m = LORADS_MAX (T->m, i+1) ;
    T->n = LORADS_MAX (T->n, j+1) ;
    return (1) ;
}

/**
 * @brief Convert a triplet matrix to compressed column format
 * @param T Pointer to triplet matrix
 * @return Pointer to compressed column matrix or NULL if conversion fails
 * @details Converts a triplet matrix to compressed column format, sorting entries by column
 */
dcs *dcs_compress (const dcs *T) {
    
    lorads_int m, n, nz, p, k, *Cp, *Ci, *w, *Ti, *Tj ;
    double *Cx, *Tx ;
    dcs *C ;
    if (!IS_TRIPLET (T)) return (NULL) ;                /* check inputs */
    m = T->m ; n = T->n ; Ti = T->i ; Tj = T->p ; Tx = T->x ; nz = T->nz ;
    C = dcs_spalloc (m, n, nz, Tx != NULL, 0) ;          /* allocate result */
    w = dcs_calloc (n, sizeof (lorads_int)) ;                   /* get workspace */
    if (!C || !w) return (dcs_done (C, w, NULL, 0)) ;    /* out of memory */
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (k = 0 ; k < nz ; k++) w [Tj [k]]++ ;           /* column counts */
    dcs_cumsum (Cp, w, n) ;                              /* column pointers */
    for (k = 0 ; k < nz ; k++) {
        Ci [p = w [Tj [k]]++] = Ti [k] ;    /* A(i,j) is the pth entry in C */
        if (Cx) Cx [p] = Tx [k] ;
    }
    return (dcs_done (C, w, NULL, 1)) ;      /* success; free w and return C */
}

/**
 * @brief Compute the 1-norm of a sparse matrix
 * @param A Pointer to sparse matrix
 * @return Matrix 1-norm or -1 if input is invalid
 * @details Computes the maximum column sum of absolute values
 */
double dcs_norm (const dcs *A) {
    
    lorads_int p, j, n, *Ap ;
    double *Ax ;
    double nrm = 0, s ;
    if (!IS_CSC (A) || !A->x) return (-1) ;             /* check inputs */
    n = A->n ; Ap = A->p ; Ax = A->x ;
    for (j = 0 ; j < n ; j++) {
        for (s = 0, p = Ap [j] ; p < Ap [j+1] ; p++) s += fabs (Ax [p]) ;
        nrm = LORADS_MAX (nrm, s) ;
    }
    return (nrm) ;
}

/**
 * @brief Print a sparse matrix
 * @param A Pointer to sparse matrix
 * @param brief Whether to print in brief format
 * @return 1 if successful, 0 otherwise
 * @details Prints matrix dimensions, number of entries, and values
 */
int dcs_print (const dcs *A, int brief) {
    
    lorads_int p, j, m, n, nzmax, nz, *Ap, *Ai ;
    double *Ax ;
    if (!A) { printf ("(null)\n") ; return (0) ; }
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    nzmax = A->nzmax ; nz = A->nz ;
    if (nz < 0) {
        printf ("%g-by-%g, nzmax: %g nnz: %g, 1-norm: %g\n", (double) m,
            (double) n, (double) nzmax, (double) (Ap [n]), dcs_norm (A)) ;
        for (j = 0 ; j < n ; j++) {
            printf ("    col %g : locations %g to %g\n", (double) j,
                (double) (Ap [j]), (double) (Ap [j+1]-1)) ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++) {
                printf ("      %g : ", (double) (Ai [p])) ;
                printf ("%50.50e \n", Ax ? Ax [p] : 1) ;
                if (brief && p > 20) { printf ("  ...\n") ; return (1) ; }
            }
        }
    }
    else {
        printf ("triplet: %g-by-%g, nzmax: %g nnz: %g\n", (double) m,
            (double) n, (double) nzmax, (double) nz) ;
        for (p = 0 ; p < nz ; p++) {
            printf ("    %g %g : ", (double) (Ai [p]), (double) (Ap [p])) ;
            printf ("%g\n", Ax ? Ax [p] : 1) ;
            if (brief && p > 20) { printf ("  ...\n") ; return (1) ; }
        }
    }
    return (1) ;
}
