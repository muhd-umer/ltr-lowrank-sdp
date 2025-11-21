/**
 * @file lorads_dense_opts.c
 * @brief LORADS Dense Matrix Operations Implementation
 * @details This file implements various dense matrix operations for LORADS,
 * including symmetric matrix operations, eigenvalue computations, and matrix-vector
 * multiplications. It provides both full dense and packed dense matrix operations.
 * 
 * @author LORADS Team
 * @date 2024
 */

#include "lorads_utils.h"
#include "lorads_dense_opts.h"
#include "lorads_vec_opts.h"
#include "def_lorads_sdp_data.h"

#include <math.h>


#ifdef MEMDEBUG
#include "memwatch.h"
#endif

/**
 * @brief BLAS layout options
 * @details Defines the matrix storage layout options for BLAS operations
 */
enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};

/**
 * @brief BLAS transpose options
 * @details Defines the matrix transpose options for BLAS operations
 */
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

/**
 * @brief BLAS upper/lower triangle options
 * @details Defines the matrix triangle options for BLAS operations
 */
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};

/**
 * @brief BLAS diagonal options
 * @details Defines the diagonal options for BLAS operations
 */
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};

/**
 * @brief BLAS side options
 * @details Defines the side options for BLAS operations
 */
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/**
 * @brief BLAS storage options
 * @details Defines the storage options for BLAS operations
 */
enum CBLAS_STORAGE {CblasPacked=151};

/**
 * @brief BLAS matrix identifier options
 * @details Defines the matrix identifier options for BLAS operations
 */
enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162};

/**
 * @brief BLAS offset options
 * @details Defines the offset options for BLAS operations
 */
enum CBLAS_OFFSET {CblasRowOffset=171, CblasColOffset=172, CblasFixOffset=173};

typedef enum CBLAS_LAYOUT CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO CBLAS_UPLO;
typedef enum CBLAS_DIAG CBLAS_DIAG;
typedef enum CBLAS_SIDE CBLAS_SIDE;
typedef enum CBLAS_STORAGE CBLAS_STORAGE;
typedef enum CBLAS_IDENTIFIER CBLAS_IDENTIFIER;
typedef enum CBLAS_OFFSET CBLAS_OFFSET;

#ifdef UNDER_BLAS
extern void dsymv_( const char *uplo, lorads_int *n, const double *alpha,
                   const double *a, lorads_int *lda, const double *x,
                   lorads_int *incx, const double *beta, double *y, lorads_int *incy );



extern void dsyevr_( const char *jobz, const char *range, const char *uplo,
                    lorads_int  *n, double *a, lorads_int *lda,
                    const double *vl, const double *vu, lorads_int *il,
                    lorads_int *iu, const double *abstol, lorads_int *m,
                    double *w, double *z, lorads_int *ldz, lorads_int *isuppz,
                    double *work, lorads_int *lwork, lorads_int *iwork, lorads_int *liwork,
                    lorads_int *info );

extern void dspmv_( const char *uplo, lorads_int *n, const double *alpha,
                   const double *ap, const double *x, lorads_int *incx,
                   const double *beta, double *y, lorads_int *incy );

extern void dsymm_( const char *side, const char *uplo, lorads_int *m,
                   lorads_int *n, const double *alpha, const double *a,
                   lorads_int *lda, const double *b, lorads_int *ldb,
                   const double *beta, double *c, lorads_int *ldc );

extern void dsyr_( char *uplo, lorads_int *n, double *alpha,
           double *x, lorads_int *incx, double *a, lorads_int *lda );

extern void dsyr2k_(const char *uplo, const char *trans, lorads_int *n, lorads_int *k, const double *alpha, const double *a, lorads_int *lda, const double *b, lorads_int *ldb, const double *beta,  double *c, lorads_int *ldc);


extern void dger_( lorads_int *m, lorads_int *n, const double *alpha,
                  const double *x, lorads_int *incx, const double *y,
                  lorads_int *incy, double *a, lorads_int *lda );
#else

extern void dsymv( const char *uplo, lorads_int *n, const double *alpha,
                   const double *a, lorads_int *lda, const double *x,
                   lorads_int *incx, const double *beta, double *y, lorads_int *incy );



extern void dsyevr( const char *jobz, const char *range, const char *uplo,
                    lorads_int  *n, double *a, lorads_int *lda,
                    const double *vl, const double *vu, lorads_int *il,
                    lorads_int *iu, const double *abstol, lorads_int *m,
                    double *w, double *z, lorads_int *ldz, lorads_int *isuppz,
                    double *work, lorads_int *lwork, lorads_int *iwork, lorads_int *liwork,
                    lorads_int *info );

extern void dspmv( const char *uplo, lorads_int *n, const double *alpha,
                   const double *ap, const double *x, lorads_int *incx,
                   const double *beta, double *y, lorads_int *incy );

extern void dsymm( const char *side, const char *uplo, lorads_int *m,
                   lorads_int *n, const double *alpha, const double *a,
                   lorads_int *lda, const double *b, lorads_int *ldb,
                   const double *beta, double *c, lorads_int *ldc );

void dsyr2k(const char *uplo, const char *trans, lorads_int *n, lorads_int *k, const double *alpha, const double *a, lorads_int *lda, const double *b, lorads_int *ldb, const double *beta,  double *c, lorads_int *ldc);


extern void dger( lorads_int *m, lorads_int *n, const double *alpha,
                  const double *x, lorads_int *incx, const double *y,
                  lorads_int *incy, double *a, lorads_int *lda );
#endif

/**
 * @brief Perform symmetric matrix-vector multiplication
 * @param n Dimension of the matrix
 * @param alpha Scalar multiplier
 * @param A Symmetric matrix in dense format
 * @param x Input vector
 * @param beta Scalar multiplier for y
 * @param y Output vector
 * @details Computes y = alpha * A * x + beta * y where A is symmetric
 */
extern void fds_symv( lorads_int n, double alpha, double *A, double *x, double beta, double *y ) {
    // symmetry matrix multiplication
    // y = alpha * A * x + beta * y
    // where A is a symmetric matrix in dense format
    // n is the dimension of the matrix
    // x and y are vectors
    // alpha and beta are scalars
    // ACharConstantUploLow UPLOLOW low triangular part of A is referenced
#ifdef UNDER_BLAS
    dsymv_(&ACharConstantUploLow, &n, &alpha, A, &n, x, &AIntConstantOne, &beta, y, &AIntConstantOne);
#else
    dsymv(&ACharConstantUploLow, &n, &alpha, A, &n, x, &AIntConstantOne, &beta, y, &AIntConstantOne);
#endif

    return;
}

/**
 * @brief Perform symmetric packed matrix-vector multiplication
 * @param n Dimension of the matrix
 * @param alpha Scalar multiplier
 * @param A Symmetric packed matrix
 * @param x Input vector
 * @param beta Scalar multiplier for y
 * @param y Output vector
 * @details Computes y = alpha * A * x + beta * y where A is symmetric packed
 */
extern void fds_symv_L(lorads_int n, double alpha, double *A, double *x, double beta, double *y){
    // dspmv y := alpha*A*x + beta*y,
#ifdef UNDER_BLAS
    dspmv_(&ACharConstantUploLow, &n, &alpha, A, x, &AIntConstantOne, &beta, y, &AIntConstantOne);
#else
    dspmv(&ACharConstantUploLow, &n, &alpha, A, x, &AIntConstantOne, &beta, y, &AIntConstantOne);
#endif
}


/**
 * @brief Compute minimum eigenvalue of a symmetric matrix
 * @param slackVar Pointer to SDP coefficient structure
 * @param minEig Pointer to store minimum eigenvalue
 * @details Computes the minimum eigenvalue of a symmetric matrix using LAPACK's dsyevr
 */
extern void fds_syev_Min_eig(sdp_coeff *slackVar, double *minEig){
    lorads_int n = slackVar->nSDPCol;
    double *completeSymMatrix;

    LORADS_INIT(completeSymMatrix, double, n * n);

    // need to check
    lorads_int retcode = LORADS_RETCODE_OK;
    char jobz = 'N'; // eigenvalues only
    char range = 'I'; // index range
    char uplo = ACharConstantUploLow;
    lorads_int il = 1;
    lorads_int iu = 1; // calculate minimal eigenvalue
    lorads_int lda = n, ldz = n;
    lorads_int m = 1;
    lorads_int info = 0;
    double *w;
    LORADS_INIT(w, double, n);
    double abstol = 1e-8;
    double *z;
    LORADS_INIT(z, double, n * n);
    lorads_int *isuppz;
    LORADS_INIT(isuppz, lorads_int, 2 * n);


    lorads_int lwork = -1;
    lorads_int liwork = -1;
    double work_query;
    lorads_int iwork_query;

#ifdef UNDER_BLAS
    dsyevr_(&jobz, &range, &uplo, &n, completeSymMatrix, &lda,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &abstol, &m, w, z,
           &ldz, isuppz, &work_query, &lwork, &iwork_query, &liwork, &info);
#else
    dsyevr(&jobz, &range, &uplo, &n, completeSymMatrix, &lda,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &abstol, &m, w, z,
           &ldz, isuppz, &work_query, &lwork, &iwork_query, &liwork, &info);
#endif

    lwork = (lorads_int)work_query;
    double *work = (double*)malloc(sizeof(double) * lwork);

    liwork = iwork_query;
    lorads_int *iwork = (lorads_int*)malloc(sizeof(lorads_int) * liwork);

#ifdef UNDER_BLAS
    dsyevr_(&jobz, &range, &uplo, &n, completeSymMatrix, &lda,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &abstol, &m, w, z,
           &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
#else
    dsyevr(&jobz, &range, &uplo, &n, completeSymMatrix, &lda,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &abstol, &m, w, z,
           &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
#endif


    if ( info != 0 ) {
        LORADS_ERROR_TRACE;
    }else{
        minEig[0] = w[0];
    }
    LORADS_FREE(isuppz);
    LORADS_FREE(z);
    LORADS_FREE(w);
    LORADS_FREE(work);
    LORADS_FREE(iwork);
    LORADS_FREE(completeSymMatrix);
}



/**
 * @brief Compute eigenvalues and eigenvectors of a symmetric matrix
 * @param n Dimension of the matrix
 * @param U Input/output matrix
 * @param d Array to store eigenvalues
 * @param Y Array to store eigenvectors
 * @param work Workspace array
 * @param iwork Integer workspace array
 * @param lwork Size of work array
 * @param liwork Size of iwork array
 * @return Info code from LAPACK
 * @details Computes eigenvalues and eigenvectors of a symmetric matrix using LAPACK's dsyevr
 */
extern lorads_int fds_syev( lorads_int n, double *U, double *d, double *Y,
                     double *work, lorads_int *iwork, lorads_int lwork, lorads_int liwork ) {
    
    char jobz = 'V', range = 'I', uplo = ACharConstantUploUp;
    lorads_int isuppz[4] = {0};
    lorads_int il = n - 1, iu = n;
    lorads_int m = 2;
    lorads_int info = 0;

    // compute eigenvalues and eigenvectors of a symmetric matrix
#ifdef UNDER_BLAS
    dsyevr_(&jobz, &range, &uplo, &n, U, &n,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &AblConstantZero, &m, d, Y,
           &n, isuppz, work, &lwork, iwork, &liwork, &info);
#else
    dsyevr(&jobz, &range, &uplo, &n, U, &n,
           &AblConstantZero, &AblConstantZero,
           &il, &iu, &AblConstantZero, &m, d, Y,
           &n, isuppz, work, &lwork, iwork, &liwork, &info);
#endif

    if ( info != 0 ) {
        LORADS_ERROR_TRACE;
    }
    return info;
}





/**
 * @brief Perform general matrix-vector multiplication
 * @param m Number of rows in matrix M
 * @param n Number of columns in matrix M
 * @param M Input matrix
 * @param v Input vector
 * @param y Output vector
 * @details Computes y = M * v (no transpose)
 */
extern void fds_gemv( lorads_int m, lorads_int n, double *M, double *v, double *y ) {
    // y = alpha * A * x + beta * y (beta = 0)

    // trans: 'N'--no transpose, 'T'--transpose, 'C'--conjugate transpose
    // m: number of rows of matrix M
    // n: number of columns of matrix M
    // alpha: scalar
    // a: matrix A
    // lda: leading dimension of a (column major means lda = m)
    // x: vector x
    // incx: the increment for the elements of x
    // beta: scalar
    // y: vector y
    // incy: the increment for the elements of y

#ifdef UNDER_BLAS
    dgemv_(&ACharConstantNoTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#else
    dgemv(&ACharConstantNoTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#endif

    return;
}


/**
 * @brief Perform transposed matrix-vector multiplication
 * @param m Number of rows in matrix M
 * @param n Number of columns in matrix M
 * @param M Input matrix
 * @param v Input vector
 * @param y Output vector
 * @details Computes y = M^T * v
 */
extern void fds_gemvTran( lorads_int m, lorads_int n, double *M, double *v, double *y ) {
    // y = alpha * A * x + beta * y (beta = 0)

    // trans: 'N'--no transpose, 'T'--transpose, 'C'--conjugate transpose
    // m: number of rows of matrix M
    // n: number of columns of matrix M
    // alpha: scalar
    // a: matrix A
    // lda: leading dimension of a (column major means lda = m)
    // x: vector x
    // incx: the increment for the elements of x
    // beta: scalar
    // y: vector y
    // incy: the increment for the elements of y

#ifdef UNDER_BLAS
    dgemv_(&ACharConstantTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#else
    dgemv(&ACharConstantTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#endif

    return;
}

/**
 * @brief Perform matrix-vector multiplication with beta scaling
 * @param m Number of rows in matrix M
 * @param n Number of columns in matrix M
 * @param M Input matrix
 * @param v Input vector
 * @param y Output vector
 * @param beta Scalar multiplier for y
 * @details Computes y = M * v + beta * y
 */
extern void fds_gemv_beta( lorads_int m, lorads_int n, double *M, double *v, double *y , double beta) {
    // y = alpha * A * x + beta * y (beta =1.0)

    // trans: 'N'--no transpose, 'T'--transpose, 'C'--conjugate transpose
    // m: number of rows of matrix M
    // n: number of columns of matrix M
    // alpha: scalar
    // a: matrix A
    // lda: leading dimension of a (column major means lda = m)
    // x: vector x
    // incx: the increment for the elements of x
    // beta: scalar
    // y: vector y
    // incy: the increment for the elements of y

#ifdef UNDER_BLAS
    dgemv_(&ACharConstantNoTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &beta, y, &AIntConstantOne);
#else
    dgemv(&ACharConstantNoTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &beta, y, &AIntConstantOne);
#endif

    return;
}

/**
 * @brief Perform transposed matrix-vector multiplication
 * @param m Number of rows in matrix M
 * @param n Number of columns in matrix M
 * @param M Input matrix
 * @param v Input vector
 * @param y Output vector
 * @details Computes y = M^T * v
 */
extern void fds_gemv_Trans( lorads_int m, lorads_int n, double *M, double *v, double *y ) {
    // y = alpha * A * x + beta * y (beta = 0)

    // trans: 'N'--no transpose, 'T'--transpose, 'C'--conjugate transpose
    // m: number of rows of matrix M
    // n: number of columns of matrix M
    // alpha: scalar
    // a: matrix A
    // lda: leading dimension of a (column major means lda = m)
    // x: vector x
    // incx: the increment for the elements of x
    // beta: scalar
    // y: vector y
    // incy: the increment for the elements of y

#ifdef UNDER_BLAS
    dgemv_(&ACharConstantTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#else
    dgemv(&ACharConstantTrans, &m, &n, &AblConstantOne, M, &m,
          v, &AIntConstantOne, &AblConstantZero, y, &AIntConstantOne);
#endif

    return;
}

/**
 * @brief Perform symmetric matrix-matrix multiplication
 * @param side Side of multiplication ('L' or 'R')
 * @param uplo Upper/lower triangle specification
 * @param m Number of rows in result matrix
 * @param n Number of columns in result matrix
 * @param alpha Scalar multiplier
 * @param a Symmetric matrix A
 * @param lda Leading dimension of A
 * @param b Matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Result matrix C
 * @param ldc Leading dimension of C
 * @details Computes C = alpha * A * B + beta * C or C = alpha * B * A + beta * C
 */
extern void fds_symm( char side, char uplo, lorads_int m, lorads_int n, double alpha, double *a, lorads_int lda,
                      double *b, lorads_int ldb, double beta, double *c, lorads_int ldc ) {
    // matrix multiply another symmetry matrix
    // side: 'L'--left, C = alpha * A * B + beta * C, 'R'--right C = alpha * B * A + beta * C
    // uplo: symmetry matrix 'U'--upper, 'L'--lower
    // m: number of rows of matrix C
    // n: number of columns of matrix C
    // alpha: scalar
    // a: symmetry matrix A, shape is (m, m)
    // lda: leading dimension of a (column major means lda = m)
    // b: symmetry matrix B, shape is (m, n) if side is 'L', shape is (n, m) if side is 'R'
#ifdef UNDER_BLAS
    dsymm_(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
    dsymm(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
    return;
}

/**
 * @brief Perform rank-1 update of a matrix
 * @param m Number of rows in matrix A
 * @param n Number of columns in matrix A
 * @param alpha Scalar multiplier
 * @param x Input vector x
 * @param incx Increment for x
 * @param y Input vector y
 * @param incy Increment for y
 * @param a Matrix to update
 * @param lda Leading dimension of A
 * @details Computes A = alpha * x * y^T + A
 */
extern void fds_ger( lorads_int m, lorads_int n, double alpha, double *x, lorads_int incx,
                     double *y, lorads_int incy, double *a, lorads_int lda) {
#ifdef UNDER_BLAS
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#else
    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
#endif

    return;
}

/**
 * @brief Print a dense matrix
 * @param n Dimension of the matrix
 * @param A Matrix to print
 * @details Prints the matrix in a formatted way
 */
extern void fds_print( lorads_int n, double *A ) {

    for ( lorads_int i = 0; i < n; ++i ) {
        for ( lorads_int j = 0; j < n; ++j ) {
            printf("%6.3e ", A[i + n * j]);
        }
        printf("\n");
    }

    return;
}

/**
 * @brief Scale a packed dense matrix
 * @param a Scalar multiplier
 * @param n Dimension of the matrix
 * @param A Matrix to scale
 * @details Scales all elements of the packed matrix by a
 */
extern void pds_scal( double a, lorads_int n, double *A ) {

    lorads_int nnz = PACK_NNZ(n), incx = 1;
    scal(&nnz, &a, A, &incx);

    return;
}

/**
 * @brief Compute sum of absolute values of packed matrix elements
 * @param n Dimension of the matrix
 * @param A Packed matrix
 * @return Sum of absolute values
 * @details Computes the sum of absolute values of all elements in the packed matrix
 */
extern double pds_sum_abs( lorads_int n, double *A ) {

    double nrm = 0.0, *p = A;
    lorads_int nrow = n, incx = 1;

    for ( lorads_int i = 0; i < n; ++i ) {
        nrm -= 0.5 * fabs(p[0]);
        nrm += nrm1(&nrow, p, &incx);
        p += n - i; nrow -= 1;
    }

    return 2.0 * nrm;
}

/**
 * @brief Compute Frobenius norm of packed matrix
 * @param n Dimension of the matrix
 * @param A Packed matrix
 * @return Frobenius norm
 * @details Computes the Frobenius norm of the packed matrix
 */
extern double pds_fro_norm( lorads_int n, double *A ) {

    double nrm = 0.0, colnrm, *p = A;
    lorads_int nrow = n, incx = 1;

    for ( lorads_int i = 0; i < n; ++i ) {
        nrm -= 0.5 * p[0] * p[0];
        colnrm = nrm2(&nrow, p, &incx);
        nrm += colnrm * colnrm;
        p += n - i; nrow -= 1;
    }

    return sqrt(2.0 * nrm);
}

/**
 * @brief Convert packed matrix to full matrix
 * @param n Dimension of the matrix
 * @param A Packed matrix
 * @param v Full matrix output
 * @details Converts a packed symmetric matrix to full storage format
 */
extern void pds_dump( lorads_int n, double *A, double *v ) {

    for ( lorads_int i = 0, j; i < n; ++i ) {
        for ( j = i; j < n; ++j ) {
            FULL_ENTRY(v, n, i, j) = FULL_ENTRY(v, n, j, i) = PACK_ENTRY(A, n, i, j);
        }
    }

    return;
}

/**
 * @brief Decompress sparse matrix to dense format
 * @param nnz Number of non-zero elements
 * @param Ci Column indices
 * @param Cx Non-zero values
 * @param A Output dense matrix
 * @details Converts a sparse matrix in CSC format to dense format
 */
extern void pds_decompress( lorads_int nnz, lorads_int *Ci, double *Cx, double *A ) {
    // decompress general dense matrix
    for ( lorads_int k = 0; k < nnz; ++k ) {
        A[Ci[k]] = Cx[k];
    }
    return;
}

/**
 * @brief Check if matrix is rank one and extract vector
 * @param n Dimension of the matrix
 * @param A Packed matrix
 * @param sgn Sign of the rank-one matrix
 * @param a Output vector
 * @return 1 if matrix is rank one, 0 otherwise
 * @details Checks if the matrix is rank one and extracts the vector if it is
 */
extern lorads_int pds_r1_extract( lorads_int n, double *A, double *sgn, double *a ) {

    /* Get the first nonzero */
    lorads_int i, j, k = 0;
    for ( i = 0; i < n; ++i ) {
        if ( A[k] != 0 ) {
            break;
        }
        k += n - i;
    }

    if ( i == n ) {
        return 0;
    }

    double s = ( A[k] > 0 ) ? 1.0 : -1.0;
    double v = sqrt(fabs(A[k]));
    double eps = 0.0;

    /* Extract diagonal */
    for ( k = 0; k < n; ++k ) {
        a[k] = PACK_ENTRY(A, n, k, i) / v;
    }

    lorads_int id = 0;
    double *pstart = NULL;
    if ( s == 1.0 ) {
        for ( i = 0; i < n; ++i ) {
            pstart = A + id;
            for ( j = 0; j < n - i; ++j ) {
                eps += fabs(pstart[j] - a[i] * a[i + j]);
            }
            id += n - i;
            if ( eps > 1e-10 ) {
                return 0;
            }
        }
    } else {
        for ( i = 0; i < n; ++i ) {
            pstart = A + id;
            for ( j = 0; j < n - i; ++j ) {
                eps += fabs(pstart[j] + a[i] * a[i + j]);
            }
            id += n - i;
            if ( eps > 1e-10 ) {
                return 0;
            }
        }
    }

    *sgn = s;
    return 1;
}

/**
 * @brief Compute quadratic form with packed matrix
 * @param n Dimension of the matrix
 * @param A Packed matrix
 * @param v Input vector
 * @param aux Auxiliary vector
 * @return Value of quadratic form
 * @details Computes v^T * A * v where A is packed symmetric
 */
extern double pds_quadform( lorads_int n, double *A, double *v, double *aux ) {
#ifdef UNDER_BLAS
    dspmv_(&ACharConstantUploLow, &n, &AblConstantOne, A, v,
          &AIntConstantOne, &AblConstantZero, aux, &AIntConstantOne);
#else
    dspmv(&ACharConstantUploLow, &n, &AblConstantOne, A, v,
          &AIntConstantOne, &AblConstantZero, aux, &AIntConstantOne);
#endif

    return dot(&n, aux, &AIntConstantOne, v, &AIntConstantOne);
}

/**
 * @brief Perform symmetric packed matrix-vector multiplication
 * @param uplo Upper/lower triangle specification
 * @param n Dimension of the matrix
 * @param alpha Scalar multiplier
 * @param ap Packed matrix
 * @param x Input vector
 * @param incx Increment for x
 * @param beta Scalar multiplier for y
 * @param y Output vector
 * @param incy Increment for y
 * @details Computes y = alpha * A * x + beta * y where A is packed symmetric
 */
extern void pds_spmv( char uplo, lorads_int n, double alpha, double *ap, double *x, lorads_int incx,
                      double beta, double *y, lorads_int incy ) {
#ifdef UNDER_BLAS
    dspmv_(&uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy);
#else
    dspmv(&uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy);
#endif

    return;
}

/**
 * @brief Perform symmetric rank-1 update
 * @param uplo Upper/lower triangle specification
 * @param n Dimension of the matrix
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param a Matrix to update
 * @param lda Leading dimension of A
 * @details Computes A = alpha * x * x^T + A where A is symmetric
 */
extern void pds_syr( char uplo, lorads_int n, double alpha, double *x, lorads_int incx, double *a, lorads_int lda ) {
#ifdef UNDER_BLAS
    dsyr_(&uplo, &n, &alpha, x, &incx, a, &lda);
#else
    dsyr(&uplo, &n, &alpha, x, &incx, a, &lda);
#endif
    return;
}

/**
 * @brief Perform symmetric rank-2k update
 * @param uplo Upper/lower triangle specification
 * @param trans Transpose flag
 * @param n Dimension of the matrix
 * @param k Number of columns in A and B
 * @param alpha Scalar multiplier
 * @param a First input matrix
 * @param b Second input matrix
 * @param beta Scalar multiplier for C
 * @param c Output matrix
 * @details Computes C = alpha * (A * B^T + B * A^T) + beta * C
 */
extern void fds_syr2k(char uplo, char trans, lorads_int n, lorads_int k, double alpha, double *a, double *b, double beta, double *c){
    lorads_int lda = n;
    lorads_int ldb = n;
    lorads_int ldc = n;
#ifdef UNDER_BLAS
    // C := alpha*A**T*B + alpha*B**T*A + beta*C,
    dsyr2k_(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
    dsyr2k(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

/**
 * @brief Compute symmetric matrix from two matrices
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of columns in U and V
 * @param U First input matrix
 * @param V Second input matrix
 * @param fullMat Output matrix
 * @details Computes fullMat = 0.5 * (U * V^T + V * U^T)
 */
extern void gemm_syr2k(lorads_int m, lorads_int n, lorads_int k, double *U, double *V, double *fullMat){
    // fullMat = 0.5 * (U * V^T + V * U^T)
    double half = 0.5; double one = 1.0; double zero = 0.0;
#ifdef UNDER_BLAS
    dgemm_(&ACharConstantNoTrans, &ACharConstantTrans, &m, &n, &k, &half, U, &m, V, &m, &zero, fullMat, &m);
    dgemm_(&ACharConstantNoTrans, &ACharConstantTrans, &m, &n, &k, &half, V, &m, U, &m, &one, fullMat, &m);
#else
    dgemm(&ACharConstantNoTrans, &ACharConstantTrans, &m, &n, &k, &half, U, &m, V, &m, &zero, fullMat, &m);
    dgemm(&ACharConstantNoTrans, &ACharConstantTrans, &m, &n, &k, &half, V, &m, U, &m, &one, fullMat, &m);
#endif
}
