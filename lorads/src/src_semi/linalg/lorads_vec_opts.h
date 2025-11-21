/**
 * @file lorads_vec_opts.h
 * @brief LORADS Vector Operations Interface
 * @details This header file declares the interface for vector operations in LORADS,
 * including BLAS-like operations, vector norms, and matrix-vector operations.
 * It provides both standard and BLAS-specific function declarations.
 * 
 * @author LORADS Team
 * @date 2024
 */

#ifndef vec_opts_h
#define vec_opts_h

/**
 * @brief BLAS operation constants
 * @details Constants used for specifying matrix operations and storage formats
 */
#define TRANS   ('T')      ///< Transpose operation
#define NOTRANS ('N')      ///< No transpose operation
#define UPLOLOW ('L')      ///< Lower triangular storage
#define UPLOUP ('U')       ///< Upper triangular storage
#define SIDELEFT ('L')     ///< Left side operation
#define SIDERIGHT ('R')    ///< Right side operation

/**
 * @brief BLAS constants
 * @details Constants used in BLAS operations, should never be modified
 */
static char ACharConstantTrans = TRANS;        ///< Transpose character constant
static char ACharConstantNoTrans = NOTRANS;    ///< No transpose character constant
static char ACharConstantUploUp = UPLOUP;      ///< Upper triangular character constant
static char ACharConstantUploLow = UPLOLOW;    ///< Lower triangular character constant
static lorads_int  AIntConstantOne = 1;        ///< Integer constant one
static double AblConstantZero = 0.0;           ///< Double constant zero
static double AblConstantOne = 1.0;            ///< Double constant one
static double AblConstantMinusOne = -1.0;      ///< Double constant minus one

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute 1-norm of a vector
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return 1-norm value
 */
extern double nrm1( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief Compute 2-norm of a vector
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return 2-norm value
 */
extern double nrm2( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief Perform vector addition with scaling
 * @param n Vector length
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param y Input/output vector
 * @param incy Increment for y
 */
extern void axpy( lorads_int *n, double *alpha, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief Perform vector addition with two scalings
 * @param n Vector length
 * @param a First scalar multiplier
 * @param x First input vector
 * @param incx Increment for x
 * @param b Second scalar multiplier
 * @param y Second input vector
 * @param incy Increment for y
 */
extern void axpby( lorads_int *n, double *a, double *x, lorads_int *incx, double *b, double *y, lorads_int *incy );

/**
 * @brief Perform vector addition with two scalings into third vector
 * @param n Vector length
 * @param a First scalar multiplier
 * @param x First input vector
 * @param b Second scalar multiplier
 * @param y Second input vector
 * @param z Output vector
 */
extern void axpbyAddition( lorads_int *n, double *a, double *x, double *b, double *y, double *z);

/**
 * @brief Compute dot product of two vectors
 * @param n Vector length
 * @param x First input vector
 * @param incx Increment for x
 * @param y Second input vector
 * @param incy Increment for y
 * @return Dot product value
 */
extern double dot( lorads_int *n, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief Scale a vector
 * @param n Vector length
 * @param sa Scalar multiplier
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void scal( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief Reciprocal scale a vector
 * @param n Vector length
 * @param sa Scalar divisor
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void rscl( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief Perform symmetric rank-1 update
 * @param uplo Upper/lower triangle specification
 * @param n Matrix dimension
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param a Matrix to update
 * @param lda Leading dimension of A
 */
extern void syr( char *uplo, lorads_int *n, double *alpha, double *x, lorads_int *incx, double *a, lorads_int *lda );

/**
 * @brief Perform symmetric matrix-vector multiplication
 * @param uplo Upper/lower triangle specification
 * @param n Matrix dimension
 * @param alpha Scalar multiplier
 * @param a Input matrix
 * @param lda Leading dimension of A
 * @param x Input vector
 * @param incx Increment for x
 * @param beta Scalar multiplier
 * @param y Input/output vector
 * @param incy Increment for y
 */
extern void symv( char *uplo, lorads_int *n, double *alpha, double *a, lorads_int *lda, double *x,
                  lorads_int *incx, double *beta, double *y, const lorads_int *incy );

/**
 * @brief Compute sum of logarithms of determinants
 * @param n Vector length
 * @param x Input vector
 * @return Sum of logarithms
 */
extern double sumlogdet( lorads_int *n, double *x );

/**
 * @brief Element-wise vector scaling
 * @param n Vector length
 * @param s Scaling factors
 * @param x Input/output vector
 */
extern void vvscl( lorads_int *n, double *s, double *x );

/**
 * @brief Element-wise vector reciprocal scaling
 * @param n Vector length
 * @param s Scaling factors
 * @param x Input/output vector
 */
extern void vvrscl( lorads_int *n, double *s, double *x );

/**
 * @brief Normalize a vector
 * @param n Vector length
 * @param a Input/output vector
 * @return Original norm of the vector
 */
extern double normalize( lorads_int *n, double *a );

#ifdef __cplusplus
}
#endif

#endif /* vec_opts_h */

#include "lorads_utils.h"

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

#include <math.h>

/**
 * @brief BLAS function declarations
 * @details These are the underlying BLAS function declarations used by the interface
 */

#ifdef UNDER_BLAS
/**
 * @brief BLAS 2-norm computation
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return 2-norm value
 */
extern double dnrm2_( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief BLAS vector addition with scaling
 * @param n Vector length
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param y Input/output vector
 * @param incy Increment for y
 */
extern void daxpy_( lorads_int *n, double *alpha, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief BLAS dot product computation
 * @param n Vector length
 * @param x First input vector
 * @param incx Increment for x
 * @param y Second input vector
 * @param incy Increment for y
 * @return Dot product value
 */
extern double ddot_( lorads_int *n, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief BLAS vector scaling
 * @param n Vector length
 * @param sa Scalar multiplier
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void dscal_( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief BLAS vector reciprocal scaling
 * @param n Vector length
 * @param sa Scalar divisor
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void drscl_( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief BLAS symmetric rank-1 update
 * @param uplo Upper/lower triangle specification
 * @param n Matrix dimension
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param a Matrix to update
 * @param lda Leading dimension of A
 */
extern void dsyr_( char *uplo, lorads_int *n, double *alpha, double *x, lorads_int *incx, double *a, lorads_int *lda );

/**
 * @brief BLAS index of maximum absolute value
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return Index of maximum absolute value
 */
extern lorads_int idamax_( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief BLAS index of minimum absolute value
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return Index of minimum absolute value
 */
extern lorads_int idamin_( lorads_int *n, double *x, lorads_int *incx );
#else
/**
 * @brief Standard 2-norm computation
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return 2-norm value
 */
extern double dnrm2( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief Standard vector addition with scaling
 * @param n Vector length
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param y Input/output vector
 * @param incy Increment for y
 */
extern void daxpy( lorads_int *n, double *alpha, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief Standard dot product computation
 * @param n Vector length
 * @param x First input vector
 * @param incx Increment for x
 * @param y Second input vector
 * @param incy Increment for y
 * @return Dot product value
 */
extern double ddot( lorads_int *n, double *x, lorads_int *incx, double *y, lorads_int *incy );

/**
 * @brief Standard vector scaling
 * @param n Vector length
 * @param sa Scalar multiplier
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void dscal( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief Standard vector reciprocal scaling
 * @param n Vector length
 * @param sa Scalar divisor
 * @param sx Input/output vector
 * @param incx Increment for x
 */
extern void drscl( lorads_int *n, double *sa, double *sx, lorads_int *incx );

/**
 * @brief Standard symmetric rank-1 update
 * @param uplo Upper/lower triangle specification
 * @param n Matrix dimension
 * @param alpha Scalar multiplier
 * @param x Input vector
 * @param incx Increment for x
 * @param a Matrix to update
 * @param lda Leading dimension of A
 */
extern void dsyr( char *uplo, lorads_int *n, double *alpha, double *x, lorads_int *incx, double *a, lorads_int *lda );

/**
 * @brief Standard index of maximum absolute value
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return Index of maximum absolute value
 */
extern lorads_int idamax( lorads_int *n, double *x, lorads_int *incx );

/**
 * @brief Standard index of minimum absolute value
 * @param n Vector length
 * @param x Input vector
 * @param incx Increment for x
 * @return Index of minimum absolute value
 */
extern lorads_int idamin( lorads_int *n, double *x, lorads_int *incx );
#endif
