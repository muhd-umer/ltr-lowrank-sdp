/**
 * @file def_lorads_lbfgs.h
 * @brief Definition of L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) data structures
 * @details This header file defines the data structures used in the L-BFGS algorithm
 * implementation for the LORADS solver, specifically the node structure for storing
 * optimization history.
 */

#ifndef DEF_LORADS_LBFGS
#define DEF_LORADS_LBFGS

/**
 * @brief Internal structure for L-BFGS history node
 * @details This structure stores the information needed for the L-BFGS algorithm,
 * including the difference vectors and scaling factors used in the quasi-Newton update.
 */
struct lbfgs_node_internal
{
    lorads_int allElem;  ///< Total number of elements in the vectors

    double *s;    ///< Difference of R vectors (Rk - Rk-1)
    double *y;    ///< Difference of gradient vectors
    double beta;  ///< Reciprocal of inner product <y, s>
    double alpha; ///< Scaling factor beta * <s, q>
    struct lbfgs_node_internal *next;  ///< Pointer to next node in history
    struct lbfgs_node_internal *prev;  ///< Pointer to previous node in history
};

/**
 * @brief Type definition for L-BFGS history node
 * @details This typedef provides a convenient name for the L-BFGS node structure
 * used throughout the codebase.
 */
typedef struct lbfgs_node_internal lbfgs_node;

#endif