#include "lorads_logging.h"
#include "lorads_utils.h"
#include "lorads_vec_opts.h"
#include "lorads_dense_opts.h"

#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#ifdef UNDER_BLAS
extern void dsyevr_(const char *jobz, const char *range, const char *uplo,
                    lorads_int *n, double *a, lorads_int *lda,
                    const double *vl, const double *vu, lorads_int *il,
                    lorads_int *iu, const double *abstol, lorads_int *m,
                    double *w, double *z, lorads_int *ldz, lorads_int *isuppz,
                    double *work, lorads_int *lwork, lorads_int *iwork, lorads_int *liwork,
                    lorads_int *info);
#else
extern void dsyevr(const char *jobz, const char *range, const char *uplo,
                   lorads_int *n, double *a, lorads_int *lda,
                   const double *vl, const double *vu, lorads_int *il,
                   lorads_int *iu, const double *abstol, lorads_int *m,
                   double *w, double *z, lorads_int *ldz, lorads_int *isuppz,
                   double *work, lorads_int *lwork, lorads_int *iwork, lorads_int *liwork,
                   lorads_int *info);
#endif

static void set_problem_name(const char *fname, char *buffer, size_t buf_len)
{
    if (!fname || !buffer || buf_len == 0)
    {
        return;
    }
    buffer[0] = '\0';
    const char *last_slash = strrchr(fname, '/');
    const char *base = last_slash ? last_slash + 1 : fname;
    snprintf(buffer, buf_len, "%s", base);
    char *dot = strrchr(buffer, '.');
    if (dot)
    {
        *dot = '\0';
    }
}

static void build_dataset_paths(const char *fname, const char *problem_name, char *traj_path, size_t traj_len, char *log_path, size_t log_len)
{
    char resolved_path[PATH_MAX];
    const char *path_src = fname;
    if (fname && realpath(fname, resolved_path))
    {
        path_src = resolved_path;
    }
    char dataset_root[PATH_MAX];
    dataset_root[0] = '\0';
    const char *marker = NULL;
    if (path_src)
    {
        marker = strstr(path_src, "/dataset/");
    }
    if (marker)
    {
        size_t root_len = (size_t)(marker - path_src) + strlen("/dataset");
        snprintf(dataset_root, sizeof(dataset_root), "%.*s", (int)root_len, path_src);
    }
    else if (path_src && strncmp(path_src, "dataset/", 8) == 0)
    {
        snprintf(dataset_root, sizeof(dataset_root), "dataset");
    }
    else
    {
        snprintf(dataset_root, sizeof(dataset_root), "dataset");
    }
    char gen_dir[PATH_MAX];
    char logs_dir[PATH_MAX];
    snprintf(gen_dir, sizeof(gen_dir), "%s/gen", dataset_root);
    snprintf(logs_dir, sizeof(logs_dir), "%s/logs", dataset_root);
    LUtilEnsureDir(dataset_root);
    LUtilEnsureDir(gen_dir);
    LUtilEnsureDir(logs_dir);
    snprintf(traj_path, traj_len, "%s/%s.csv", gen_dir, problem_name);
    snprintf(log_path, log_len, "%s/%s.log", logs_dir, problem_name);
}

void lorads_log_printf(lorads_solver *solver, const char *fmt, ...)
{
    if (!fmt)
    {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    if (solver && solver->log_ctx.log_fp)
    {
        va_list args_file;
        va_start(args_file, fmt);
        vfprintf(solver->log_ctx.log_fp, fmt, args_file);
        va_end(args_file);
        fflush(solver->log_ctx.log_fp);
    }
}

void lorads_logging_init(lorads_solver *solver, const lorads_params *params, double solve_start_time)
{
    if (!solver || !params)
    {
        return;
    }
    solver->oracleMethod = params->oracleRankMethod;
    solver->oracleEpsilon = LORADS_ORACLE_EPS;
    solver->log_ctx.solve_start_time = solve_start_time;
    solver->log_ctx.trajectory_fp = NULL;
    solver->log_ctx.log_fp = NULL;
    solver->log_ctx.problem_name[0] = '\0';
    solver->log_ctx.trajectory_path[0] = '\0';
    solver->log_ctx.log_path[0] = '\0';
    solver->log_ctx.naive_fallback_warned = 0;
    set_problem_name(params->fname, solver->log_ctx.problem_name, sizeof(solver->log_ctx.problem_name));
    build_dataset_paths(params->fname, solver->log_ctx.problem_name, solver->log_ctx.trajectory_path, sizeof(solver->log_ctx.trajectory_path), solver->log_ctx.log_path, sizeof(solver->log_ctx.log_path));
    solver->log_ctx.log_fp = fopen(solver->log_ctx.log_path, "w");
    if (solver->log_ctx.log_fp)
    {
        fprintf(solver->log_ctx.log_fp, "problem:%s\n", solver->log_ctx.problem_name);
        fprintf(solver->log_ctx.log_fp, "oracle_method:%d epsilon:%g\n", solver->oracleMethod, solver->oracleEpsilon);
        fflush(solver->log_ctx.log_fp);
    }
    solver->log_ctx.trajectory_fp = fopen(solver->log_ctx.trajectory_path, "w");
    if (solver->log_ctx.trajectory_fp)
    {
        fprintf(solver->log_ctx.trajectory_fp, "phase,iter,elapsed_sec,primal_obj,dual_obj,constr_violation_l1,constr_violation_inf,primal_dual_gap,current_rank,oracle_rank,cg_iter\n");
        fflush(solver->log_ctx.trajectory_fp);
    }
}

void lorads_logging_close(lorads_solver *solver)
{
    if (!solver)
    {
        return;
    }
    if (solver->log_ctx.trajectory_fp)
    {
        fclose(solver->log_ctx.trajectory_fp);
        solver->log_ctx.trajectory_fp = NULL;
    }
    if (solver->log_ctx.log_fp)
    {
        fclose(solver->log_ctx.log_fp);
        solver->log_ctx.log_fp = NULL;
    }
}

lorads_int lorads_sum_rank(const lorads_solver *solver)
{
    if (!solver || !solver->var)
    {
        return 0;
    }
    lorads_int total_rank = 0;
    for (lorads_int iCone = 0; iCone < solver->nCones; ++iCone)
    {
        if (solver->var->R && solver->var->R[iCone])
        {
            total_rank += solver->var->R[iCone]->rank;
        }
    }
    return total_rank;
}

static void build_gram_from_factor(const lorads_sdp_dense *factor, double *gram)
{
    if (!factor || !gram)
    {
        return;
    }
    lorads_int n = factor->nRows;
    lorads_int r = factor->rank;
    LORADS_ZERO(gram, double, r * r);
    for (lorads_int col1 = 0; col1 < r; ++col1)
    {
        const double *u1 = factor->matElem + col1 * n;
        for (lorads_int col2 = col1; col2 < r; ++col2)
        {
            const double *u2 = factor->matElem + col2 * n;
            double sum = 0.0;
            for (lorads_int row = 0; row < n; ++row)
            {
                sum += u1[row] * u2[row];
            }
            gram[col1 * r + col2] = sum;
            gram[col2 * r + col1] = sum;
        }
    }
}

static void build_gram_from_average(const lorads_sdp_dense *U, const lorads_sdp_dense *V, double *gram)
{
    if (!U || !V || !gram)
    {
        return;
    }
    lorads_int n = U->nRows;
    lorads_int r = U->rank;
    LORADS_ZERO(gram, double, r * r);
    for (lorads_int col1 = 0; col1 < r; ++col1)
    {
        const double *u1 = U->matElem + col1 * n;
        const double *v1 = V->matElem + col1 * n;
        for (lorads_int col2 = col1; col2 < r; ++col2)
        {
            const double *u2 = U->matElem + col2 * n;
            const double *v2 = V->matElem + col2 * n;
            double sum = 0.0;
            for (lorads_int row = 0; row < n; ++row)
            {
                double avg1 = 0.5 * (u1[row] + v1[row]);
                double avg2 = 0.5 * (u2[row] + v2[row]);
                sum += avg1 * avg2;
            }
            gram[col1 * r + col2] = sum;
            gram[col2 * r + col1] = sum;
        }
    }
}

static lorads_int count_significant_from_matrix(double *matrix, lorads_int n, double epsilon)
{
    if (!matrix || n <= 0)
    {
        return 0;
    }
    double *matrix_copy;
    LORADS_INIT(matrix_copy, double, (size_t)n * (size_t)n);
    if (!matrix_copy)
    {
        return -1;
    }
    LORADS_MEMCPY(matrix_copy, matrix, double, (size_t)n * (size_t)n);
    double *eigvals;
    LORADS_INIT(eigvals, double, n);
    double *z_dummy;
    LORADS_INIT(z_dummy, double, 1);
    lorads_int *isuppz;
    LORADS_INIT(isuppz, lorads_int, 2 * n);
    double work_query = 0.0;
    lorads_int iwork_query = 0;
    lorads_int lwork = -1;
    lorads_int liwork = -1;
    lorads_int m = 0;
    lorads_int info = 0;
    lorads_int ldz = 1;
    char jobz = 'N';
    char range = 'A';
    char uplo = ACharConstantUploUp;
    double vl = 0.0;
    double vu = 0.0;
    lorads_int il = 1;
    lorads_int iu = n;
#ifdef UNDER_BLAS
    dsyevr_(&jobz, &range, &uplo, &n, matrix_copy, &n, &vl, &vu, &il, &iu, &AblConstantZero, &m, eigvals, z_dummy, &ldz, isuppz, &work_query, &lwork, &iwork_query, &liwork, &info);
#else
    dsyevr(&jobz, &range, &uplo, &n, matrix_copy, &n, &vl, &vu, &il, &iu, &AblConstantZero, &m, eigvals, z_dummy, &ldz, isuppz, &work_query, &lwork, &iwork_query, &liwork, &info);
#endif
    if (info != 0)
    {
        LORADS_FREE(matrix_copy);
        LORADS_FREE(eigvals);
        LORADS_FREE(z_dummy);
        LORADS_FREE(isuppz);
        return -1;
    }
    lwork = (lorads_int)work_query;
    liwork = iwork_query;
    lorads_int min_lwork = 26 * n;
    lorads_int min_liwork = 10 * n;
    if (lwork < min_lwork)
    {
        lwork = min_lwork;
    }
    if (liwork < min_liwork)
    {
        liwork = min_liwork;
    }
    double *work;
    LORADS_INIT(work, double, lwork);
    lorads_int *iwork;
    LORADS_INIT(iwork, lorads_int, liwork);
#ifdef UNDER_BLAS
    dsyevr_(&jobz, &range, &uplo, &n, matrix_copy, &n, &vl, &vu, &il, &iu, &AblConstantZero, &m, eigvals, z_dummy, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
#else
    dsyevr(&jobz, &range, &uplo, &n, matrix_copy, &n, &vl, &vu, &il, &iu, &AblConstantZero, &m, eigvals, z_dummy, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
#endif
    lorads_int rank = 0;
    if (info == 0 && m > 0)
    {
        double lambda_max = eigvals[m - 1];
        if (lambda_max > 0)
        {
            double cutoff = epsilon * lambda_max;
            for (lorads_int i = 0; i < m; ++i)
            {
                if (eigvals[i] > cutoff)
                {
                    rank += 1;
                }
            }
        }
    }
    else
    {
        rank = -1;
    }
    LORADS_FREE(matrix_copy);
    LORADS_FREE(eigvals);
    LORADS_FREE(z_dummy);
    LORADS_FREE(isuppz);
    LORADS_FREE(work);
    LORADS_FREE(iwork);
    return rank;
}

static lorads_int oracle_rank_from_factor(const lorads_sdp_dense *factor, double epsilon)
{
    if (!factor)
    {
        return 0;
    }
    lorads_int r = factor->rank;
    double *gram;
    LORADS_INIT(gram, double, (size_t)r * (size_t)r);
    if (!gram)
    {
        return -1;
    }
    build_gram_from_factor(factor, gram);
    lorads_int rank = count_significant_from_matrix(gram, r, epsilon);
    LORADS_FREE(gram);
    return rank;
}

static lorads_int oracle_rank_from_average(const lorads_sdp_dense *U, const lorads_sdp_dense *V, double epsilon)
{
    if (!U || !V)
    {
        return 0;
    }
    lorads_int r = U->rank;
    double *gram;
    LORADS_INIT(gram, double, (size_t)r * (size_t)r);
    if (!gram)
    {
        return -1;
    }
    build_gram_from_average(U, V, gram);
    lorads_int rank = count_significant_from_matrix(gram, r, epsilon);
    LORADS_FREE(gram);
    return rank;
}

static lorads_int oracle_rank_naive_factor(const lorads_sdp_dense *factor, double epsilon, lorads_solver *solver)
{
    if (!factor)
    {
        return 0;
    }
    lorads_int n = factor->nRows;
    lorads_int r = factor->rank;
    if (n <= 0 || r <= 0)
    {
        return 0;
    }
    if (n > 2000)
    {
        if (solver && solver->log_ctx.naive_fallback_warned == 0)
        {
            lorads_log_printf(solver, "skip naive oracle rank for n=%ld, falling back to gram approach.\n", (long)n);
            solver->log_ctx.naive_fallback_warned = 1;
        }
        return oracle_rank_from_factor(factor, epsilon);
    }
    double *full_matrix;
    LORADS_INIT(full_matrix, double, (size_t)n * (size_t)n);
    if (!full_matrix)
    {
        return -1;
    }
    LORADS_ZERO(full_matrix, double, (size_t)n * (size_t)n);
    for (lorads_int i = 0; i < n; ++i)
    {
        for (lorads_int j = i; j < n; ++j)
        {
            double sum = 0.0;
            for (lorads_int k = 0; k < r; ++k)
            {
                const double *col = factor->matElem + k * n;
                sum += col[i] * col[j];
            }
            full_matrix[i + j * n] = sum;
            full_matrix[j + i * n] = sum;
        }
    }
    lorads_int rank = count_significant_from_matrix(full_matrix, n, epsilon);
    LORADS_FREE(full_matrix);
    return rank;
}

static lorads_int oracle_rank_naive_average(const lorads_sdp_dense *U, const lorads_sdp_dense *V, double epsilon, lorads_solver *solver)
{
    if (!U || !V)
    {
        return 0;
    }
    lorads_int n = U->nRows;
    lorads_int r = U->rank;
    if (n <= 0 || r <= 0)
    {
        return 0;
    }
    if (n > 2000)
    {
        if (solver && solver->log_ctx.naive_fallback_warned == 0)
        {
            lorads_log_printf(solver, "skip naive oracle rank for n=%ld, falling back to gram approach.\n", (long)n);
            solver->log_ctx.naive_fallback_warned = 1;
        }
        return oracle_rank_from_average(U, V, epsilon);
    }
    double *full_matrix;
    LORADS_INIT(full_matrix, double, (size_t)n * (size_t)n);
    if (!full_matrix)
    {
        return -1;
    }
    LORADS_ZERO(full_matrix, double, (size_t)n * (size_t)n);
    for (lorads_int i = 0; i < n; ++i)
    {
        for (lorads_int j = i; j < n; ++j)
        {
            double sum = 0.0;
            for (lorads_int k = 0; k < r; ++k)
            {
                const double *u_col = U->matElem + k * n;
                const double *v_col = V->matElem + k * n;
                double avg_i = 0.5 * (u_col[i] + v_col[i]);
                double avg_j = 0.5 * (u_col[j] + v_col[j]);
                sum += avg_i * avg_j;
            }
            full_matrix[i + j * n] = sum;
            full_matrix[j + i * n] = sum;
        }
    }
    lorads_int rank = count_significant_from_matrix(full_matrix, n, epsilon);
    LORADS_FREE(full_matrix);
    return rank;
}

lorads_int lorads_compute_oracle_rank(lorads_solver *solver, int phase)
{
    if (!solver)
    {
        return 0;
    }
    double epsilon = solver->oracleEpsilon > 0 ? solver->oracleEpsilon : LORADS_ORACLE_EPS;
    lorads_int total_rank = 0;
    for (lorads_int iCone = 0; iCone < solver->nCones; ++iCone)
    {
        lorads_int block_rank = 0;
        if (phase == 1)
        {
            if (solver->oracleMethod == LORADS_ORACLE_RANK_GRAM)
            {
                block_rank = oracle_rank_from_factor(solver->var->R[iCone], epsilon);
            }
            else
            {
                block_rank = oracle_rank_naive_factor(solver->var->R[iCone], epsilon, solver);
            }
        }
        else
        {
            if (solver->oracleMethod == LORADS_ORACLE_RANK_GRAM)
            {
                block_rank = oracle_rank_from_average(solver->var->U[iCone], solver->var->V[iCone], epsilon);
            }
            else
            {
                block_rank = oracle_rank_naive_average(solver->var->U[iCone], solver->var->V[iCone], epsilon, solver);
            }
        }
        if (block_rank < 0)
        {
            return block_rank;
        }
        total_rank += block_rank;
    }
    return total_rank;
}

void lorads_append_trajectory(lorads_solver *solver, int phase, lorads_int iter, double elapsed, double primal_obj, double dual_obj, double constr_l1, double constr_inf, double pd_gap, lorads_int current_rank, lorads_int oracle_rank, lorads_int cg_iter)
{
    if (!solver || !solver->log_ctx.trajectory_fp)
    {
        return;
    }
    fprintf(solver->log_ctx.trajectory_fp, "%d,%lld,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%lld,%lld,%lld\n", phase, (long long)iter, elapsed, primal_obj, dual_obj, constr_l1, constr_inf, pd_gap, (long long)current_rank, (long long)oracle_rank, (long long)cg_iter);
    fflush(solver->log_ctx.trajectory_fp);
}
