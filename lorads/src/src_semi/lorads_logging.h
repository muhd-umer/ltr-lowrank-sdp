#ifndef LORADS_LOGGING_H
#define LORADS_LOGGING_H

#include <stdarg.h>
#include "lorads_solver.h"
#include "lorads.h"

void lorads_logging_init(lorads_solver *solver, const lorads_params *params, double solve_start_time);
void lorads_logging_close(lorads_solver *solver);
lorads_int lorads_sum_rank(const lorads_solver *solver);
lorads_int lorads_compute_oracle_rank(lorads_solver *solver, int phase);
void lorads_append_trajectory(lorads_solver *solver, int phase, lorads_int iter, double elapsed, double primal_obj, double dual_obj, double constr_l1, double constr_inf, double pd_gap, lorads_int current_rank, lorads_int oracle_rank, lorads_int cg_iter);
void lorads_log_printf(lorads_solver *solver, const char *fmt, ...);
void lorads_write_json_output(lorads_solver *solver, lorads_int final_oracle_rank, double primal_obj, double dual_obj, double constr_l1, double constr_inf, double pd_gap, double solve_time, double rho_max, double heuristic_factor);

#endif
