/**
 * @file constants.hpp 
*/
#ifndef PROXSUITE_QP_CONSTANTS_HPP
#define PROXSUITE_QP_CONSTANTS_HPP

#include <veg/type_traits/core.hpp>
#include "qp/sparse/fwd.hpp"

namespace proxsuite {
namespace qp {

// SOLVER STATUS
enum struct QPSolverOutput {
	PROXQP_SOLVED, // the problem is solved.
	PROXQP_MAX_ITER_REACHED, // the maximum number of iterations has been reached.
	PROXQP_PRIMAL_INFEASIBLE, // the problem is primal infeasible.
	PROXQP_DUAL_INFEASIBLE // the problem is dual infeasible.
};
// INITIAL GUESS STATUS
enum struct InitialGuessStatus {
	NO_INITIAL_GUESS,
	EQUALITY_CONSTRAINED_INITIAL_GUESS, 
	WARM_START_WITH_PREVIOUS_RESULT,
	WARM_START, 
	COLD_START_WITH_PREVIOUS_RESULT 
};
// PRECONDITIONER STATUS
enum struct PreconditionerStatus {
	EXECUTE,// initialize or update with qp in entry
	KEEP, // keep previous preconditioner (for update method)
	IDENTITY // do not execute, hence use identity preconditioner (for init method)
};


} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_CONSTANTS_HPP */