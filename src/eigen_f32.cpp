#include <ldlt/views.hpp>

namespace ldlt {
namespace detail {
LDLT_EXPLICIT_TPL_DEF(4, noalias_mul_add<f32>);
LDLT_EXPLICIT_TPL_DEF(3, assign_cwise_prod<f32>);
LDLT_EXPLICIT_TPL_DEF(3, assign_scalar_prod<f32>);
LDLT_EXPLICIT_TPL_DEF(2, trans_tr_unit_up_solve_in_place_on_right<f32>);
LDLT_EXPLICIT_TPL_DEF(3, apply_diag_inv_on_right<f32>);
LDLT_EXPLICIT_TPL_DEF(3, apply_diag_on_right<f32>);
LDLT_EXPLICIT_TPL_DEF(3, noalias_mul_sub_tr_lo<f32>);
} // namespace detail
} // namespace ldlt
