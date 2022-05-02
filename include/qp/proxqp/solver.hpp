#ifndef INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS

#include "qp/views.hpp"
#include "qp/proxqp/line_search.hpp"
#include <cmath>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <veg/util/dynstack_alloc.hpp>

#include <dense-ldlt/ldlt.hpp>

template <typename Derived>
void save_data(
		const std::string& filename, const Eigen::MatrixBase<Derived>& mat) {
	// https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static Eigen::IOFormat CSVFormat(
			Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << mat.format(CSVFormat);
		file.close();
	}
}

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")
template <typename T>
auto positive_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

template <typename T>
void refactorize(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T rho_new) {

	if (!qpwork.constraints_changed && rho_new == qpresults.rho) {
		return;
	}

	qpwork.dw_aug.setZero();
	qpwork.kkt.diagonal().head(qpmodel.dim).array() += rho_new - qpresults.rho;
	qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
			-qpresults.mu_eq_inv;

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
	qpwork.ldl.factorize(qpwork.kkt, stack);

	isize n = qpmodel.dim;
	isize n_eq = qpmodel.n_eq;
	isize n_in = qpmodel.n_in;
	isize n_c = qpresults.n_c;

	LDLT_TEMP_MAT(T, new_cols, n + n_eq + n_c, n_c, stack);

	for (isize i = 0; i < n_in; ++i) {
		isize j = qpwork.current_bijection_map[i];
		if (j < n_c) {
			auto col = new_cols.col(j);
			col.head(n) = qpwork.C_scaled.row(i);
			col.segment(n, n_eq + n_c).setZero();
			col(n + n_eq + j) = -qpresults.mu_in_inv;
		}
	}
	qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);

	qpwork.constraints_changed = false;

	qpwork.dw_aug.setZero();
}

template <typename T>
void mu_update(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T mu_eq_new_inv,
		T mu_in_new_inv) {
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};

	isize n = qpmodel.dim;
	isize n_eq = qpmodel.n_eq;
	isize n_c = qpresults.n_c;

	if ((n_eq + n_c) == 0) {
		return;
	}

	LDLT_TEMP_VEC_UNINIT(T, rank_update_alpha, n_eq + n_c, stack);
	rank_update_alpha.head(n_eq).setConstant(qpresults.mu_eq_inv - mu_eq_new_inv);
	rank_update_alpha.tail(n_c).setConstant(qpresults.mu_in_inv - mu_in_new_inv);

	{
		auto _indices =
				stack.make_new_for_overwrite(veg::Tag<isize>{}, n_eq + n_c).unwrap();
		isize* indices = _indices.ptr_mut();
		for (isize k = 0; k < n_eq; ++k) {
			indices[k] = n + k;
		}
		for (isize k = 0; k < n_c; ++k) {
			indices[n_eq + k] = n + n_eq + k;
		}
		qpwork.ldl.diagonal_update_clobber_indices(
				indices, n_eq + n_c, rank_update_alpha, stack);
	}

	qpwork.constraints_changed = true;
}

template <typename T>
void iterative_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		isize inner_pb_dim) {

	qpwork.err.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);

	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.H_scaled.template selfadjointView<Eigen::Lower>() *
			qpwork.dw_aug.head(qpmodel.dim);
	qpwork.err.head(qpmodel.dim) -=
			qpresults.rho * qpwork.dw_aug.head(qpmodel.dim);

	// PERF: fuse {A, C}_scaled multiplication operations
	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.A_scaled.transpose() *
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.n_c) {
			qpwork.err.head(qpmodel.dim).noalias() -=
					qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
					qpwork.C_scaled.row(i);
			qpwork.err(qpmodel.dim + qpmodel.n_eq + j) -=
					(qpwork.C_scaled.row(i).dot(qpwork.dw_aug.head(qpmodel.dim)) -
			     qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
			         qpresults.mu_in_inv); // mu stores the inverse of mu
		}
	}
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() -=
			qpwork.A_scaled *
			qpwork.dw_aug.head(qpmodel.dim); // mu stores the inverse of mu
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) +=
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) *
			qpresults.mu_eq_inv; // mu stores the inverse of mu
}

template <typename T>
void iterative_solve_with_permut_fact( //
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps,
		isize inner_pb_dim) {

	qpwork.err.setZero();
	i32 it = 0;
	i32 it_stability = 0;

	qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
	qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), stack);

	qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

	++it;
	T preverr = infty_norm(qpwork.err.head(inner_pb_dim));
	if (qpsettings.verbose) {
		std::cout << "infty_norm(res) "
							<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
	}
	while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

		if (it >= qpsettings.nb_iterative_refinement) {
			break;
		}

		++it;
		qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
		qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

		qpwork.err.head(inner_pb_dim).setZero();
		qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
			it_stability += 1;

		} else {
			it_stability = 0;
		}
		if (it_stability == 2) {
			break;
		}
		preverr = infty_norm(qpwork.err.head(inner_pb_dim));

		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
	}

	if (infty_norm(qpwork.err.head(inner_pb_dim)) >=
	    std::max(eps, qpsettings.eps_refact)) {
		refactorize(qpmodel, qpresults, qpwork, qpresults.rho);
		it = 0;
		it_stability = 0;

		qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
		qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), stack);

		qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		preverr = infty_norm(qpwork.err.head(inner_pb_dim));
		++it;
		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
		while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

			if (it >= qpsettings.nb_iterative_refinement) {
				break;
			}
			++it;
			qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
			qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

			qpwork.err.head(inner_pb_dim).setZero();
			qp::detail::iterative_residual<T>(
					qpmodel, qpresults, qpwork, inner_pb_dim);

			if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
				it_stability += 1;

			} else {
				it_stability = 0;
			}
			if (it_stability == 2) {
				break;
			}
			preverr = infty_norm(qpwork.err.head(inner_pb_dim));

			if (qpsettings.verbose) {
				std::cout << "infty_norm(res) "
									<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
			}
		}
	}
	qpwork.rhs.head(inner_pb_dim).setZero();
}

template <typename T>
void bcl_update(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& primal_feasibility_lhs_new,
		T& bcl_eta_ext,
		T& bcl_eta_in,

		T bcl_eta_ext_init,
		T eps_in_min,

		T& new_bcl_mu_in,
		T& new_bcl_mu_eq,
		T& new_bcl_mu_in_inv,
		T& new_bcl_mu_eq_inv

) {

	if (primal_feasibility_lhs_new <= bcl_eta_ext) {
		if (qpsettings.verbose) {
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext * pow(qpresults.mu_in_inv, qpsettings.beta_bcl);
		bcl_eta_in = max2(bcl_eta_in * qpresults.mu_in_inv, eps_in_min);
	} else {
		if (qpsettings.verbose) {
			std::cout << "bad step" << std::endl;
		}

		qpresults.y = qpwork.y_prev;
		qpresults.z = qpwork.z_prev;

		new_bcl_mu_in = std::min(
				qpresults.mu_in * qpsettings.mu_update_factor, qpsettings.mu_max_in);
		new_bcl_mu_eq = std::min(
				qpresults.mu_eq * qpsettings.mu_update_factor, qpsettings.mu_max_eq);
		new_bcl_mu_in_inv = max2(
				qpresults.mu_in_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_in_inv); // mu stores the inverse of mu
		new_bcl_mu_eq_inv = max2(
				qpresults.mu_eq_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_eq_inv); // mu stores the inverse of mu
		bcl_eta_ext =
				bcl_eta_ext_init * pow(new_bcl_mu_in_inv, qpsettings.alpha_bcl);
		bcl_eta_in = max2(new_bcl_mu_in_inv, eps_in_min);
	}
}

template <typename T>
void global_primal_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T& primal_feasibility_eq_lhs,
		T& primal_feasibility_in_lhs) {

	qpwork.primal_residual_eq_scaled.noalias() = qpwork.A_scaled * qpresults.x;
	qpwork.primal_residual_in_scaled_up.noalias() = qpwork.C_scaled * qpresults.x;

	qpwork.ruiz.unscale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm(qpwork.primal_residual_eq_scaled);
	qpwork.ruiz.unscale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_in_scaled_up});
	primal_feasibility_in_rhs_0 = infty_norm(qpwork.primal_residual_in_scaled_up);

	qpwork.primal_residual_in_scaled_low =
			detail::positive_part(qpwork.primal_residual_in_scaled_up - qpmodel.u) +
			detail::negative_part(qpwork.primal_residual_in_scaled_up - qpmodel.l);
	qpwork.primal_residual_eq_scaled -= qpmodel.b;

	primal_feasibility_in_lhs = infty_norm(qpwork.primal_residual_in_scaled_low);
	primal_feasibility_eq_lhs = infty_norm(qpwork.primal_residual_eq_scaled);
	primal_feasibility_lhs =
			max2(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	qpwork.ruiz.scale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
}

template <typename T>
void global_dual_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3) {

	qpwork.dual_residual_scaled = qpwork.g_scaled;
	qpwork.CTz.noalias() =
			qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * qpresults.x;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_0 = infty_norm(qpwork.CTz);
	qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * qpresults.y;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_1 = infty_norm(qpwork.CTz);

	qpwork.CTz.noalias() = qpwork.C_scaled.transpose() * qpresults.z;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_3 = infty_norm(qpwork.CTz);

	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});

	dual_feasibility_lhs = infty_norm(qpwork.dual_residual_scaled);

	qpwork.ruiz.scale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});
};

template <typename T>
auto compute_inner_loop_saddle_point(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) -> T {

	qpwork.active_part_z =
			qp::detail::positive_part(qpwork.primal_residual_in_scaled_up) +
			qp::detail::negative_part(qpwork.primal_residual_in_scaled_low) -
			qpresults.z * qpresults.mu_in_inv; // contains now : [Cx-u+z_prev/mu_in]+
	                                       // + [Cx-l+z_prev/mu_in]- - z/mu_in

	T err = infty_norm(qpwork.active_part_z);
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) =
			qpwork.primal_residual_eq_scaled; // contains now Ax-b-(y-y_prev)/mu

	T prim_eq_e = infty_norm(
			qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)); // ||Ax-b-(y-y_prev)/mu||
	err = max2(err, prim_eq_e);
	T dual_e =
			infty_norm(qpwork.dual_residual_scaled); // contains ||Hx + rho(x-xprev) +
	                                             // g + Aty + Ctz||
	err = max2(err, dual_e);

	return err;
}

template <typename T>
void primal_dual_semi_smooth_newton_step(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps) {

	/* MUST BE
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+1./mu_eq (y_prev-y)
	 *  primal_residual_in_scaled_up = Cx-u+1./mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+1./mu_in(z_prev)
	 */

	qpwork.active_set_up.array() =
			(qpwork.primal_residual_in_scaled_up.array() >= 0);
	qpwork.active_set_low.array() =
			(qpwork.primal_residual_in_scaled_low.array() <= 0);
	qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
	isize numactive_inequalities = qpwork.active_inequalities.count();

	isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
	qpwork.rhs.setZero();
	qpwork.dw_aug.setZero();

	qp::line_search::active_set_change(qpmodel, qpresults, qpwork);

	qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled;

	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
			-qpwork.primal_residual_eq_scaled;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.n_c) {
			if (qpwork.active_set_up(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_up(i) +
						qpresults.z(i) * qpresults.mu_in_inv;
			} else if (qpwork.active_set_low(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_low(i) +
						qpresults.z(i) * qpresults.mu_in_inv;
			}
		} else {
			qpwork.rhs.head(qpmodel.dim) +=
					qpresults.z(i) *
					qpwork.C_scaled.row(i); // unactive unrelevant columns
		}
	}

	iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			eps,
			inner_pb_dim);

	// use active_part_z as a temporary variable to derive unpermutted dz step
	for (isize j = 0; j < qpmodel.n_in; ++j) {
		isize i = qpwork.current_bijection_map(j);
		if (i < qpresults.n_c) {
			qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
		} else {
			qpwork.active_part_z(j) = -qpresults.z(j);
		}
	}
	qpwork.dw_aug.tail(qpmodel.n_in) = qpwork.active_part_z;

}

template <typename T>
T primal_dual_newton_semi_smooth(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps_int) {

	/* MUST CONTAIN IN ENTRY WITH x = x_prev ; y = y_prev ; z = z_prev
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+1./mu_eq (y_prev-y)
	 *  primal_residual_in_scaled_up = Cx-u+1./mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+1./mu_in(z_prev)
	 */

	T err_in = 1.e6;

	for (i64 iter = 0; iter <= qpsettings.max_iter_in; ++iter) {

		if (iter == qpsettings.max_iter_in) {
			qpresults.n_tot += qpsettings.max_iter_in;
			break;
		}
		qp::detail::primal_dual_semi_smooth_newton_step<T>(
				qpsettings, qpmodel, qpresults, qpwork, eps_int);

		veg::dynstack::DynStackMut stack{
				veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
		LDLT_TEMP_VEC(T, ATdy, qpmodel.dim, stack);
		LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);

		auto& Hdx = qpwork.Hdx;
		auto& Adx = qpwork.Adx;
		auto& Cdx = qpwork.Cdx;

		auto dx = qpwork.dw_aug.head(qpmodel.dim);
		auto dy = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
		auto dz = qpwork.dw_aug.segment(qpmodel.dim + qpmodel.n_eq, qpmodel.n_in);

		Hdx.setZero();
		Adx.setZero();
		Cdx.setZero();

		Hdx.noalias() +=
				qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * dx;

		Adx.noalias() += qpwork.A_scaled * dx;
		ATdy.noalias() += qpwork.A_scaled.transpose() * dy;

		Cdx.noalias() += qpwork.C_scaled * dx;
		CTdz.noalias() += qpwork.C_scaled.transpose() * dz;

		if (qpmodel.n_in > 0) {
			qp::line_search::primal_dual_ls(qpmodel, qpresults, qpwork, qpsettings);
		}
		auto alpha = qpwork.alpha;

		if (infty_norm(alpha * qpwork.dw_aug) < 1.E-11 && iter > 0) {
			qpresults.n_tot += iter + 1;
			if (qpsettings.verbose) {
				std::cout << "infty_norm(alpha_step * dx) "
									<< infty_norm(alpha * qpwork.dw_aug) << std::endl;
			}
			break;
		}

		qpresults.x += alpha * dx;

		// contains now :  C(x+alpha dx)-u + z_prev/mu_in
		qpwork.primal_residual_in_scaled_up += alpha * Cdx;

		// contains now :  C(x+alpha dx)-l + z_prev/mu_in
		qpwork.primal_residual_in_scaled_low += alpha * Cdx;

		qpwork.primal_residual_eq_scaled +=
				alpha * (Adx - qpresults.mu_eq_inv * dy);

		qpresults.y += alpha * dy;
		qpresults.z += alpha * dz;

		qpwork.dual_residual_scaled +=
				alpha * (qpresults.rho * dx + Hdx + ATdy + CTdz);

		err_in =
				detail::compute_inner_loop_saddle_point(qpmodel, qpresults, qpwork);

		if (qpsettings.verbose) {
			std::cout << "---it in " << iter << " projection norm " << err_in
								<< " alpha " << alpha << std::endl;
		}

		if (err_in <= eps_int) {
			qpresults.n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}

template <typename T>
void qp_solve( //
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	using namespace ldlt::tags;

	/*** TEST WITH MATRIX FULL OF NAN FOR DEBUG
	  static constexpr Layout layout = rowmajor;
	  static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	RowMat test(2,2); // test it is full of nan for debug
	std::cout << "test " << test << std::endl;
	*/

	//::Eigen::internal::set_is_malloc_allowed(false);

	T bcl_eta_ext_init = pow(T(0.1), qpsettings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(qpsettings.eps_abs, T(1.E-9));

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);
	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);

	for (i64 iter = 0; iter <= qpsettings.max_iter; ++iter) {

		qpresults.n_ext += 1;
		if (iter == qpsettings.max_iter) {
			break;
		}

		// compute primal residual

		// PERF: fuse matrix product computations in global_{primal, dual}_residual
		global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);

		T new_bcl_mu_in(qpresults.mu_in);
		T new_bcl_mu_eq(qpresults.mu_eq);
		T new_bcl_mu_in_inv(qpresults.mu_in_inv);
		T new_bcl_mu_eq_inv(qpresults.mu_eq_inv);

		T rhs_pri(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_pri +=
					qpsettings.eps_rel *
					max2(
							max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
							max2(
									max2(
											qpwork.primal_feasibility_rhs_1_eq,
											qpwork.primal_feasibility_rhs_1_in_u),
									qpwork.primal_feasibility_rhs_1_in_l));
		}
		bool is_primal_feasible = primal_feasibility_lhs <= rhs_pri;

		T rhs_dua(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_dua +=
					qpsettings.eps_rel *
					max2(
							max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
							max2(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2));
		}

		bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;

		if (qpsettings.verbose) {
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext
								<< " bcl_eta_in : " << bcl_eta_in << " rho : " << qpresults.rho
								<< " bcl_mu_eq : " << qpresults.mu_eq
								<< " bcl_mu_in : " << qpresults.mu_in << std::endl;
			std::cout << "qpsettings.eps_abs " << qpsettings.eps_abs
								<< "  qpsettings.eps_rel *rhs "
								<< qpsettings.eps_rel *
											 max2(
													 max2(
															 primal_feasibility_eq_rhs_0,
															 primal_feasibility_in_rhs_0),
													 max2(
															 max2(
																	 qpwork.primal_feasibility_rhs_1_eq,
																	 qpwork.primal_feasibility_rhs_1_in_u),
															 qpwork.primal_feasibility_rhs_1_in_l))
								<< std::endl;
			std::cout << "is_primal_feasible " << is_primal_feasible
								<< " is_dual_feasible " << is_dual_feasible << std::endl;
		}
		if (is_primal_feasible) {

			if (dual_feasibility_lhs >=
			        qpsettings.refactor_dual_feasibility_threshold &&
			    qpresults.rho != qpsettings.refactor_rho_threshold) {

				T rho_new(qpsettings.refactor_rho_threshold);

				refactorize(qpmodel, qpresults, qpwork, rho_new);

				qpresults.rho = rho_new;
			}
			if (is_dual_feasible) {

				break;
			}
		}

		qpwork.x_prev = qpresults.x;
		qpwork.y_prev = qpresults.y;
		qpwork.z_prev = qpresults.z;

		// primal dual version from gill and robinson

		qpwork.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{
				from_eigen,
				qpwork.primal_residual_in_scaled_up}); // contains now scaled(Cx)
		qpwork.primal_residual_in_scaled_up +=
				qpwork.z_prev *
				qpresults.mu_in_inv; // contains now scaled(Cx+z_prev/mu_in)
		qpwork.primal_residual_in_scaled_low = qpwork.primal_residual_in_scaled_up;
		qpwork.primal_residual_in_scaled_up -=
				qpwork.u_scaled; // contains now scaled(Cx-u+z_prev/mu_in)
		qpwork.primal_residual_in_scaled_low -=
				qpwork.l_scaled; // contains now scaled(Cx-l+z_prev/mu_in)

		T err_in = qp::detail::primal_dual_newton_semi_smooth(
				qpsettings, qpmodel, qpresults, qpwork, bcl_eta_in);
		if (qpsettings.verbose) {
			std::cout << " inner loop residual : " << err_in << std::endl;
		}

		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		is_primal_feasible =
				primal_feasibility_lhs_new <=
				(qpsettings.eps_abs +
		     qpsettings.eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 qpwork.primal_feasibility_rhs_1_eq,
												 qpwork.primal_feasibility_rhs_1_in_u),
										 qpwork.primal_feasibility_rhs_1_in_l)));

		if (is_primal_feasible) {
			T dual_feasibility_lhs_new(dual_feasibility_lhs);

			global_dual_residual(
					qpmodel,
					qpresults,
					qpwork,
					dual_feasibility_lhs_new,
					dual_feasibility_rhs_0,
					dual_feasibility_rhs_1,
					dual_feasibility_rhs_3);

			is_dual_feasible =
					dual_feasibility_lhs_new <=
					(qpsettings.eps_abs +
			     qpsettings.eps_rel *
			         max2(
									 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
									 max2(
											 dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)));

			if (is_dual_feasible) {

				break;
			}
		}

		bcl_update(
				qpsettings,
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				bcl_eta_ext_init,
				eps_in_min,

				new_bcl_mu_in,
				new_bcl_mu_eq,
				new_bcl_mu_in_inv,
				new_bcl_mu_eq_inv

		);

		// COLD RESTART

		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);

		if (primal_feasibility_lhs_new >= primal_feasibility_lhs &&
		    dual_feasibility_lhs_new >= dual_feasibility_lhs &&
		    qpresults.mu_in >= T(1e5)) {

			if (qpsettings.verbose) {
				std::cout << "cold restart" << std::endl;
			}

			new_bcl_mu_in = qpsettings.cold_reset_mu_in;
			new_bcl_mu_eq = qpsettings.cold_reset_mu_eq;
			new_bcl_mu_in_inv = qpsettings.cold_reset_mu_in_inv;
			new_bcl_mu_eq_inv = qpsettings.cold_reset_mu_eq_inv;
		}

		/// effective mu upddate

		if (qpresults.mu_in != new_bcl_mu_in || qpresults.mu_eq != new_bcl_mu_eq) {
			{ ++qpresults.n_mu_change; }
			mu_update(
					qpmodel, qpresults, qpwork, new_bcl_mu_eq_inv, new_bcl_mu_in_inv);
		}

		qpresults.mu_eq = new_bcl_mu_eq;
		qpresults.mu_in = new_bcl_mu_in;
		qpresults.mu_eq_inv = new_bcl_mu_eq_inv;
		qpresults.mu_in_inv = new_bcl_mu_in_inv;
	}

	qpwork.ruiz.unscale_primal_in_place(
			VectorViewMut<T>{from_eigen, qpresults.x});
	qpwork.ruiz.unscale_dual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpresults.y});
	qpwork.ruiz.unscale_dual_in_place_in(
			VectorViewMut<T>{from_eigen, qpresults.z});

	{
		// EigenAllowAlloc _{};
		for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
			qpresults.objValue +=
					0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
			qpresults.objValue +=
					qpresults.x(j) * T(qpmodel.H.col(j)
			                           .tail(qpmodel.dim - j - 1)
			                           .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
		}
		qpresults.objValue += (qpmodel.g).dot(qpresults.x);
	}
}

template <typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using MatRef =
		Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const>;

template <typename Mat, typename T>
void QPsetup_generic( //
		Mat const& H,
		VecRef<T> g,
		Mat const& A,
		VecRef<T> b,
		Mat const& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults) {

	qpmodel.H = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(H);
	qpmodel.g = g;
	qpmodel.A = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(A);
	qpmodel.b = b;
	qpmodel.C = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(C);
	qpmodel.u = u;
	qpmodel.l = l;

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut,
			qpwork.ldl_stack.as_mut(),
	};
	qpwork.ruiz.scale_qp_in_place(qp_scaled, stack);
	qpwork.dw_aug.setZero();

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv);

	qpwork.ldl.factorize(qpwork.kkt, stack);

	if (!QPSettings.warm_start) {
		qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
		qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;
		qp::detail::iterative_solve_with_permut_fact( //
				QPSettings,
				qpmodel,
				QPResults,
				qpwork,
				T(1),
				qpmodel.dim + qpmodel.n_eq);

		QPResults.x = qpwork.dw_aug.head(qpmodel.dim); 
		QPResults.y = qpwork.dw_aug.segment(
				qpmodel.dim, qpmodel.n_eq); 
		qpwork.dw_aug.setZero();
	}

	qpwork.rhs.setZero();
}

template <typename T>
void QPsetup_dense( //
		MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults

) {
	detail::QPsetup_generic(
			H, g, A, b, C, u, l, QPSettings, qpmodel, qpwork, QPResults);
}

template <typename T>
void QPsetup( //
		const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults) {
	detail::QPsetup_generic(
			H, g, A, b, C, u, l, QPSettings, qpmodel, qpwork, QPResults);
}

} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS */
