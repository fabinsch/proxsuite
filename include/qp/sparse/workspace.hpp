/** \file */

#ifndef PROXSUITE_QP_SPARSE_WORKSPACE_HPP
#define PROXSUITE_QP_SPARSE_WORKSPACE_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/settings.hpp>
#include <veg/vec.hpp>
#include "qp/sparse/views.hpp"
#include "qp/sparse/model.hpp"
#include "qp/results.hpp"
#include "qp/sparse/utils.hpp"

#include <iostream>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {

template <typename T, typename I>
struct Workspace;

template <typename T, typename I>
void refactorize(
		Results<T> const& results,
		bool do_ldlt,
		isize n_tot,
		linearsolver::sparse::MatMut<T, I> kkt_active,
		veg::SliceMut<bool> active_constraints,
		Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>& iterative_solver,
		Model<T, I> data,
		I* etree,
		I* ldl_nnz_counts,
		I* ldl_row_indices,
		I* perm_inv,
		T* ldl_values,
		I* perm,
		I* ldl_col_ptrs,
		veg::dynstack::DynStackMut stack,
		linearsolver::sparse::MatMut<T, I> ldl,
		detail::AugmentedKkt<T, I>& aug_kkt,
		veg::Tag<T>& xtag) {
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	if (do_ldlt) {
		linearsolver::sparse::factorize_symbolic_non_zeros(
				ldl_nnz_counts, etree, perm_inv, perm, kkt_active.symbolic(), stack);

		auto _diag = stack.make_new_for_overwrite(xtag, n_tot).unwrap();
		T* diag = _diag.ptr_mut();

		for (isize i = 0; i < data.dim; ++i) {
			diag[i] = results.info.rho;
		}
		for (isize i = 0; i < data.n_eq; ++i) {
			diag[data.dim + i] = mu_eq_neg;
		}
		for (isize i = 0; i < data.n_in; ++i) {
			diag[(data.dim + data.n_eq) + i] =
					active_constraints[i] ? mu_in_neg : T(1);
		}

		linearsolver::sparse::factorize_numeric(
				ldl_values,
				ldl_row_indices,
				diag,
				perm,
				ldl_col_ptrs,
				etree,
				perm_inv,
				kkt_active.as_const(),
				stack);
		isize ldl_nnz = 0;
		for (isize i = 0; i < n_tot; ++i) {
			ldl_nnz = linearsolver::sparse::util::checked_non_negative_plus(
					ldl_nnz, isize(ldl_nnz_counts[i]));
		}
		ldl._set_nnz(ldl_nnz);
	} else {
		aug_kkt = {
				{kkt_active.as_const(),
		     active_constraints.as_const(),
		     data.dim,
		     data.n_eq,
		     data.n_in,
		     results.info.rho,
		     results.info.mu_eq_inv,
		     results.info.mu_in_inv}};
		iterative_solver.compute(aug_kkt);
	}
};

template <typename T, typename I>
struct Workspace {

	struct /* NOLINT */ {
		// temporary allocations
		veg::Vec<veg::mem::byte> storage;
		bool do_ldlt;
		// persistent allocations

		Eigen::Matrix<T, Eigen::Dynamic, 1> g_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> b_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> l_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> u_scaled;
		veg::Vec<I> kkt_nnz_counts;

		veg::Vec<I> ldl_col_ptrs;
		veg::Vec<I> ldl_nnz_counts;
		veg::Vec<I> ldl_row_indices;
		veg::Vec<T> ldl_values;
		veg::Vec<I> perm_inv;
		veg::Vec<I> etree;

		// stored in unique_ptr because we need a stable address
		std::unique_ptr<detail::AugmentedKkt<T, I>> matrix_free_kkt;
		std::unique_ptr<Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>>
				matrix_free_solver;

		auto stack_mut() -> veg::dynstack::DynStackMut {
			return {
					veg::from_slice_mut,
					storage.as_mut(),
			};
		}

		template <typename P>
		void setup_impl(
				QpView<T, I> qp,
				Results<T>& results,
				Model<T, I>& data,
				P& precond,
				veg::dynstack::StackReq precond_req) {
			data.dim = qp.H.nrows();
			data.n_eq = qp.AT.ncols();
			data.n_in = qp.CT.ncols();
			data.H_nnz = qp.H.nnz();
			data.A_nnz = qp.AT.nnz();
			data.C_nnz = qp.CT.nnz();

			data.g = qp.g.to_eigen();
			data.b = qp.b.to_eigen();
			data.l = qp.l.to_eigen();
			data.u = qp.u.to_eigen();

			using namespace veg::dynstack;
			using namespace linearsolver::sparse::util;

			using SR = StackReq;
			veg::Tag<I> itag;
			veg::Tag<T> xtag;

			isize n = qp.H.nrows();
			isize n_eq = qp.AT.ncols();
			isize n_in = qp.CT.ncols();
			isize n_tot = n + n_eq + n_in;

			isize nnz_tot = qp.H.nnz() + qp.AT.nnz() + qp.CT.nnz();

			// form the full kkt matrix
			// assuming H, AT, CT are sorted
			// and H is upper triangular
			{
				data.kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
				data.kkt_row_indices.resize_for_overwrite(nnz_tot);
				data.kkt_values.resize_for_overwrite(nnz_tot);

				I* kktp = data.kkt_col_ptrs.ptr_mut();
				I* kkti = data.kkt_row_indices.ptr_mut();
				T* kktx = data.kkt_values.ptr_mut();

				kktp[0] = 0;
				usize col = 0;
				usize pos = 0;

				auto insert_submatrix = [&](linearsolver::sparse::MatRef<T, I> m,
				                            bool assert_sym_hi) -> void {
					I const* mi = m.row_indices();
					T const* mx = m.values();
					isize ncols = m.ncols();

					for (usize j = 0; j < usize(ncols); ++j) {
						usize col_start = m.col_start(j);
						usize col_end = m.col_end(j);

						kktp[col + 1] =
								checked_non_negative_plus(kktp[col], I(col_end - col_start));
						++col;

						for (usize p = col_start; p < col_end; ++p) {
							usize i = zero_extend(mi[p]);
							if (assert_sym_hi) {
								VEG_ASSERT(i <= j);
							}

							kkti[pos] = veg::nb::narrow<I>{}(i);
							kktx[pos] = mx[p];

							++pos;
						}
					}
				};

				insert_submatrix(qp.H, true);
				insert_submatrix(qp.AT, false);
				insert_submatrix(qp.CT, false);
			}

			storage.resize_for_overwrite( //
					(StackReq::with_len(itag, n_tot) &
			     linearsolver::sparse::factorize_symbolic_req( //
							 itag,                                     //
							 n_tot,                                    //
							 nnz_tot,                                  //
							 linearsolver::sparse::Ordering::amd))     //
							.alloc_req()                               //
			);

			ldl_col_ptrs.resize_for_overwrite(n_tot + 1);
			perm_inv.resize_for_overwrite(n_tot);

			DynStackMut stack = stack_mut();

			bool overflow = false;
			{
				etree.resize_for_overwrite(n_tot);
				auto etree_ptr = etree.ptr_mut();

				using namespace veg::literals;
				auto kkt_sym = linearsolver::sparse::SymbolicMatRef<I>{
						linearsolver::sparse::from_raw_parts,
						n_tot,
						n_tot,
						nnz_tot,
						data.kkt_col_ptrs.ptr(),
						nullptr,
						data.kkt_row_indices.ptr(),
				};
				linearsolver::sparse::factorize_symbolic_non_zeros( //
						ldl_col_ptrs.ptr_mut() + 1,
						etree_ptr,
						perm_inv.ptr_mut(),
						static_cast<I const*>(nullptr),
						kkt_sym,
						stack);

				auto pcol_ptrs = ldl_col_ptrs.ptr_mut();
				pcol_ptrs[0] = I(0);

				using veg::u64;
				u64 acc = 0;

				for (usize i = 0; i < usize(n_tot); ++i) {
					acc += u64(zero_extend(pcol_ptrs[i + 1]));
					if (acc != u64(I(acc))) {
						overflow = true;
					}
					pcol_ptrs[(i + 1)] = I(acc);
				}
			}

			auto lnnz = isize(zero_extend(ldl_col_ptrs[n_tot]));

			// if ldlt is too sparse
			// do_ldlt = !overflow && lnnz < (10000000);
			do_ldlt = !overflow && lnnz < 10000000;

#define PROX_QP_ALL_OF(...)                                                    \
	::veg::dynstack::StackReq::and_(::veg::init_list(__VA_ARGS__))
#define PROX_QP_ANY_OF(...)                                                    \
	::veg::dynstack::StackReq::or_(::veg::init_list(__VA_ARGS__))

			auto refactorize_req =
					do_ldlt
							? PROX_QP_ANY_OF({
										linearsolver::sparse::
												factorize_symbolic_req( // symbolic ldl
														itag,
														n_tot,
														nnz_tot,
														linearsolver::sparse::Ordering::user_provided),
										PROX_QP_ALL_OF({
												SR::with_len(xtag, n_tot), // diag
												linearsolver::sparse::
														factorize_numeric_req( // numeric ldl
																xtag,
																itag,
																n_tot,
																nnz_tot,
																linearsolver::sparse::Ordering::user_provided),
										}),
								})
							: PROX_QP_ALL_OF({
										SR::with_len(itag, 0),
										SR::with_len(xtag, 0),
								});

			auto x_vec = [&](isize n) noexcept -> StackReq {
				return linearsolver::dense::temp_vec_req(xtag, n);
			};

			auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // tmp
					x_vec(n_tot), // err
					x_vec(n_tot), // work
			});

			auto unscaled_primal_dual_residual_req = x_vec(n); // Hx
			auto line_search_req = PROX_QP_ALL_OF({
					x_vec(2 * n_in), // alphas
					x_vec(n),        // Cdx_active
					x_vec(n_in),     // active_part_z
					x_vec(n_in),     // tmp_lo
					x_vec(n_in),     // tmp_up
			});
			auto primal_dual_newton_semi_smooth_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // dw
					PROX_QP_ANY_OF({
							ldl_solve_in_place_req,
							PROX_QP_ALL_OF({
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_lo
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_up
									SR::with_len(
											veg::Tag<bool>{}, n_in), // new_active_constraints
									(do_ldlt && n_in > 0)
											? PROX_QP_ANY_OF({
														linearsolver::sparse::add_row_req(
																xtag, itag, n_tot, false, n, n_tot),
														linearsolver::sparse::delete_row_req(
																xtag, itag, n_tot, n_tot),
												})
											: refactorize_req,
							}),
							PROX_QP_ALL_OF({
									x_vec(n),    // Hdx
									x_vec(n_eq), // Adx
									x_vec(n_in), // Cdx
									x_vec(n),    // ATdy
									x_vec(n),    // CTdz
							}),
					}),
					line_search_req,
			});

			auto iter_req = PROX_QP_ANY_OF({
					PROX_QP_ALL_OF(
							{x_vec(n_eq), // primal_residual_eq_scaled
			         x_vec(n_in), // primal_residual_in_scaled_lo
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n),    // dual_residual_scaled
			         PROX_QP_ANY_OF({
									 unscaled_primal_dual_residual_req,
									 PROX_QP_ALL_OF({
											 x_vec(n),    // x_prev
											 x_vec(n_eq), // y_prev
											 x_vec(n_in), // z_prev
											 primal_dual_newton_semi_smooth_req,
									 }),
							 })}),
					refactorize_req, // mu_update
			});

			auto req = //
					PROX_QP_ALL_OF({
							x_vec(n),                             // g_scaled
							x_vec(n_eq),                          // b_scaled
							x_vec(n_in),                          // l_scaled
							x_vec(n_in),                          // u_scaled
							SR::with_len(veg::Tag<bool>{}, n_in), // active constr
							SR::with_len(itag, n_tot),            // kkt nnz counts
							refactorize_req,
							PROX_QP_ANY_OF({
									precond_req,
									PROX_QP_ALL_OF({
											do_ldlt ? PROX_QP_ALL_OF({
																		SR::with_len(itag, n_tot), // perm
																		SR::with_len(itag, n_tot), // etree
																		SR::with_len(itag, n_tot), // ldl nnz counts
																		SR::with_len(itag, lnnz), // ldl row indices
																		SR::with_len(xtag, lnnz), // ldl values
																})
															: PROX_QP_ALL_OF({
																		SR::with_len(itag, 0),
																		SR::with_len(xtag, 0),
																}),
											iter_req,
									}),
							}),
					});

			storage.resize_for_overwrite(req.alloc_req());

			// preconditioner
			auto kkt = data.kkt_mut();
			auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n);

			linearsolver::sparse::MatMut<T, I> H_scaled =
					detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);

			linearsolver::sparse::MatMut<T, I> AT_scaled =
					detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, data.A_nnz);

			linearsolver::sparse::MatMut<T, I> CT_scaled =
					detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, data.C_nnz);

			g_scaled = data.g;
			b_scaled = data.b;
			l_scaled = data.l;
			u_scaled = data.u;

			QpViewMut<T, I> qp_scaled = {
					H_scaled,
					{linearsolver::sparse::from_eigen, g_scaled},
					AT_scaled,
					{linearsolver::sparse::from_eigen, b_scaled},
					CT_scaled,
					{linearsolver::sparse::from_eigen, l_scaled},
					{linearsolver::sparse::from_eigen, u_scaled},
			};
			stack = stack_mut();
			precond.scale_qp_in_place(qp_scaled, stack);

			// initial factorization
			kkt_nnz_counts.resize_for_overwrite(n_tot);

			// H and A are always active
			for (usize j = 0; j < usize(n + n_eq); ++j) {
				kkt_nnz_counts[isize(j)] = I(kkt.col_end(j) - kkt.col_start(j));
			}
			// ineq constraints initially inactive
			for (isize j = 0; j < n_in; ++j) {
				kkt_nnz_counts[n + n_eq + j] = 0;
			}

			linearsolver::sparse::MatMut<T, I> kkt_active = {
					linearsolver::sparse::from_raw_parts,
					n_tot,
					n_tot,
					data.H_nnz + data.A_nnz,
					kkt.col_ptrs_mut(),
					kkt_nnz_counts.ptr_mut(),
					kkt.row_indices_mut(),
					kkt.values_mut(),
			};

			using MatrixFreeSolver = Eigen::MINRES<
					detail::AugmentedKkt<T, I>,
					Eigen::Upper | Eigen::Lower,
					Eigen::IdentityPreconditioner>;
			matrix_free_solver = std::unique_ptr<MatrixFreeSolver>{
					new MatrixFreeSolver,
			};
			matrix_free_kkt = std::unique_ptr<detail::AugmentedKkt<T, I>>{
					new detail::AugmentedKkt<T, I>{
							{
									kkt_active.as_const(),
									{},
									n,
									n_eq,
									n_in,
									{},
									{},
									{},
							},
					}};

			auto zx = linearsolver::sparse::util::zero_extend;
			auto max_lnnz = isize(zx(ldl_col_ptrs[n_tot]));
			isize ldlt_ntot = do_ldlt ? n_tot : 0;
			isize ldlt_lnnz = do_ldlt ? max_lnnz : 0;

			ldl_nnz_counts.resize_for_overwrite(ldlt_ntot);
			ldl_row_indices.resize_for_overwrite(ldlt_lnnz);
			ldl_values.resize_for_overwrite(ldlt_lnnz);

			auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
			I* perm = _perm.ptr_mut();
			if (do_ldlt) {
				// compute perm from perm_inv
				for (isize i = 0; i < n_tot; ++i) {
					perm[isize(zx(perm_inv[i]))] = I(i);
				}
			}

			linearsolver::sparse::MatMut<T, I> ldl = {
					linearsolver::sparse::from_raw_parts,
					n_tot,
					n_tot,
					0,
					ldl_col_ptrs.ptr_mut(),
					do_ldlt ? ldl_nnz_counts.ptr_mut() : nullptr,
					ldl_row_indices.ptr_mut(),
					ldl_values.ptr_mut(),
			};

			refactorize(
					results,
					do_ldlt,
					n_tot,
					kkt_active,
					results.active_constraints.as_mut(),
					*matrix_free_solver.get(),
					data,
					etree.ptr_mut(),
					ldl_nnz_counts.ptr_mut(),
					ldl_row_indices.ptr_mut(),
					perm_inv.ptr_mut(),
					ldl_values.ptr_mut(),
					perm,
					ldl_col_ptrs.ptr_mut(),
					stack,
					ldl,
					*matrix_free_kkt.get(),
					xtag);
		}
	} _;

	Workspace() = default;

	auto ldl_col_ptrs() const -> I const* {
		return _.ldl_col_ptrs.ptr();
	}
	auto ldl_col_ptrs_mut() -> I* {
		return _.ldl_col_ptrs.ptr_mut();
	}
	auto stack_mut() -> veg::dynstack::DynStackMut {
		return _.stack_mut();
	}
};

} //namespace sparse
} //namespace qp
} //namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WORKSPACE_HPP */
