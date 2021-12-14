#ifndef INRIA_LDLT_QPWORKSPACE_HPP_VCVSK3EOS
#define INRIA_LDLT_QPWORKSPACE_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include <qp/precond/ruiz.hpp>

namespace qp {

template <typename T>
struct Qpworkspace {
public:
	static constexpr auto DYN = Eigen::Dynamic;
    enum { layout = Eigen::RowMajor };

	using ColMat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	using VecMap = Eigen::Map<Vec const>;
	using VecMapMut = Eigen::Map<Vec>;

    using MatMap = Eigen::Map<ColMat const>;
	using MatMapMut = Eigen::Map<ColMat>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
    using VecISize = Eigen::Matrix<isize, DYN, 1>;

    using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
    using VecBool = Eigen::Matrix<bool, DYN, 1>;

    ///// Equilibrator
    qp::preconditioner::RuizEquilibration<T> _ruiz;

    ///// Cholesky Factorization
    ldlt::Ldlt<T> _ldl;

    ///// QP STORAGE
    ColMat _h_scaled;
    Vec _g_scaled;
    ColMat _a_scaled;
    ColMat _c_scaled;
    Vec _b_scaled;
    Vec _u_scaled;
    Vec _l_scaled;

    ///// Initial variable loading

    Vec _xe;
    Vec _ye;
    Vec _ze;

    ///// KKT system storage
    ColMat _kkt;

    //// Active set & permutation vector 
    VecISize _current_bijection_map;
    VecISize _new_bijection_map;

    VecBool _l_active_set_n_u;
    VecBool _l_active_set_n_l;
    VecBool _active_inequalities;

    //// First order residuals for line search
    
	Vec _d_dual_for_eq;
	Vec _Cdx;
	Vec _d_primal_residual_eq;

    Vec _residual_in_z_u_plus_alpha;
    Vec _residual_in_z_l_plus_alpha;

    Vec _active_part_z;
    std::vector<T> alphas;

    ///// Newton variables
    Vec _dw_aug;
    Vec _rhs;
    Vec _err;
    
    //// Residuals storage

    Vec _primal_residual_eq_scaled;
	Vec _primal_residual_in_scaled_u;
	Vec _primal_residual_in_scaled_l;

    Vec _Hx;
    Vec _ATy;
    Vec _CTz;
	Vec _dual_residual_scaled; // used for storing stores Hx + g + ATy + CTz for ex

    //// Relative residuals constants

    T _primal_feasibility_rhs_1_eq;
    T _primal_feasibility_rhs_1_in_u;
    T _primal_feasibility_rhs_1_in_l;
    T _dual_feasibility_rhs_2;
    T _correction_guess_rhs_g;
    T _alpha;

	Qpworkspace( isize dim=0, isize n_eq=0, isize n_in=0)
			: //
                _ruiz(qp::preconditioner::RuizEquilibration<T>{dim,n_eq + n_in}),
                //_ldl(ldlt::reserve_uninit, dim+n_eq,dim+n_eq+n_in), new vesion for debug
                _ldl(ldlt::reserve_uninit, dim+n_eq), // old version with alloc
                _h_scaled(dim, dim),
				_g_scaled(dim),
				_a_scaled(n_eq,dim),
				_c_scaled(n_in,dim),
                _b_scaled(n_eq),
                _u_scaled(n_in),
                _l_scaled(n_in),
                _xe(dim),
                _ye(n_eq),
                _ze(n_in),
                _kkt(dim+n_eq,dim+n_eq),
                _current_bijection_map(n_in),
                _new_bijection_map(n_in),
                _l_active_set_n_u(n_in),
                _l_active_set_n_l(n_in),
                _active_inequalities(n_in),
                _d_dual_for_eq(dim),
                _Cdx(n_in),
                _d_primal_residual_eq(n_eq),
                _residual_in_z_u_plus_alpha(n_in),
                _residual_in_z_l_plus_alpha(n_in),
                _active_part_z(n_in),
                _dw_aug(dim+n_eq+n_in),
                _rhs(dim+n_eq+n_in),
                _err(dim+n_eq+n_in),
                _primal_residual_eq_scaled(n_eq),
                _primal_residual_in_scaled_u(n_in),
                _primal_residual_in_scaled_l(n_in),
                _Hx(dim),
                _ATy(dim),
                _CTz(dim),
                _dual_residual_scaled(dim)
            {
                    alphas.reserve( 3*n_in );
                    _h_scaled.setZero();
                    _g_scaled.setZero();
                    _a_scaled.setZero();
                    _c_scaled.setZero();
                    _b_scaled.setZero();
                    _u_scaled.setZero();
                    _l_scaled.setZero();
                    _xe.setZero();
                    _ye.setZero();
                    _ze.setZero();
                    _kkt.setZero();
                    // start with identity permutation
                    //_current_bijection_map = Eigen::VectorXi::LinSpaced(n_in,0,n_in-1); Deprecated in eigen ??? https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#a1c6d1dbfeb9f6491173a83eb44e14c1d
                    //_new_bijection_map = _current_bijection_map;
                    for (isize i = 0; i < n_in; i++) {
                        _current_bijection_map(i) = i;
                        _new_bijection_map(i) = i;
                    }
                    _d_dual_for_eq.setZero();
                    _Cdx.setZero();
                    _d_primal_residual_eq.setZero();
                    _residual_in_z_u_plus_alpha.setZero();
                    _residual_in_z_l_plus_alpha.setZero();
                    _active_part_z.setZero();
                    _dw_aug.setZero();
                    _rhs.setZero();
                    _err.setZero();
                    _primal_residual_eq_scaled.setZero();
                    _primal_residual_in_scaled_u.setZero();
                    _primal_residual_in_scaled_l.setZero();
                    _Hx.setZero();
                    _ATy.setZero();
                    _CTz.setZero();
                    _dual_residual_scaled.setZero();

                    _primal_feasibility_rhs_1_eq = 0;
                    _primal_feasibility_rhs_1_in_u = 0;
                    _primal_feasibility_rhs_1_in_l = 0;
                    _dual_feasibility_rhs_2 = 0;
                    _correction_guess_rhs_g = 0;
                    _alpha = 0.;
                    
            }

        
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPWORKSPACE_HPP_VCVSK3EOS */