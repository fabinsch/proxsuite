//
// Copyright (c) 2022 INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <proxsuite/proxqp/sparse/wrapper.hpp>
#include <proxsuite/linalg/qdldl/SparseMatrixHelper.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

#include <proxsuite/linalg/qdldl/types.h>
#include <proxsuite/linalg/qdldl/qdldl_types.h>
#include <proxsuite/linalg/qdldl/qdldl_interface.h>

#ifndef EMBEDDED
#include <proxsuite/linalg/qdldl/amd.h>
#endif

#if EMBEDDED != 1
#include <proxsuite/linalg/qdldl/kkt.h>
#endif

  namespace proxsuite {
  namespace linalg {
  namespace qdldl {

  /**
   * Compute LDL factorization of matrix A
   * @param  A    Matrix to be factorized
   * @param  p    Private workspace
   * @param  nvar Number of QP variables
   * @return      exitstatus (0 is good)
   */
  c_int LDL_factor(csc* A, qdldl_solver* p, c_int nvar);

  /* solves P'LDL'P x = b for x */
  void LDLSolve(c_float* x,
                c_float* b,
                const csc* L,
                const c_float* Dinv,
                const c_int* P,
                c_float* bp);

#ifdef __cplusplus
  }
#endif
  }
  }
} // namespace proxsuite::linalg::qdldl

namespace proxsuite {
namespace linalg {
namespace sparse {
namespace python {

typedef Eigen::Array<c_int, -1, 1> ArrayCintXd;
typedef Eigen::Matrix<c_float, -1, -1, Eigen::ColMajor> MatrixCfloatXd;

template<typename T, typename I>
struct SparseQDLDL
{
  qdldl_solver* ldl;
  isize n_tot;

  SparseQDLDL(isize n_tot_)
    : ldl{}
    , n_tot(n_tot_)
  {
    ldl = (qdldl_solver*)c_calloc(1, sizeof(qdldl_solver));

    PROXSUITE_THROW_PRETTY(n_tot == 0,
                           std::invalid_argument,
                           "wrong argument size: the dimension of the "
                           "matrix to factorize should be strictly positive.");
  }
  ~SparseQDLDL() { free_linsys_solver_qdldl(ldl); }
  void factorize(proxsuite::proxqp::sparse::SparseMat<T, I>& mat_eigen)
  {
    c_float* info;
    c_int amd_status;
    csc* mat_temp;

    // Convert input eigen sparse matrix to osqp csc sparse upper triangular
    // matrix
    proxsuite::proxqp::sparse::SparseMat<T, I> mat_ =
      mat_eigen.template triangularView<Eigen::Upper>();
    csc* mat = nullptr;
    if (!OsqpEigen::SparseMatrixHelper::createOsqpSparseMatrix(mat_, mat))
      std::cout << "ERROR converting sparse matrix to csc";

    // Set number of threads to 1 (single threaded)
    ldl->nthreads = 1;

    // Sparse matrix L (lower triangular)
    // NB: We do not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    ldl->L = (csc*)c_malloc(sizeof(csc));
    ldl->L->m = n_tot;
    ldl->L->n = n_tot;
    ldl->L->nz = -1;

    // Diagonal matrix stored as a vector D
    ldl->Dinv = (QDLDL_float*)c_malloc(sizeof(QDLDL_float) * n_tot);
    ldl->D = (QDLDL_float*)c_malloc(sizeof(QDLDL_float) * n_tot);

    // Permutation vector P
    ldl->P = (c_int*)c_malloc(sizeof(QDLDL_int) * n_tot);
    c_int* Pinv;

    // Working vector
    ldl->bp = (QDLDL_float*)c_malloc(sizeof(QDLDL_float) * n_tot);

    // Solution vector
    ldl->sol = (QDLDL_float*)c_malloc(sizeof(QDLDL_float) * n_tot);

    // Elimination tree workspace
    ldl->etree = (QDLDL_int*)c_malloc(n_tot * sizeof(QDLDL_int));
    ldl->Lnz = (QDLDL_int*)c_malloc(n_tot * sizeof(QDLDL_int));

    // Preallocate L matrix (Lx and Li are sparsity dependent)
    ldl->L->p = (c_int*)c_malloc((n_tot + 1) * sizeof(QDLDL_int));

    // Lx and Li are sparsity dependent, so set them to
    // null initially so we don't try to free them prematurely
    ldl->L->i = OSQP_NULL;
    ldl->L->x = OSQP_NULL;

    // Preallocate workspace
    ldl->iwork = (QDLDL_int*)c_malloc(sizeof(QDLDL_int) * (3 * n_tot));
    ldl->bwork = (QDLDL_bool*)c_malloc(sizeof(QDLDL_bool) * n_tot);
    ldl->fwork = (QDLDL_float*)c_malloc(sizeof(QDLDL_float) * n_tot);

    // Compute permutation matrix P using AMD
    info = (c_float*)c_malloc(AMD_INFO * sizeof(c_float));

#ifdef QDLDL_WITH_LONG
    amd_status =
      amd_l_order(mat->n, mat->p, mat->i, ldl->P, (c_float*)OSQP_NULL, info);
#else
    amd_status =
      amd_order(mat->n, mat->p, mat->i, ldl->P, (c_float*)OSQP_NULL, info);
#endif

    if (amd_status < 0) {
      // Free Amd info and return an error
      c_free(info);
      std::cout << "ERROR amd_status:" << amd_status << std::endl;
    }
    // Inverse of the permutation vector
    Pinv = csc_pinv(ldl->P, mat->n);

    // Permute matrix
    mat_temp = csc_symperm(mat, Pinv, OSQP_NULL, 1);

    if (!proxsuite::linalg::qdldl::LDL_factor(mat_temp, ldl, n_tot) == 0)
      std::cout << "ERROR LDL_factorization" << std::endl;

    // Free matrices
    csc_spfree((mat));
    csc_spfree(mat_temp);
    // Free Pinv
    c_free(Pinv);
    // Free Amd info
    c_free(info);
  };

  MatrixCfloatXd l()
  {
    Eigen::SparseMatrix<c_float> L_spa;
    OsqpEigen::SparseMatrixHelper::osqpSparseMatrixToEigenSparseMatrix(ldl->L,
                                                                       L_spa);
    // std::cout << "L_spa: " << L_spa << std::endl;
    MatrixCfloatXd L = MatrixCfloatXd(L_spa);
    return L;
  }

  MatrixCfloatXd lt()
  {
    return SparseQDLDL::l().transpose();
  }

  ArrayCintXd p()
  {
    return Eigen::Map<ArrayCintXd>(ldl->P, n_tot);
  }
  ArrayCintXd pt()
  {
    ArrayCintXd p = SparseQDLDL::p();
    ArrayCintXd pinv = ArrayCintXd(n_tot);
    for (isize k = 0; k < n_tot; k++)
      pinv[p[k]] = k; /* invert the permutation */
    return pinv;
  }
  void solve_in_place(proxsuite::proxqp::sparse::VecRefMut<T> rhs_e)
  {
    proxsuite::linalg::qdldl::LDLSolve(
      ldl->sol, rhs_e.data(), ldl->L, ldl->Dinv, ldl->P, ldl->bp);

    // write back solution the RHS
    for (isize i = 0; i < n_tot; ++i) {
      rhs_e[i] = ldl->sol[i];
    }
  }
};

template<typename T, typename I>
void
exposeSparseQDLDL(pybind11::module_ m)
{

  ::pybind11::class_<SparseQDLDL<T, I>>(
    m, "SparseQDLDL", pybind11::module_local())
    .def(::pybind11::init<proxsuite::linalg::veg::i64>(),
         pybind11::arg_v("n_tot", 0, "dimension of the matrix to factorize."),
         "Constructor for defining sparse QDLDL object.") // constructor
    .def("factorize",
         &SparseQDLDL<T, I>::factorize,
         "Factorizes a sparse symmetric and invertible matrix.")
    .def("solve_in_place",
         &SparseQDLDL<T, I>::solve_in_place,
         "Solve in place a linear system using the sparse factorization.")
    .def("l",
         &SparseQDLDL<T, I>::l,
         "Outputs the lower triangular part of the sparse QDLDL factorization "
         "in dense format.")
    .def("lt",
         &SparseQDLDL<T, I>::lt,
         "Outputs the transpose of the lower triangular part of the sparse "
         "QDLDL factorization "
         "in dense format.")
    .def("p",
         &SparseQDLDL<T, I>::p,
         "Outputs the permutation matrix of the QDLDL factorization "
         "in vector format.")
    .def("pt",
         &SparseQDLDL<T, I>::pt,
         "Outputs the inverse permutation matrix of the QDLDL factorization "
         "in dense format.");
}

} // namespace python
} // namespace sparse

} // namespace linalg
} // namespace proxsuite
