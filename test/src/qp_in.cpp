#include <doctest.h>
#include <util.hpp>
#include <matio.h>
#include <iostream>
#include <Eigen/SparseCore>
#include <qp/proxqp/solver.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>

using namespace qp;
using T = double;

#define MAROS_MESZAROS_DIR PROBLEM_PATH "/data/maros_meszaros_data/"

char const* files[] = {
		MAROS_MESZAROS_DIR "AUG2D.mat",    MAROS_MESZAROS_DIR "AUG2DC.mat",
		MAROS_MESZAROS_DIR "AUG2DCQP.mat", MAROS_MESZAROS_DIR "AUG2DQP.mat",
		MAROS_MESZAROS_DIR "AUG3D.mat",    MAROS_MESZAROS_DIR "AUG3DC.mat",
		MAROS_MESZAROS_DIR "AUG3DCQP.mat", MAROS_MESZAROS_DIR "AUG3DQP.mat",
		MAROS_MESZAROS_DIR "BOYD1.mat",    MAROS_MESZAROS_DIR "BOYD2.mat",
		MAROS_MESZAROS_DIR "CONT-050.mat", MAROS_MESZAROS_DIR "CONT-100.mat",
		MAROS_MESZAROS_DIR "CONT-101.mat", MAROS_MESZAROS_DIR "CONT-200.mat",
		MAROS_MESZAROS_DIR "CONT-201.mat", MAROS_MESZAROS_DIR "CONT-300.mat",
		MAROS_MESZAROS_DIR "CVXQP1_L.mat", MAROS_MESZAROS_DIR "CVXQP1_M.mat",
		MAROS_MESZAROS_DIR "CVXQP1_S.mat", MAROS_MESZAROS_DIR "CVXQP2_L.mat",
		MAROS_MESZAROS_DIR "CVXQP2_M.mat", MAROS_MESZAROS_DIR "CVXQP2_S.mat",
		MAROS_MESZAROS_DIR "CVXQP3_L.mat", MAROS_MESZAROS_DIR "CVXQP3_M.mat",
		MAROS_MESZAROS_DIR "CVXQP3_S.mat", MAROS_MESZAROS_DIR "DPKLO1.mat",
		MAROS_MESZAROS_DIR "DTOC3.mat",    MAROS_MESZAROS_DIR "DUAL1.mat",
		MAROS_MESZAROS_DIR "DUAL2.mat",    MAROS_MESZAROS_DIR "DUAL3.mat",
		MAROS_MESZAROS_DIR "DUAL4.mat",    MAROS_MESZAROS_DIR "DUALC1.mat",
		MAROS_MESZAROS_DIR "DUALC2.mat",   MAROS_MESZAROS_DIR "DUALC5.mat",
		MAROS_MESZAROS_DIR "DUALC8.mat",   MAROS_MESZAROS_DIR "EXDATA.mat",
		MAROS_MESZAROS_DIR "GENHS28.mat",  MAROS_MESZAROS_DIR "GOULDQP2.mat",
		MAROS_MESZAROS_DIR "GOULDQP3.mat", MAROS_MESZAROS_DIR "HS118.mat",
		MAROS_MESZAROS_DIR "HS21.mat",     MAROS_MESZAROS_DIR "HS268.mat",
		MAROS_MESZAROS_DIR "HS35.mat",     MAROS_MESZAROS_DIR "HS35MOD.mat",
		MAROS_MESZAROS_DIR "HS51.mat",     MAROS_MESZAROS_DIR "HS52.mat",
		MAROS_MESZAROS_DIR "HS53.mat",     MAROS_MESZAROS_DIR "HS76.mat",
		MAROS_MESZAROS_DIR "HUES-MOD.mat", MAROS_MESZAROS_DIR "HUESTIS.mat",
		MAROS_MESZAROS_DIR "KSIP.mat",     MAROS_MESZAROS_DIR "LASER.mat",
		MAROS_MESZAROS_DIR "LISWET1.mat",  MAROS_MESZAROS_DIR "LISWET10.mat",
		MAROS_MESZAROS_DIR "LISWET11.mat", MAROS_MESZAROS_DIR "LISWET12.mat",
		MAROS_MESZAROS_DIR "LISWET2.mat",  MAROS_MESZAROS_DIR "LISWET3.mat",
		MAROS_MESZAROS_DIR "LISWET4.mat",  MAROS_MESZAROS_DIR "LISWET5.mat",
		MAROS_MESZAROS_DIR "LISWET6.mat",  MAROS_MESZAROS_DIR "LISWET7.mat",
		MAROS_MESZAROS_DIR "LISWET8.mat",  MAROS_MESZAROS_DIR "LISWET9.mat",
		MAROS_MESZAROS_DIR "LOTSCHD.mat",  MAROS_MESZAROS_DIR "MOSARQP1.mat",
		MAROS_MESZAROS_DIR "MOSARQP2.mat", MAROS_MESZAROS_DIR "POWELL20.mat",
		MAROS_MESZAROS_DIR "PRIMAL1.mat",  MAROS_MESZAROS_DIR "PRIMAL2.mat",
		MAROS_MESZAROS_DIR "PRIMAL3.mat",  MAROS_MESZAROS_DIR "PRIMAL4.mat",
		MAROS_MESZAROS_DIR "PRIMALC1.mat", MAROS_MESZAROS_DIR "PRIMALC2.mat",
		MAROS_MESZAROS_DIR "PRIMALC5.mat", MAROS_MESZAROS_DIR "PRIMALC8.mat",
		MAROS_MESZAROS_DIR "Q25FV47.mat",  MAROS_MESZAROS_DIR "QADLITTL.mat",
		MAROS_MESZAROS_DIR "QAFIRO.mat",   MAROS_MESZAROS_DIR "QBANDM.mat",
		MAROS_MESZAROS_DIR "QBEACONF.mat", MAROS_MESZAROS_DIR "QBORE3D.mat",
		MAROS_MESZAROS_DIR "QBRANDY.mat",  MAROS_MESZAROS_DIR "QCAPRI.mat",
		MAROS_MESZAROS_DIR "QE226.mat",    MAROS_MESZAROS_DIR "QETAMACR.mat",
		MAROS_MESZAROS_DIR "QFFFFF80.mat", MAROS_MESZAROS_DIR "QFORPLAN.mat",
		MAROS_MESZAROS_DIR "QGFRDXPN.mat", MAROS_MESZAROS_DIR "QGROW15.mat",
		MAROS_MESZAROS_DIR "QGROW22.mat",  MAROS_MESZAROS_DIR "QGROW7.mat",
		MAROS_MESZAROS_DIR "QISRAEL.mat",  MAROS_MESZAROS_DIR "QPCBLEND.mat",
		MAROS_MESZAROS_DIR "QPCBOEI1.mat", MAROS_MESZAROS_DIR "QPCBOEI2.mat",
		MAROS_MESZAROS_DIR "QPCSTAIR.mat", MAROS_MESZAROS_DIR "QPILOTNO.mat",
		MAROS_MESZAROS_DIR "QPTEST.mat",   MAROS_MESZAROS_DIR "QRECIPE.mat",
		MAROS_MESZAROS_DIR "QSC205.mat",   MAROS_MESZAROS_DIR "QSCAGR25.mat",
		MAROS_MESZAROS_DIR "QSCAGR7.mat",  MAROS_MESZAROS_DIR "QSCFXM1.mat",
		MAROS_MESZAROS_DIR "QSCFXM2.mat",  MAROS_MESZAROS_DIR "QSCFXM3.mat",
		MAROS_MESZAROS_DIR "QSCORPIO.mat", MAROS_MESZAROS_DIR "QSCRS8.mat",
		MAROS_MESZAROS_DIR "QSCSD1.mat",   MAROS_MESZAROS_DIR "QSCSD6.mat",
		MAROS_MESZAROS_DIR "QSCSD8.mat",   MAROS_MESZAROS_DIR "QSCTAP1.mat",
		MAROS_MESZAROS_DIR "QSCTAP2.mat",  MAROS_MESZAROS_DIR "QSCTAP3.mat",
		MAROS_MESZAROS_DIR "QSEBA.mat",    MAROS_MESZAROS_DIR "QSHARE1B.mat",
		MAROS_MESZAROS_DIR "QSHARE2B.mat", MAROS_MESZAROS_DIR "QSHELL.mat",
		MAROS_MESZAROS_DIR "QSHIP04L.mat", MAROS_MESZAROS_DIR "QSHIP04S.mat",
		MAROS_MESZAROS_DIR "QSHIP08L.mat", MAROS_MESZAROS_DIR "QSHIP08S.mat",
		MAROS_MESZAROS_DIR "QSHIP12L.mat", MAROS_MESZAROS_DIR "QSHIP12S.mat",
		MAROS_MESZAROS_DIR "QSIERRA.mat",  MAROS_MESZAROS_DIR "QSTAIR.mat",
		MAROS_MESZAROS_DIR "QSTANDAT.mat", MAROS_MESZAROS_DIR "S268.mat",
		MAROS_MESZAROS_DIR "STADAT1.mat",  MAROS_MESZAROS_DIR "STADAT2.mat",
		MAROS_MESZAROS_DIR "STADAT3.mat",  MAROS_MESZAROS_DIR "STCQP1.mat",
		MAROS_MESZAROS_DIR "STCQP2.mat",   MAROS_MESZAROS_DIR "TAME.mat",
		MAROS_MESZAROS_DIR "UBH1.mat",     MAROS_MESZAROS_DIR "VALUES.mat",
		MAROS_MESZAROS_DIR "YAO.mat",      MAROS_MESZAROS_DIR "ZECEVIC2.mat",
};

struct MarosMeszarosQp {
	using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, mat_int32_t>;
	using Vec = Eigen::VectorXd;

	std::string filename;

	Mat P;
	Vec q;
	Mat A;
	Vec l;
	Vec u;
};

auto load_qp(char const* filename) -> MarosMeszarosQp {
	using Mat = MarosMeszarosQp::Mat;
	using Vec = MarosMeszarosQp::Vec;

	mat_t* mat_fp = Mat_Open(filename, MAT_ACC_RDONLY);
	VEG_ASSERT(mat_fp != nullptr);
	auto&& _mat_fp_cleanup = veg::defer([&] { Mat_Close(mat_fp); });
	veg::unused(_mat_fp_cleanup);

	auto load_mat = [&](char const* name) -> Mat {
		matvar_t* mat_var = Mat_VarRead(mat_fp, name);
		VEG_ASSERT(mat_var != nullptr);
		auto&& _mat_var_cleanup = veg::defer([&] { Mat_VarFree(mat_var); });
		veg::unused(_mat_var_cleanup);

		VEG_ASSERT(int(mat_var->class_type) == int(matio_classes::MAT_C_SPARSE));
		auto const* ptr = static_cast<mat_sparse_t const*>(mat_var->data);

		isize nrows = isize(mat_var->dims[0]);
		isize ncols = isize(mat_var->dims[1]);

		auto optr = reinterpret_cast<mat_int32_t const*>(ptr->jc); // NOLINT
		auto iptr = reinterpret_cast<mat_int32_t const*>(ptr->ir); // NOLINT
		auto vptr = static_cast<T const*>(ptr->data);              // NOLINT

		Mat out;
		out.resize(nrows, ncols);
		out.reserve(ptr->nzmax);
		for (isize j = 0; j < ncols; ++j) {
			isize col_start = optr[j];
			isize col_end = optr[j + 1];

			for (isize p = col_start; p < col_end; ++p) {

				isize i = iptr[p];
				T v = vptr[p];

				out.insert(i, j) = v;
			}
		}

		return out;
	};

	auto load_vec = [&](char const* name) -> Vec {
		matvar_t* mat_var = Mat_VarRead(mat_fp, name);
		VEG_ASSERT(mat_var != nullptr);
		auto&& _mat_var_cleanup = veg::defer([&] { Mat_VarFree(mat_var); });
		veg::unused(_mat_var_cleanup);

		VEG_ASSERT(int(mat_var->data_type) == int(matio_types::MAT_T_DOUBLE));
		auto const* ptr = static_cast<T const*>(mat_var->data);

		auto view = Eigen::Map<Vec const>{
				ptr,
				long(mat_var->dims[0]),
		};
		return view;
	};

	return {
			filename,
			load_mat("P"),
			load_vec("q"),
			load_mat("A"),
			load_vec("l"),
			load_vec("u"),
	};
}

TEST_CASE("maros meszaros wip") {
	for (auto const* file : files) {
		auto qp = load_qp(file);
		isize n = qp.P.rows();
		isize n_eq = 0;
		isize n_in = qp.A.rows();

		bool skip = n > 1000 || n_in > 1000;
		::fmt::print(
				"path: {}, n: {}, n:_in: {}.{}\n",
				qp.filename,
				n,
				n_in,
				skip ? "skipping" : "");

		if (!skip) {
			QPSettings<T> settings;
			QPData<T> data{n, n_eq, n_in};
			QPResults<T> results{n, n_eq, n_in};
			QPWorkspace<T> work{n, n_eq, n_in};

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A{n_eq, n};
			Eigen::Matrix<T, Eigen::Dynamic, 1> b{n_eq};

			auto H = qp.P.toDense();
			auto C = qp.A.toDense();
			auto g = qp.q;
			auto u = qp.u;
			auto l = qp.l;

			results.x.setZero();
			results.y.setZero();
			results.z.setZero();

			detail::QPsetup_dense<T>(
					H, g, A, b, C, u, l, settings, data, work, results, 1e-9, 0, false);
			detail::qp_solve(settings, data, results, work);
			auto& x = results.x;
			auto& y = results.y;
			auto& z = results.z;
			auto& eps = settings.eps_abs;

			CHECK((H * x + g + A.transpose() * y + C.transpose() * z).norm() < eps);
			CHECK((C * x - l).minCoeff() > -eps);
			CHECK((C * x - u).maxCoeff() < eps);
		}
	}
}
