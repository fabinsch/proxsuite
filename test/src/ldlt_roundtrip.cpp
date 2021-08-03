#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <ldlt/ldlt.hpp>
#include <fmt/ostream.h>

using namespace ldlt;

template <typename T>
struct DoNotDeduceImpl {
	using Type = T;
};

template <typename T>
using DoNotDeduce = typename DoNotDeduceImpl<T>::Type;

template <typename T>
using Mat = Eigen::Matrix<T, -1, -1>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <typename T>
auto matmul(Mat<T> const& a, DoNotDeduce<Mat<T>> const& b) -> Mat<T> {
	using Upscaled = typename std::
			conditional<std::is_floating_point<T>::value, long double, T>::type;

	return (a.template cast<Upscaled>() * b.template cast<Upscaled>())
	    .template cast<T>();
}

template <typename T>
auto matmul3(
		Mat<T> const& a, DoNotDeduce<Mat<T>> const& b, DoNotDeduce<Mat<T>> const& c)
		-> Mat<T> {
	return ::matmul(::matmul(a, b), c);
}

using T = f32;

struct Error {
	T eigen;
	T ours;
};

template <typename Fn>
auto ldlt_roundtrip_error(usize n, Fn fn) -> Error {
	Mat<T> mat(n, n);
	Mat<T> l(n, n);
	Vec<T> d(n);
	std::srand(unsigned(n));
	mat.setRandom();
	mat = (mat.transpose() * mat).eval();

	auto m_view = MatrixView<T, colmajor>{mat.data(), n};
	auto l_view = LowerTriangularMatrixViewMut<T, colmajor>{l.data(), n};
	auto d_view = DiagonalMatrixViewMut<T>{d.data(), n};

	fn(l_view, d_view, m_view);

	T ours = (matmul3(l, d.asDiagonal(), l.transpose()) - mat).norm();

	auto ldlt = mat.ldlt();
	auto const& L = ldlt.matrixL();
	auto const& P = ldlt.transpositionsP();
	auto const& D = ldlt.vectorD();

	Mat<T> tmp = P.transpose() * Mat<T>(L);
	T eigen = (matmul3(tmp, D.asDiagonal(), tmp.transpose()) - mat).norm();
	return {
			eigen,
			ours,
	};
}

auto main() -> int {
	for (usize n = 9; n <= 9; ++n) {

		{
			auto err = ::ldlt_roundtrip_error(
					n,
					[](LowerTriangularMatrixViewMut<T, colmajor> l_view,
			       DiagonalMatrixViewMut<T> d_view,
			       MatrixView<T, colmajor> m_view) {
						ldlt::factorize_ldlt_unblocked(l_view, d_view, m_view);
					});

			fmt::print(
					"n = {}, standard: eigen: {}, ours: {}\n", n, err.eigen, err.ours);
		}

		{
			auto err = ::ldlt_roundtrip_error(
					n,
					[](LowerTriangularMatrixViewMut<T, colmajor> l_view,
			       DiagonalMatrixViewMut<T> d_view,
			       MatrixView<T, colmajor> m_view) {
						ldlt::factorize_ldlt_unblocked(
								l_view, d_view, m_view, accumulators::Kahan<T>{});
					});
			fmt::print(
					"n = {}, kahan   : eigen: {}, ours: {}\n", n, err.eigen, err.ours);
		}

		{
			auto err = ::ldlt_roundtrip_error(
					n,
					[](LowerTriangularMatrixViewMut<T, colmajor> l_view,
			       DiagonalMatrixViewMut<T> d_view,
			       MatrixView<T, colmajor> m_view) {
						ldlt::factorize_ldlt_unblocked(
								l_view, d_view, m_view, accumulators::Vectorized<T>{});
					});
			fmt::print(
					"n = {}, simd    : eigen: {}, ours: {}\n", n, err.eigen, err.ours);
		}
	}
}
