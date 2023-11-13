#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;

int
main()
{

  srand(static_cast<unsigned>(time(0)));
  int N_smooth = 1000000;

  for (int dim = 2; dim <= 60; dim += 2) {
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> matrix_col_major =
      MatrixXd::Random(dim, dim).eval();
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> matrix_row_major =
      matrix_col_major.transpose();

    VectorXd vector = VectorXd::Random(dim);
    MatrixXd result_col_major(dim, N_smooth);
    MatrixXd result_row_major(dim, N_smooth);

    // Benchmark col-major
    auto start_time_col_major = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_smooth; ++i) {
      result_col_major.col(i).noalias() = matrix_col_major * vector;
    }
    auto end_time_col_major = std::chrono::high_resolution_clock::now();
    auto duration_col_major =
      std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_col_major - start_time_col_major)
        .count();

    // Benchmark row-major
    auto start_time_row_major = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_smooth; ++i) {
      result_row_major.col(i).noalias() = matrix_row_major * vector;
    }
    auto end_time_row_major = std::chrono::high_resolution_clock::now();
    auto duration_row_major =
      std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_row_major - start_time_row_major)
        .count();

    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Col-Major Time: " << duration_col_major << " microseconds"
              << std::endl;
    std::cout << "Row-Major Time: " << duration_row_major << " microseconds"
              << std::endl;
  }
  return 0;
}