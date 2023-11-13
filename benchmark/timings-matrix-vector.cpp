#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <x86intrin.h>

using namespace Eigen;

int
main()
{

  srand(static_cast<unsigned>(time(0)));
  int N_smooth = 3000000;
  uint64_t avg_row_major = 0;
  uint64_t avg_col_major = 0;

  for (int dim = 2; dim <= 60; dim += 2) {
    Eigen::MatrixXd random_matrix = MatrixXd::Random(dim, dim);
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> matrix_col_major =
      random_matrix.transpose() + random_matrix;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> matrix_row_major =
      matrix_col_major.transpose();

    VectorXd vector = VectorXd::Random(dim);
    MatrixXd result_col_major(dim, N_smooth);
    MatrixXd result_row_major(dim, N_smooth);

    // Benchmark col-major
    // auto start_time_col_major = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_smooth; ++i) {
      const uint64_t start = __rdtsc();
      result_col_major.col(i).noalias() = matrix_col_major * vector;
      avg_col_major += __rdtsc() - start;
    }
    // auto end_time_col_major = std::chrono::high_resolution_clock::now();
    // auto duration_col_major =
    //   std::chrono::duration_cast<std::chrono::microseconds>(
    //     end_time_col_major - start_time_col_major)
    //     .count();

    // Benchmark row-major
    // auto start_time_row_major = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_smooth; ++i) {
      const uint64_t start = __rdtsc();
      result_row_major.col(i).noalias() = matrix_row_major * vector;
      avg_row_major += __rdtsc() - start;
    }
    // auto end_time_row_major = std::chrono::high_resolution_clock::now();
    // auto duration_row_major =
    //   std::chrono::duration_cast<std::chrono::microseconds>(
    //     end_time_row_major - start_time_row_major)
    //     .count();

    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Col-Major avg cycles: " << avg_col_major / N_smooth
              << std::endl;
    std::cout << "Row-Major avg cycles: " << avg_row_major / N_smooth
              << std::endl;
  }
  return 0;
}