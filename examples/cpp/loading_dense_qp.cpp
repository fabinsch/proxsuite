#include "qp/dense/dense.hpp"

using namespace qp;
using T = double;
int
main()
{
  dense::isize dim = 10;
  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  dense::QP<T> Qp(dim, n_eq, n_in);
}
