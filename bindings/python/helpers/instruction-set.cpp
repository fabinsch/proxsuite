//
// Copyright (c) 2022 INRIA
//

#include <nanobind/nanobind.h>

#include <proxsuite/helpers/instruction-set.hpp>

namespace proxsuite {
namespace helpers {
namespace python {

NB_MODULE(instructionset, m)
{
  m.doc() = R"pbdoc(
        CPU info library
    ------------------------

    .. currentmodule:: instructionset
    .. autosummary::
        :toctree: _generate

        instructionset
    )pbdoc";

#define PYTHON_EXPOSE_FIELD(field_name)                                        \
  m.def(#field_name, proxsuite::helpers::InstructionSet::field_name)

  PYTHON_EXPOSE_FIELD(vendor);
  PYTHON_EXPOSE_FIELD(brand);

  PYTHON_EXPOSE_FIELD(has_SSE3);
  PYTHON_EXPOSE_FIELD(has_PCLMULQDQ);
  PYTHON_EXPOSE_FIELD(has_MONITOR);
  PYTHON_EXPOSE_FIELD(has_SSSE3);
  PYTHON_EXPOSE_FIELD(has_FMA);
  PYTHON_EXPOSE_FIELD(has_CMPXCHG16B);
  PYTHON_EXPOSE_FIELD(has_SSE41);
  PYTHON_EXPOSE_FIELD(has_SSE42);
  PYTHON_EXPOSE_FIELD(has_MOVBE);
  PYTHON_EXPOSE_FIELD(has_POPCNT);
  PYTHON_EXPOSE_FIELD(has_AES);
  PYTHON_EXPOSE_FIELD(has_XSAVE);
  PYTHON_EXPOSE_FIELD(has_OSXSAVE);
  PYTHON_EXPOSE_FIELD(has_AVX);
  PYTHON_EXPOSE_FIELD(has_F16C);
  PYTHON_EXPOSE_FIELD(has_RDRAND);

  PYTHON_EXPOSE_FIELD(has_MSR);
  PYTHON_EXPOSE_FIELD(has_CX8);
  PYTHON_EXPOSE_FIELD(has_SEP);
  PYTHON_EXPOSE_FIELD(has_CMOV);
  PYTHON_EXPOSE_FIELD(has_CLFSH);
  PYTHON_EXPOSE_FIELD(has_MMX);
  PYTHON_EXPOSE_FIELD(has_FXSR);
  PYTHON_EXPOSE_FIELD(has_SSE);
  PYTHON_EXPOSE_FIELD(has_SSE2);

  PYTHON_EXPOSE_FIELD(has_FSGSBASE);
  PYTHON_EXPOSE_FIELD(has_AVX512VBMI);
  PYTHON_EXPOSE_FIELD(has_BMI1);
  PYTHON_EXPOSE_FIELD(has_HLE);
  PYTHON_EXPOSE_FIELD(has_AVX2);
  PYTHON_EXPOSE_FIELD(has_BMI2);
  PYTHON_EXPOSE_FIELD(has_ERMS);
  PYTHON_EXPOSE_FIELD(has_INVPCID);
  PYTHON_EXPOSE_FIELD(has_RTM);
  PYTHON_EXPOSE_FIELD(has_AVX512F);
  PYTHON_EXPOSE_FIELD(has_AVX512DQ);
  PYTHON_EXPOSE_FIELD(has_ADX);
  PYTHON_EXPOSE_FIELD(has_AVX512IFMA);
  PYTHON_EXPOSE_FIELD(has_AVX512PF);
  PYTHON_EXPOSE_FIELD(has_AVX512ER);
  PYTHON_EXPOSE_FIELD(has_AVX512CD);
  PYTHON_EXPOSE_FIELD(has_SHA);
  PYTHON_EXPOSE_FIELD(has_AVX512BW);
  PYTHON_EXPOSE_FIELD(has_AVX512VL);

  PYTHON_EXPOSE_FIELD(has_PREFETCHWT1);

  PYTHON_EXPOSE_FIELD(has_LAHF);
  PYTHON_EXPOSE_FIELD(has_LZCNT);
  PYTHON_EXPOSE_FIELD(has_ABM);
  PYTHON_EXPOSE_FIELD(has_SSE4a);
  PYTHON_EXPOSE_FIELD(has_XOP);
  PYTHON_EXPOSE_FIELD(has_FMA4);
  PYTHON_EXPOSE_FIELD(has_TBM);

  PYTHON_EXPOSE_FIELD(has_SYSCALL);
  PYTHON_EXPOSE_FIELD(has_MMXEXT);
  PYTHON_EXPOSE_FIELD(has_RDTSCP);
  PYTHON_EXPOSE_FIELD(has_x64);
  PYTHON_EXPOSE_FIELD(has_3DNOWEXT);
  PYTHON_EXPOSE_FIELD(has_3DNOW);

#undef PYTHON_EXPOSE_FIELD
}

} // namespace python

} // namespace proxqp
} // namespace proxsuite
