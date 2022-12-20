//
// Copyright (c) 2022 INRIA
//
/**
 * @file serialize.hpp
 */

#ifndef PROXSUITE_HELPERS_SERIALIZE_HPP
#define PROXSUITE_HELPERS_SERIALIZE_HPP

#include <Eigen/Eigen>
#include <cereal/cereal.hpp>

#include <fstream>
#include <string>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {
template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
save(
  Archive& ar,
  Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const& m)
{
  int32_t rows = m.rows();
  int32_t cols = m.cols();
  ar(rows);
  ar(cols);

  for (int i = 0; i < m.size(); i++)
    ar(m.data()[i]);
}

template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
load(Archive& ar,
     Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m)
{
  int32_t rows;
  int32_t cols;
  ar(rows);
  ar(cols);

  m.resize(rows, cols);

  for (int i = 0; i < m.size(); i++)
    ar(m.data()[i]);
}

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::dense::Model<T>& model)
{
  archive(CEREAL_NVP(model.dim),
          CEREAL_NVP(model.n_eq),
          CEREAL_NVP(model.n_in),
          CEREAL_NVP(model.n_total),
          CEREAL_NVP(model.H),
          CEREAL_NVP(model.g),
          CEREAL_NVP(model.A),
          CEREAL_NVP(model.b),
          CEREAL_NVP(model.C),
          CEREAL_NVP(model.l),
          CEREAL_NVP(model.u));
}

template<class Archive>
void
serialize(Archive& archive, proxsuite::proxqp::SparseBackend& sb)
{
  auto sbi = static_cast<int>(sb);
  archive(cereal::make_nvp("sbi", sbi));
  sb = static_cast<proxsuite::proxqp::SparseBackend>(sbi);
}

} // namespace cereal

namespace proxsuite {
namespace serialization {

///
/// \brief Loads an object from a std::stringstream.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in]  is  string stream constaining the serialized content of the
/// object.
///
template<typename T>
inline void
loadFromStringStream(T& object, std::istringstream& is)
{
  cereal::JSONInputArchive ia(is);
  ia(object);
}

///
/// \brief Saves an object inside a std::stringstream.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in]   object Object in which the loaded data are copied.
/// \param[out]  ss String stream constaining the serialized content of the
/// object.
///
template<typename T>
inline void
saveToStringStream(const T& object, std::stringstream& ss)
{
  cereal::JSONOutputArchive oa(ss);
  oa(object);
}

///
/// \brief Loads an object from a std::string
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in]  str  string constaining the serialized content of the object.
///
template<typename T>
inline void
loadFromString(T& object, const std::string& str)
{
  std::istringstream is(str);
  loadFromStringStream(object, is);
}

///
/// \brief Saves an object inside a std::string
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
///
/// \returns a string  constaining the serialized content of the object.
///
template<typename T>
inline std::string
saveToString(const T& object)
{
  std::stringstream ss;
  saveToStringStream(object, ss);
  return ss.str();
}

///
/// \brief Loads an object from a binary file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
inline void
loadFromBinary(T& object, const std::string& filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  if (ifs) {
    cereal::BinaryInputArchive ia(ifs);
    ia(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Saves an object inside a binary file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
void
saveToBinary(const T& object, const std::string& filename)
{
  std::ofstream ofs(filename.c_str(), std::ios::binary);
  if (ofs) {
    cereal::BinaryOutputArchive oa(ofs);
    oa(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Loads an object from a JSON file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
inline void
loadFromJSON(T& object, const std::string& filename)
{
  std::ifstream ifs(filename.c_str());
  if (ifs) {
    cereal::JSONInputArchive ia(ifs);
    ia(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Saves an object inside a JSON file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
void
saveToJSON(const T& object, const std::string& filename)
{
  std::ofstream ofs(filename.c_str());
  if (ofs) {
    cereal::JSONOutputArchive oa(ofs);
    oa(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}
}
}

#endif /* end of include guard PROXSUITE_HELPERS_SERIALIZE_HPP */
