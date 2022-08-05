// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)
#ifndef EXAMPLES_CERES_POSE_GRAPH_3D_ERROR_TERM_H_
#define EXAMPLES_CERES_POSE_GRAPH_3D_ERROR_TERM_H_
#include <iostream>
#include <fstream>
#include <utility>
#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"

using Eigen::Vector3d;
using Eigen::Quaterniond;
namespace ceres::examples {
// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement. We have two poses x_a
// and x_b. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the poses and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position, p. The quaternion ordering is
// [x, y, z, w].
// The estimated measurement is:
//      t_ab = [ p_ab ]  = [ R(q_a)^T * (p_b - p_a) ]
//             [ q_ab ]    [ q_a^{-1] * q_b         ]
//
// where ^{-1} denotes the inverse and R(q) is the rotation matrix for the
// quaternion. Now we can compute an error metric between the estimated and
// measurement transformation. For the orientation error, we will use the
// standard multiplicative error resulting in:
//
//   error = [ p_ab - \hat{p}_ab                 ]
//           [ 2.0 * Vec(q_ab * \hat{q}_ab^{-1}) ]
//
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance.
struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  // The name of the data type in the g2o file format.
  static std::string name() { return "VERTEX_SE3:QUAT"; }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class PoseGraph3dErrorTerm
{
 public:
  PoseGraph3dErrorTerm(int v1, int v2)
      : _v1(v1),
        _v2(v2) {}
  template <typename T>
  bool operator()(const T* const p_a_ptr,
                  const T* const q_a_ptr,
                  const T* const p_b_ptr,
                  const T* const q_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);
    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;
    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);
    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();
    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }
  static ceres::CostFunction* Create(const int v1, const int v2){
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
        new PoseGraph3dErrorTerm(v1, v2));
  }

      bool read(std::istream &is) {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        t_ab_measured_.q = q;
        t_ab_measured_.p = Vector3d(data[0], data[1], data[2]);
        for (int i = 0; i < sqrt_information_.rows() && is.good(); i++)
            for (int j = i; j < sqrt_information_.cols() && is.good(); j++) {
                is >> sqrt_information_(i, j);
                if (i != j)
                    sqrt_information_(j, i) = sqrt_information_(i, j);
            }
        return true;
    }

    virtual bool write(std::ostream &os) const {
        os << _v1 << " " << _v2 << " ";
        Eigen::Quaterniond q = t_ab_measured_.q;
        os << t_ab_measured_.p.transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix 
        for (int i = 0; i < sqrt_information_.rows(); i++)
            for (int j = i; j < sqrt_information_.cols(); j++) {
                os << sqrt_information_(i, j) << " ";
            }
        os << std::endl;
        return true;
    }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int _v1, _v2;
  // The measurement for the position of B relative to A in the A frame.
  Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  Eigen::Matrix<double, 6, 6> sqrt_information_;
};
}  // namespace ceres::examples
#endif  // EXAMPLES_CERES_POSE_GRAPH_3D_ERROR_TERM_H_