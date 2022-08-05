#ifndef CERES_SE3PARAMTERIZATION_H
#define CERES_SE3PARAMTERIZATION_H


#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

using Sophus::SE3d;

class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    // SE3 plus operation for Ceres
    //
    // exp(delta)*T
    //
    virtual bool Plus(double const *T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<SE3d const> const T(T_raw);

        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);

        Eigen::Map<Eigen::Vector3d const> delta_phi(delta_raw);
        Eigen::Map<Eigen::Vector3d const> delta_rho(delta_raw+3);
        Eigen::Matrix<double,6,1> delta;

        delta.block<3,1>(0,0) = delta_rho;//translation
        delta.block<3,1>(3,0) = delta_phi;//rotation

        T_plus_delta = SE3d::exp(delta) * T;
        return true;
    }

    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6,Eigen::RowMajor>> jacobian(jacobian_raw);
        jacobian.setZero();
        jacobian.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        jacobian.block<3,3>(4,3) = Eigen::Matrix3d::Identity();
        return true;
    }

    virtual int GlobalSize() const { return SE3d::num_parameters; }
    virtual int LocalSize() const {return SE3d::DoF; }
};


#endif // easy_slam_SE3_PARAMETERIZATION_H