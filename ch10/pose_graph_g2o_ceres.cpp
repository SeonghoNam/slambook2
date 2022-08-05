#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include <sophus/se3.hpp>
#include "ceres_SE3parmterization.h"
#include "PoseGraph3dError.h"

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity();    // try Identity if you want
    return J;
}

class SE3PoseFactorAuto 
{
    public:
    SE3PoseFactorAuto(int id, int v1, int v2):_id(id),_v1(v1),_v2(v2)
    {
    }

    bool read(istream &is) {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        _measure = SE3d(q, Vector3d(data[0], data[1], data[2]));
        for (int i = 0; i < _cov.rows() && is.good(); i++)
            for (int j = i; j < _cov.cols() && is.good(); j++) {
                is >> _cov(i, j);
                if (i != j)
                    _cov(j, i) = _cov(i, j);
            }
        return true;
    }

    bool write(ostream &os) const {
        os << _v1 << " " << _v2 << " ";
        SE3d m = _measure;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix 
        for (int i = 0; i < _cov.rows(); i++)
            for (int j = i; j < _cov.cols(); j++) {
                os << _cov(i, j) << " ";
            }
        os << endl;
        return true;
    }

    template <typename T>
    bool operator()(const T* const SE3a_ptr,
                    const T* const SE3b_ptr,
                    T* residuals_ptr) const {
        Eigen::Map<Sophus::SE3<T> const> T_prv(SE3a_ptr);
        Eigen::Map<Sophus::SE3<T> const> T_cur(SE3b_ptr);
        Eigen::Map<Vector<T,6>> error(residuals_ptr);
        Eigen::Map<Sophus::SE3<T> const> T_measure((T*)_measure.data());

        error = (_measure.inverse() * T_cur.inverse() * T_prv).log();
        error.applyOnTheLeft(_cov.template cast<T>());
        return true;
    }
private:
    int _id,_v1,_v2;
    SE3d _measure;
    Matrix6d _cov;

};
class SE3PoseFactor : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    SE3PoseFactor(int id, int v1, int v2):_id(id),_v1(v1),_v2(v2)
    {
    }

    bool read(istream &is) {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        _measure = SE3d(q, Vector3d(data[0], data[1], data[2]));
        for (int i = 0; i < _cov.rows() && is.good(); i++)
            for (int j = i; j < _cov.cols() && is.good(); j++) {
                is >> _cov(i, j);
                if (i != j)
                    _cov(j, i) = _cov(i, j);
            }
        return true;
    }

    virtual bool write(ostream &os) const {
        os << _v1 << " " << _v2 << " ";
        SE3d m = _measure;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix 
        for (int i = 0; i < _cov.rows(); i++)
            for (int j = i; j < _cov.cols(); j++) {
                os << _cov(i, j) << " ";
            }
        os << endl;
        return true;
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Eigen::Map<SE3d const> T_prv(parameters[0]);
        Eigen::Map<SE3d const> T_cur(parameters[1]);
        Eigen::Map<Vector6d> error(residuals);

        Eigen::LLT<Matrix6d> llt(_cov);
        Matrix6d sqrt_information = llt.matrixU();
        error = (_measure.inverse() * T_cur.inverse() * T_prv).log();
        if (jacobians)
        {
            Matrix6d J = JRInv(SE3d::exp(error));
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobianXi(jacobians[0]);
                // jacobianXi = -J * T_cur.inverse().Adj();
                // 尝试把J近似为I？
                jacobianXi.setZero();
                jacobianXi.block<6,6>(0,0) = -J * T_cur.inverse().Adj();
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobianXj(jacobians[1]);
                // jacobianXj = J * T_cur.inverse().Adj();
                jacobianXj.setZero();
                jacobianXj.block<6,6>(0,0) = J * T_cur.inverse().Adj();
            }
        }
        error.applyOnTheLeft(sqrt_information);

        return true;
    }
private:
    int _id,_v1,_v2;
    SE3d _measure;
    Matrix6d _cov;

};

#define MY_MODEL 0
int main(int argc, char **argv) {
    cout << argc << " " << argv[0] << " " << argv[1] << endl;
    if (argc != 2) {
        cout << "Usage: pose_graph_ceres sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    ceres::Problem problem;
    ceres::Problem problem_q;

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量

    vector<SE3d> vertices;
    vector<SE3PoseFactor *> edges;
    vector<ceres::examples::Pose3d> poses;
    vector<ceres::examples::PoseGraph3dErrorTerm*> constraints;
    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();
    ceres::Manifold* quat_manifold = new ceres::EigenQuaternionManifold;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            int index = 0;
            fin >> index;
            double data[7];
            for (int i = 0; i < 7; i++)
                fin >> data[i];
            Quaterniond q(data[6], data[3], data[4], data[5]);
            q.normalize();
#if MY_MODEL
            vertices.push_back(SE3d(q, Vector3d(data[0], data[1], data[2])));
#else
            ceres::examples::Pose3d pose;
            pose.p = Vector3d(Vector3d(data[0], data[1], data[2]));
            pose.q = q;
            poses.push_back(pose);
#endif

            vertexCnt++;
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;
#if MY_MODEL
            // SE3PoseFactorAuto *e = new SE3PoseFactorAuto(edgeCnt, idx1, idx2);
            // ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<SE3PoseFactorAuto, 6, 7, 7>(e);
            // e->read(fin);
            // problem.AddResidualBlock(cost_func, nullptr, vertices[idx1].data(), vertices[idx2].data());   
            // problem.SetParameterization(vertices[idx1].data(), local_parameterization);         
            // problem.SetParameterization(vertices[idx2].data(), local_parameterization);   
            // edges.push_back(e);


            SE3PoseFactor *e = new SE3PoseFactor(edgeCnt, idx1, idx2);
            e->read(fin);
            problem.AddResidualBlock(e, nullptr, vertices[idx1].data(), vertices[idx2].data());   
            problem.SetParameterization(vertices[idx1].data(), local_parameterization);         
            problem.SetParameterization(vertices[idx2].data(), local_parameterization);   
            edges.push_back(e);
#else
            ceres::examples::PoseGraph3dErrorTerm *err = new ceres::examples::PoseGraph3dErrorTerm(idx1,idx2);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ceres::examples::PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>
            (err);
            err->read(fin);
            problem_q.AddResidualBlock(cost_function, nullptr, poses[idx1].p.data(), poses[idx1].q.coeffs().data(),
                                        poses[idx2].p.data(), poses[idx2].q.coeffs().data());
            problem_q.SetManifold(poses[idx1].q.coeffs().data(), quat_manifold);
            problem_q.SetManifold(poses[idx2].q.coeffs().data(), quat_manifold);
            constraints.push_back(err);
#endif
            edgeCnt++;
        }
        if (!fin.good()) break;
    }

#if MY_MODEL
    problem.SetParameterBlockConstant(vertices[0].data());
#else
    problem_q.SetParameterBlockConstant(poses[0].p.data());
    problem_q.SetParameterBlockConstant(poses[0].q.coeffs().data());
#endif            
    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;

    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;   // 输出到cout
    ceres::Solver::Summary summary;                // 优化信息
#if MY_MODEL
    ceres::Solve(options, &problem,  &summary);
#else
    ceres::Solve(options, &problem_q, &summary);
#endif
    cout << summary.FullReport() << endl;

    cout << "saving optimization results ..." << endl;

#if MY_MODEL
    ofstream fout("result_ceres.g2o");
    vertexCnt = 0;
    for (SE3d v:vertices) {
        fout << "VERTEX_SE3:QUAT ";
        fout << vertexCnt++ << " ";
        Quaterniond q = v.unit_quaternion();
        fout << v.translation().transpose() << " ";
        fout << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
    }
    for (SE3PoseFactor *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
#else
    ofstream fout("result_ceres.g2o");
    vertexCnt = 0;
    for (ceres::examples::Pose3d pose:poses) {
        fout << "VERTEX_SE3:QUAT ";
        fout << vertexCnt++ << " ";
        Quaterniond q = pose.q;
        fout << pose.p.transpose() << " ";
        fout << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
    }
    for (ceres::examples::PoseGraph3dErrorTerm *e:constraints) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
#endif
    return 0;
}
