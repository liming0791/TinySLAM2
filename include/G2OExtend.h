#ifndef G2OEXTEND_H
#define G2OEXTEND_H


#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

#include <g2o/types/slam3d/se3_ops.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Geometry>

namespace g2o {

    using namespace Eigen;

    typedef Matrix<double, 6, 6> Matrix6d;

    ///**
    // *   * \brief SE3 Vertex parameterized internally with a transformation matrix
    // *     and externally with its exponential map
    // *       */
    //class  VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
    //    public:
    //        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //            VertexSE3Expmap();

    //        bool read(std::istream& is);

    //        bool write(std::ostream& os) const;

    //        virtual void setToOriginImpl() {
    //            _estimate = SE3Quat();
    //        }

    //        virtual void oplusImpl(const double* update_)  {
    //            Eigen::Map<const Vector6d> update(update_);
    //            setEstimate(SE3Quat::exp(update)*estimate());
    //        }
    //};

    class  EdgeSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<2, Eigen::Vector2d, VertexSE3Expmap>{
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                EdgeSE3ProjectXYZOnlyPose(){}

            bool read(std::istream& is);

            bool write(std::ostream& os) const;

            void computeError()  {
                const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
                Eigen::Vector2d obs(_measurement);
                _error = obs-cam_project(v1->estimate().map(Xw));
            }

            bool isDepthPositive() {
                const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
                return (v1->estimate().map(Xw))(2)>0.0;
            }


            virtual void linearizeOplus();

            Eigen::Vector2d cam_project(const Eigen::Vector3d & trans_xyz) const;

            Eigen::Vector3d Xw;
            double fx, fy, cx, cy;
    };

}

#endif
