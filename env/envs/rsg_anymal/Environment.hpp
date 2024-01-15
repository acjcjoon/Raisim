//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//


#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <iostream>
#include <cmath>


namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_+"/test/anymal_b_simple_description/robots/anymal-kinova-collision-wrench.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim();
            gvDim_ = anymal_->getDOF();
            nJoints_ = gvDim_ - 6;

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_),prevTarget_.setZero(gcDim_),prevPrevTarget_.setZero(gcDim_);

            /// this is nominal configuration of anymal
            gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

            /// set pd
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 36;
            actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            double action_std;
            READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
            actionStd_.setConstant(action_std);

            /// Reward coefficients sssss
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
            footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

//            auto RR_footIndex = anymal_->getBodyIdx("LF_SHANK");
//            auto RL_footIndex = anymal_->getBodyIdx("RF_SHANK");
//            auto FR_footIndex = anymal_->getBodyIdx("LH_SHANK");
//            auto FL_footIndex = anymal_->getBodyIdx("RH_SHANK");

//            true_contact.setZero();


            TEEpos_.setZero();
            exforceX.setZero();
            exforceY.setZero();
            exforceZ.setZero();
            basepos_.setZero();


            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(anymal_);
                visual_target = server_->addVisualSphere("visual_target",0.05,1,0,0,0.4);
                external_force = server_->addVisualArrow("visual_force",0.25,0.5,1,0,0);
            }
//            rasim:Vec<3> EE;
//            rasim:Vec<3> BS;
//            rasim:Vec<3> MBS;

//            auto EEFrameIndex_ = anymal_->getFrameIdxByName("kinova_joint_end_effector");
//            auto MBSFrameIndex_ = anymal_->getFrameIdxByName("kinova_joint_jaco_mounting_block");
//            anymal_->getFramePosition(EEFrameIndex_, EE);
//            anymal_->getBasePosition(BS);
//            anymal_->getFramePosition(MBSFrameIndex_, MBS);
        }

        void init() final { }


        void reset() final {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();
            if (visualizable_) {
                Eigen::Vector3d des_pos(0.2*uniDist_(gen_)+2.5,0.2*uniDist_(gen_)-1.5,0.2*uniDist_(gen_)+0.6);
                visual_target->setPosition(des_pos);
                TEEpos_ = visual_target->getPosition();
                external_force->setPosition(basepos_);
//                external_force->setPosition(1,1,1);
            }
            forcetime = 0.0;
            double size =5.0;
            exforceX << uniDist_(gen_),uniDist_(gen_),uniDist_(gen_);
            exforceY << uniDist_(gen_),uniDist_(gen_),uniDist_(gen_);
            exforceZ << uniDist_(gen_),uniDist_(gen_),uniDist_(gen_);
            exforceX = size * exforceX;
            exforceY = size * exforceY;
            exforceZ = size * exforceZ;
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
            /// action scaling
            prevPrevTarget_=  prevTarget_;
            prevTarget_ = pTarget_;
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget_.tail(nJoints_) = pTarget12_;


            anymal_->setPdTarget(pTarget_, vTarget_);


            forcetime += 0.01;
            Eigen::Vector3d exforce3d(findQuadraticFunction(exforceX,forcetime),findQuadraticFunction(exforceY,forcetime),findQuadraticFunction(exforceZ,forcetime));
            Eigen::Quaterniond quaternion;
            quaternion = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), exforce3d.normalized());
            Eigen::Vector4d vector4d = quaternion.coeffs();

            for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();


            if (visualizable_) {
                external_force->setPosition(basepos_);
                external_force->setOrientation(vector4d);
                anymal_->setExternalForce("kinova_link_base_to_kinova_link_base_inertia",exforce3d);
            }

            double heightreward;
//            relaxedLogBarrier(0.1,0.46,0.53,gc_(2),heightreward);
            heightreward= (gc_(2)-0.47)*(gc_(2)-0.47);

            Eigen::VectorXd jointPosTemp(12), jointPosWeight(12);
            jointPosWeight << 1.0, 0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.;
            jointPosTemp = gc_.tail(nJoints_) - gc_init_.tail(nJoints_);
            jointPosTemp = jointPosWeight.cwiseProduct(jointPosTemp.eval());
//      rewards_.record("footSlip", footSlip_.sum());

            rewards_.record("basepos", std::exp(-baseError_.head(2).norm()-0.6));
            rewards_.record("Height", heightreward);


            rewards_.record("Lsmoothness1",(pTarget_.tail(nJoints_) - prevTarget_.tail(nJoints_)).squaredNorm());
            rewards_.record("Jsmoothness1",(pTarget_.tail(6) - prevTarget_.tail(6)).squaredNorm());
            rewards_.record("smoothness2", (pTarget_ - 2 * prevTarget_ + prevPrevTarget_).squaredNorm());
            rewards_.record("jointPos", jointPosTemp.squaredNorm());
            rewards_.record("pTarget", (pTarget_-actionMean_).squaredNorm());
            rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
//    rewards_.record("bodyLinearVel", bodyLinearVel_.norm());
//    rewards_.record("bodyAngularVel", bodyAngularVel_.norm());

//    rewards_.record("torque", Tor.squaredNorm());

//    rewards_.record("forwardvel", std::min(4.0,bodyLinearVel_[0]));
//    std::cout <<"T1: "<<std::exp(-posError_.norm())<<std::endl;
//    std::cout <<"T1: "<<std::min(4.0,bodyLinearVel_[0])<<std::endl;

//    std::cout << posError_.norm() << std::endl;

            return rewards_.sum();
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
            raisim::Vec<4> quat;
            raisim::Mat<3,3> rot;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

            raisim::Vec<3> mbpos;
            auto mbaseindex = anymal_->getFrameIdxByName("kinova_link_base_to_kinova_link_base_inertia");
            anymal_->getFramePosition(mbaseindex, mbpos);
            basepos_ = mbpos.e();
            baseError_ = TEEpos_- basepos_;
            Eigen::Vector3d baseError = rot.e().transpose() * (baseError_);


            obDouble_ << gc_[2], /// body height : 1
                    rot.e().row(2).transpose(), /// body orientation : 3
                    gc_.tail(12), /// joint angles : 12
                    bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity : 6
                    gv_.tail(12),
                    baseError.head(2);
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float& terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            /// if the contact body is not feet
            for(auto& contact: anymal_->getContacts())
                if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
                    return true;

            terminalReward = 0.f;

            return false;
        }

        void curriculumUpdate() { };

        void relaxedLogBarrier(const double& delta,const double& alpha_lower,const double& alpha_upper,const double& x, double& y){
            /// positive reward, boundary 밖에서 gradient 가 큼
            double x_temp = x-alpha_lower;
            // lower bound
            if (x_temp < delta){
                y = 0.5*(pow((x_temp-2*delta)/delta,2)-1) - log(delta);
            }else{
                y = -log(x_temp);
            }
            // upper bound
            x_temp = -(x-alpha_upper);
            if (x_temp < delta){
                y += 0.5*(pow((x_temp-2*delta)/delta,2)-1) - log(delta);
            }else{
                y += -log(x_temp);
            }
            y *= -1;
        }

        double findQuadraticFunction(const Eigen::Vector3d randcoeffi, const double dt){
            Eigen::Matrix3d timemat;
            timemat << 1, 1, 1,
                      4, 2, 1,
                     16, 4, 1;
            Eigen::Vector3d dtvec;
            dtvec << dt*dt, dt, 1;
            double result = dtvec.transpose()*timemat.inverse()*randcoeffi;
            return  result;
        }

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* anymal_;
        raisim::Visuals* visual_target;
        raisim::Visuals* visual_EEpos;
        raisim::Visuals* external_force;

        raisim::Visuals* visual_target2;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_,prevTarget_,prevPrevTarget_, vTarget_,true_contact;
        double terminalRewardCoeff_ = -10.,forcetime;
        raisim::Vec<3> PEEpos_;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, TEEpos_, posError_,baseError_,basepos_,exforceX,exforceY,exforceZ;
        std::set<size_t> footIndices_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        thread_local static std::mt19937 gen_;
        thread_local static std::normal_distribution<double> normDist_;
        thread_local static std::uniform_real_distribution<double> uniDist_;
    };
    thread_local std::mt19937  raisim::ENVIRONMENT::gen_;
    thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
    thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}
