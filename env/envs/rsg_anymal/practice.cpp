//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "HeightMap.hpp"

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            hound_ = world_->addArticulatedSystem(resourceDir_+"/test/anymal_b_simple_description/robots/anymal-kinova-collision-wrech.urdf");
            hound_->setName("hound");
            hound_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            /// initialize containers
            gc_.setZero(19); gcInit_.setZero(); gcNoise_.setZero();
            gv_.setZero(18); gvInit_.setZero(); gvNoise_.setZero();
            gcDes_.setZero(); gvDes_.setZero();
            pTarget_.setZero(); prevTarget_.setZero(); prevPrevTarget_.setZero(); preJointVel_.setZero();
            jointFrictions_.setZero();

            /// this is nominal configuration of anymal
//            double hip = 0.62;
//            gcInit_ << 0, 0, 0.59, 1.0, 0.0, 0.0, 0.0, 0.0, hip, -2*hip, 0.0, hip, -2*hip, 0.0, hip, -2*hip, 0.0, hip, -2*hip;
            gcInit_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
            gcInit_.segment(3,4).normalize();
            gc_ = gcInit_;

            /// set pd gains
            jointPgain_.setZero(); jointPgain_.tail(12).setConstant(30.0);
            jointDgain_.setZero(); jointDgain_.tail(12).setConstant(1.0);
//    hound_->setPdGains(jointPgain_, jointDgain_);
            hound_->setPdGains(Eigen::Vector<double,18>::Zero(), Eigen::Vector<double,18>::Zero());
            hound_->setGeneralizedForce(Eigen::VectorXd::Zero(18));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 144;
            actionDim_ = 12;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);


            /// action scaling
            actionMean_ = gcInit_.tail(12);
            for (int i=0; i<4; i++){
                actionStd_.segment(i*3,3) << 0.3, 0.3, 0.3;
            }

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// indices of links that should not make contact with ground
            footIndices_.push_back(hound_->getBodyIdx("RH_SHANK"));
            footIndices_.push_back(hound_->getBodyIdx("LH_SHANK"));
            footIndices_.push_back(hound_->getBodyIdx("RF_SHANK"));
            footIndices_.push_back(hound_->getBodyIdx("LF_SHANK"));
            footFrames_.push_back("RH_ADAPTER_TO_FOOT");
            footFrames_.push_back("LH_ADAPTER_TO_FOOT");
            footFrames_.push_back("RF_ADAPTER_TO_FOOT");
            footFrames_.push_back("LF_ADAPTER_TO_FOOT");
            rollJointFrames_.push_back("RH_HAA");
            rollJointFrames_.push_back("LH_HAA");
            rollJointFrames_.push_back("RF_HAA");
            rollJointFrames_.push_back("LF_HAA");

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(hound_);
                arrows_.push_back(server_->addVisualArrow("command_xy",0.1,0.05,0,1,0,1));
                arrows_.push_back(server_->addVisualArrow("command_yaw",0.1,0.05,1,0,0,1));
            }
            visualizationOn_ = false;

            /// set limit for log barrier function
            for (int i=0;i<4;i++){
                limitJointPos_.row(i*3+0) << -0.523599,0.523599; // roll : (-pi/6, pi/6)
                limitJointPos_.row(i*3+1) << 0.4-0.785398,0.4+0.785398; // hip
                limitJointPos_.row(i*3+2) << -0.8-1.04,-0.8+1.04; // knee
            }
//    limitBodyHeight_ << 0.48, 0.68;
            limitBodyHeight_ << 0.23, 0.71;
            limitBaseMotion_ << -0.3,0.3;
            limitJointVel_ << -8,8;
            limitTargetVel_ << -0.4,0.4;
            limitFootContact_ << -0.6,2;
            limitFootClearance_ << -0.08,1.0; // 어차피 desired_foot_clearance 를

            /// initialize
//            command_.setZero();
//            footContact_.setZero();
//            footVel_.resize(4); footPos_.resize(4), rollJointPos_.resize(4), refBodyToFoot_.resize(4);
//            footContactPhase_.setZero();
//            footClearance_.setZero();
//            footSlip_.setZero();
//            standingMode_ = false;
            standingRegulation_ = 0.0;
//            phaseSin_.setZero();
            footObsNoise_.setZero();
            standingSmoothness_ = 1.0;
            heightDifferenceWeight_ = 1.0;
            bodyOriWeight = 0.0;
//            jointPosWeight_.setZero(12);
            footPosWeight_.setZero();
            bodyFrameHeight_.setZero();

            /// initialize history
            jointPosErrorHist_ = std::vector<Eigen::Vector<double,12>>(18,Eigen::Vector<double,12>::Zero()); //수정
            jointVelHist_ = std::vector<Eigen::Vector<double,12>>(18,Eigen::Vector<double,12>::Zero()); //수정
            genForceTargetHist_ = std::vector<Eigen::Vector<double,18>>(3,Eigen::Vector<double,18>::Zero()); //수정
            /// initialize gait
            phase_ = 0.0;
            gait_hz_ = 0.72;

            /// heightMap_ initialization
            heightMap_ = HeightMapSample(world_.get(),0,0.,gen_,uniDist_);
            curriculum_ = 0.0;
            iter_ = 0;
            mu_ = 0.7 + 0.3 * uniDist_(gen_);  // [0.4, 1.0]
            world_->setDefaultMaterial(mu_, 0, 0);

            /// initial body to foot pos
            hound_->setState(gcInit_,gvInit_);
            for(int i = 0; i < 4; i++) {
                hound_->getFramePosition(footFrames_[i], footPos_[i]);
                hound_->getFramePosition(rollJointFrames_[i], rollJointPos_[i]);
                refBodyToFoot_[i] =   footPos_[i] - rollJointPos_[i];
//          std::cout << i<<"th initial body to foot pos : " << refBodyToFoot_[i].e().transpose() << std::endl;
            }
        }

        void init() final { }

        void reset() final {

            jointPgain_.setZero(); jointPgain_.tail(12).setConstant(30.0 + 2.5*uniDist_(gen_));//수정
            jointDgain_.setZero(); jointDgain_.tail(12).setConstant(1.0 + 0.1*uniDist_(gen_));//수정
//    hound_->setPdGains(jointPgain_, jointDgain_);
            /// foot obs noise
            for (int i=0;i<12;i++){
                footObsNoise_(i) = 0.02 * uniDist_(gen_);
            }

            /// with standing mode
            if (uniDist_(gen_) > 0.8) { // 10 %
                standingMode_ = true;
                command_.setZero();
            }else{
                standingMode_ = false;

                double comCurriculum = (double)iter_ * 1.0/1000;
                comCurriculum = (comCurriculum > 1.0) ? 1.0 : comCurriculum; // [0,1.0]
        do {
//            double maxCommand = (iter_ % 4 == 0) ? (1.0 + comCurriculum * 1.0) : (1.0 + comCurriculum * 0.5); // 평지 lin x max 2.0, other 1.5
            double maxCommand = 1.8;
            command_ << maxCommand * uniDist_(gen_), 0.6 * uniDist_(gen_), 0.6 * uniDist_(gen_);     // [lix x max, 0.6, 0.6]
                    command_(0) = (command_(0) < -1.0) ? command_(0)+1.2 : command_(0);           // 뒤로가는 건 max -1.0
//            command_ << maxCommand/2 * (uniDist_(gen_)+1), 0, 0;     // [lix x max, 0.6, 0.6]
      } while (command_.norm() < 0.2);
            }

            mu_ = 0.7 + 0.3 * uniDist_(gen_);
            world_->setDefaultMaterial(mu_, 0, 0);

            /// initialize the pose
            bool reset = !standingMode_;
            if (standingMode_){ reset = (uniDist_(gen_) > 0.0);}

            if(!reset){ /// command -> sudden stop
                gcNoise_ = gc_;
                gvNoise_ = gv_;
                gcNoise_.head(3) = gcInit_.head(3);
            }else{ /// reset
                /// initialize with noise
                gcNoise_ = gcInit_;
                /// rot noise
                if (uniDist_(gen_)>0.2 and !standingMode_){ // 40 % -> 올라가는 거 고정
                    yawNoise_ = 3.141592/2.0;
                    command_.tail(2).setZero();
                    command_(0) = abs(command_(0));
                }else{
                    yawNoise_ = uniDist_(gen_) * 3.141592;
                }
                rotYawNoise_ << cos(yawNoise_),-sin(yawNoise_),0,sin(yawNoise_),cos(yawNoise_),0,0,0,1;
                quat_.coeffs() << uniDist_(gen_)*0.2, uniDist_(gen_)*0.2, 0.0, 1.0; // xyz w
                quat_.normalize();
                rotTotalNoise_ = quat_;
                rotTotalNoise_ = rotTotalNoise_.eval() * rotYawNoise_;
                quat_ = rotTotalNoise_;
                quat_.normalize();
                gcNoise_.segment(3,4) << quat_.coeffs().w(), quat_.coeffs().head(3);
                /// joint noise
                for (int i = 7; i < 19; i++){
                    gcNoise_(i) += uniDist_(gen_) * 0.2 * ((standingMode_)? 2.0 : 1.0);
                }
                /// Generalized Velocities randomization.
                gvNoise_.setZero();
                for (int i = 0; i < 18; i++) {
                    if (i < 3) {
                        gvNoise_(i) = uniDist_(gen_) * 0.5;
                    } else if (i < 6) {
                        gvNoise_(i) = uniDist_(gen_) * 0.5;
                    } else {
                        gvNoise_(i) = uniDist_(gen_) * 2.0;
                    }
                    if (standingMode_) {gvNoise_(i) *= 2.0;}
                }
            }

            /// preventing foot penetration
            hound_->setState(gcNoise_,gvNoise_);
            double heightShift = 1e3;  /// -> 공중에 있는 것도 데리고 옴
            double temp = 0.0;

            for (int i = 0; i < 4; i++){
                hound_->getFramePosition(footFrames_[i], footPos_[i]);
//        temp = footPos_[i](2) - 0.025 - heightMap_->getHeight(footPos_[i](0), footPos_[i](1));
                temp = footPos_[i](2) - heightMap_->getHeight(footPos_[i](0), footPos_[i](1));
                if (temp < heightShift){heightShift = temp;}
            }
            gcNoise_(2) -= heightShift;
            hound_->setState(gcNoise_, gvNoise_);
            updateObservation();

            for (auto& vec : genForceTargetHist_) { vec.setZero(); }
            /// reset (except the standingMode_ -> which preserves previous state for sudden command stop)
            if (reset){
                pTarget_ = gc_.tail(12);
                gcDes_.tail(12) = pTarget_; prevTarget_ = pTarget_; prevPrevTarget_ = pTarget_; preJointVel_.setZero();
                for (auto& vec : jointPosErrorHist_) { vec.setZero(); }
                for (auto& vec : jointVelHist_) { vec.setZero(); }

                if (uniDist_(gen_)<=0.0){
                    phase_ = 0.0;
                }else{
                    phase_ = gait_hz_/2.0;
                }
                footContactPhase_.setZero();
                footClearance_.setZero();
            }

//            /// random joint friction
//            double jFrictionHAA_1 = 0.86 + 0.15 * uniDist_(gen_);
//            double jFrictionHFE_1 = 0.86 + 0.15 * uniDist_(gen_);
//            double jFrictionKFE_1 = 0.93 + 0.3 * (uniDist_(gen_) + 0.5);
//            double jFrictionHAA_2 = 2.24 + 0.15 * uniDist_(gen_);
//            double jFrictionHFE_2 = 1.37 + 0.15 * uniDist_(gen_);
//            double jFrictionKFE_2 = 1.16 + 0.3 * (uniDist_(gen_) + 0.5);
//            double jFrictionHAA_3 = 0.90 + 0.15 * uniDist_(gen_);
//            double jFrictionHFE_3 = 3.47 + 0.15 * uniDist_(gen_);
//            double jFrictionKFE_3 = 3.03 + 0.3 * (uniDist_(gen_) + 0.5);
//            double jFrictionHAA_4 = 0.98 + 0.15 * uniDist_(gen_);
//            double jFrictionHFE_4 = 1.26 + 0.15 * uniDist_(gen_);
//            double jFrictionKFE_4 = 1.35 + 0.3 * (uniDist_(gen_) + 0.5);
//            jointFrictions_ << jFrictionHAA_1, jFrictionHFE_1, jFrictionKFE_1, jFrictionHAA_2, jFrictionHFE_2, jFrictionKFE_2,
//                    jFrictionHAA_3, jFrictionHFE_3, jFrictionKFE_3, jFrictionHAA_4, jFrictionHFE_4, jFrictionKFE_4;
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
            /// action scaling
            pTarget_ = action.cast<double>();
            pTarget_ = pTarget_.cwiseProduct(actionStd_);
            pTarget_ += actionMean_;                                  /// joint P target

            /// simulation
            double avgReward = 0.0;
            barrierReward_ = 0.0;
            for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
                /// compute target torque
                computeTorque();
                hound_->setGeneralizedForce(genForceTargetHist_[0]); /// 2ms delay (torque command in PC -> actual torque in real robot)
                /// simpulation
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
                updateObservation();
                avgReward += getReward();
                barrierReward_+= getLogBarReward();

                if(visualizationOn_){
                    visualizeCommand();
                }
            }

            avgReward /= (control_dt_ / simulation_dt_ + 1e-10);
            barrierReward_ /=(control_dt_ / simulation_dt_ + 1e-10);
            updateHistory();

            return avgReward;
        }

        void computeTorque(){
            genForceTargetHist_.erase(genForceTargetHist_.begin());
            Eigen::Vector<double,18> tempGenForce; tempGenForce.head(6).setZero();
            tempGenForce.tail(12) = jointPgain_.tail(12).cwiseProduct(pTarget_-gc_.tail(12))
                                    + jointDgain_.tail(12).cwiseProduct(-gv_.tail(12));

//            std::cout << "tempGenForce : " << tempGenForce.transpose() << std::endl;
            /// joint friction (static friction, torque 잡아먹는 효과)
//            for (int i = 0; i < 12; i++){
//                double jTorque = tempGenForce.tail(12)(i);
//                jTorque = (jTorque>0) ? std::min(jointFrictions_(i), jTorque) : std::max(-jointFrictions_(i), jTorque);
//                tempGenForce.tail(12)(i) -= jTorque;
//            }
//            std::cout << "afterenForce : " << tempGenForce.transpose() << std::endl;
            genForceTargetHist_.push_back(tempGenForce);
        }

        float getBarrierReward() final {
            return barrierReward_;
        }

        void updateHistory(){
            prevPrevTarget_ = prevTarget_;
            prevTarget_ = pTarget_;

            jointVelHist_.erase(jointVelHist_.begin());
            jointVelHist_.push_back(gv_.tail(12));

            jointPosErrorHist_.erase(jointPosErrorHist_.begin());
            jointPosErrorHist_.push_back(pTarget_ - gc_.tail(12));
        }

        double getReward(){
            standingReward(); /// there is order
            rewards_.record("negSumPos",getNegPosReward());
            return rewards_.getReward("negSumPos");
        }

        void standingReward(){
            /// for standingMode
            if (!standingMode_){
                limitBaseMotion_ << -0.3,0.3;
                standingSmoothness_ = 1.0;
                heightDifferenceWeight_ = 1.0;
                bodyOriWeight = 0.0;
                jointPosWeight_ << 1.0, 0.6,0.6,1.,0.6,0.6,1.,0.6,0.6,1.,0.6,0.6;
                footPosWeight_ << 0.6,1.0,0.4;
            } else {
                limitBaseMotion_ << -0.1,0.1;
                standingSmoothness_ = 2.8;
                heightDifferenceWeight_ = 4.0;
                bodyOriWeight = 1.0;
                jointPosWeight_ << 1.0, 0.8,0.8,1.,0.8,0.8,1.,0.8,0.8,1.,0.8,0.8;
                footPosWeight_ << 1.0,1.0,1.0;
            }
        }

        float getNegPosReward(){
            /// pos reward
            rewards_.record("comAngularVel", std::exp(-5.0 * pow(command_(2) - bodyAngularVel_(2),2)));
            rewards_.record("comLinearVel", std::exp(-5.0 * (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm()));

            /// neg reward
            footSlip_.setZero();
            for (int i=0; i<4; i++){
                if (footContact_(i)){
                    footSlip_(i) = footVel_[i].e().head(2).squaredNorm();
                }
            }

            rewards_.record("footSlip", footSlip_.sum());
            rewards_.record("bodyOri", std::acos(rot_(8)) * std::acos(rot_(8)) * bodyOriWeight);
//      jointPosWeight_ << 1.0, 0.6,0.6,1.,0.6,0.6,1.,0.6,0.6,1.,0.6,0.6;
            rewards_.record("smoothness2", ((pTarget_ - 2 * prevTarget_ + prevPrevTarget_).cwiseProduct(jointPosWeight_)).squaredNorm()  * standingSmoothness_);
            rewards_.record("torque", hound_->getGeneralizedForce().squaredNorm());

            /// joint pos regulation -> not used
            Eigen::VectorXd jointPosTemp(12); jointPosTemp.setZero();
//      jointPosTemp = gc_.tail(12) - gcInit_.tail(12);
//      jointPosTemp = jointPosWeight_.cwiseProduct(jointPosTemp.eval());
            rewards_.record("jointPos", jointPosTemp.squaredNorm());

            /// task space foot pos regulation -> used
            Eigen::Vector3d  tempVec;
            double tempReward = 0.0;
            for(int index_leg = 0; index_leg < 4; index_leg++){
                tempVec = (footPos_[index_leg].e() - rollJointPos_[index_leg].e());
                tempVec = rot_.e().transpose() *  tempVec.eval();
                tempReward += footPosWeight_.cwiseProduct(tempVec-refBodyToFoot_[index_leg].e()).squaredNorm();
            }
            rewards_.record("footPos", tempReward);

            /// body height difference -> used
            bodyFrameHeight_.setZero();
            for (int i=0; i<2; i++) {
                for (int j = 0; j < 2; j++) {
                    int index_leg = i * 2 + j;
                    tempVec = (footPos_[index_leg].e() - rollJointPos_[index_leg].e());
                    tempVec(2) = heightMap_->getHeight(footPos_[index_leg](0), footPos_[index_leg](1)) -
                                 rollJointPos_[index_leg](2);
                    tempVec = rot_.e().transpose() * tempVec.eval();
                    bodyFrameHeight_(i) -= tempVec(2);
                }
            }
            bodyFrameHeight_ = bodyFrameHeight_.eval()/2.0;
            rewards_.record("bodyHeightDifference", pow(bodyFrameHeight_(0)-bodyFrameHeight_(1),2.0) * heightDifferenceWeight_);
//            std::cout << "bodyHeightDifference : " << rewards_.getReward("bodyHeightDifference") << std::endl;

            rewards_.record("jointVel", gv_.tail(12).squaredNorm()  * (double)(standingMode_));                 /// only for standingMode_
            rewards_.record("jointAcc", (gv_.tail(12) - preJointVel_).squaredNorm() * (double)(standingMode_)); /// only for standingMode_

            /// sum
            float posReward, negReward;
            posReward = (float)(rewards_.getReward("comAngularVel") + rewards_.getReward("comLinearVel"));
            negReward = (float)(rewards_.getReward("bodyHeightDifference")+ rewards_.getReward("footPos")+rewards_.getReward("jointPos") + rewards_.getReward("jointVel") + rewards_.getReward("jointAcc") + rewards_.getReward("torque") + rewards_.getReward("footSlip") + rewards_.getReward("smoothness2"));
            rewards_.record("negReward2", negReward); /// only for recording

            return (float)(std::exp(0.2 * negReward) * posReward);
        }

        float getLogBarReward(){
            /// for gait enforcing & foot clearance
            phase_ += simulation_dt_;
            footContactPhase_(0) = sin(phase_/gait_hz_ * 2*3.141592); // RR
            footContactPhase_(1) = footContactPhase_(0); // RL
            footContactPhase_(2) = -footContactPhase_(0); // FR
            footContactPhase_(3) = -footContactPhase_(0); // FL

            phaseSin_(0) = sin(phase_/gait_hz_ * 2*3.141592); // for observation
            phaseSin_(1) = cos(phase_/gait_hz_ * 2*3.141592); // for observation
//
            if (!standingMode_){ /// walking
                /// footContactDouble_ -> limit_foot_contact 에 있도록 (-0.3,3) -> Gait Enforcing (요 -0.3 이 벗어나도 되는 범위)
                for(int i=0; i<4; i++) {
                    if (footContact_(i)) { footContactDouble_(i) = 1.0 * footContactPhase_(i); }
                    else { footContactDouble_(i) = -1.0 * footContactPhase_(i); }
                }
                /// footClearance_ -> limit_foot_clearance 에 있도록 (-0.12,0.12) -> foot 드는 거 enforcing
                double desiredFootZPosition = 0.15;
                for (int i=0; i<4; i++){
                    if (footContactPhase_(i) < -0.6) { /// during swing, 전체시간의 33 %
                        footClearance_(i) =
                                footToTerrain_.segment(i * 5, 5).minCoeff() - desiredFootZPosition; // 대략, 0.17 sec, 0 보다 크거나 같으면 됨 (enforcing clearance)
                    }else{ footClearance_(i) = 0.0; } // max reward (not enforcing clearance)
                }
            } else { /// under standingMode_
                /// standingMode_ 는 zero command 로 부터 유추 가능, command 는 obs 이기 때문에, robot 은 standingMode_인지 아닌지 충분히 알 수 있음
                for (int i=0; i<4; i++){
                    footContactDouble_(i) = 1.0; // around max reward, where this value should go under (-0.3,3)
                    footClearance_(i) = 0.0; // max reward (not enforcing clearance)
                }
            }

            /// compute barrier reward
            double barrierJointPos = 0.0, barrierBodyHeight = 0.0, barrierBaseMotion = 0.0, barrierJointVel = 0.0, barrierTargetVel = 0.0, barrierFootContact = 0.0, barrierFootClearance = 0.0;
            double tempReward = 0.0;
            /// Log Barrier - limit_joint_pos
            for (int i=0;i<4;i++){
                for (int j=0;j<3;j++){
                    int index_joint = i*3+j;
//              relaxedLogBarrier(0.09,limitJointPos_(index_leg,0),limitJointPos_(index_leg,1),gc_(7+index_leg),tempReward);
                    relaxedLogBarrier(0.08,limitJointPos_(index_joint,0),limitJointPos_(index_joint,1),gc_(7+index_joint),tempReward);
                    barrierJointPos += tempReward;
//                              std::cout << index_leg<<" th joint : " << tempReward << std::endl;
                }
            }
            /// Log Barrier - limit_body_height
            for (int i=0; i<2; i++){
                relaxedLogBarrier(0.04,limitBodyHeight_(0),limitBodyHeight_(1),bodyFrameHeight_(i),tempReward);
                barrierBodyHeight += tempReward;
            }

            /// Log Barrier - limit_base_motion
            relaxedLogBarrier(0.2,limitBaseMotion_(0,0),limitBaseMotion_(0,1),bodyLinearVel_(2),tempReward);
            barrierBaseMotion += tempReward;
            for (int i=0;i<2;i++){
                relaxedLogBarrier(0.3,limitBaseMotion_(1,0),limitBaseMotion_(1,1),bodyAngularVel_(i),tempReward);
                barrierBaseMotion += tempReward;
            }
            /// Log Barrier - limit_joint_vel
            for (int i=0;i<12;i++){
                relaxedLogBarrier(2.0,limitJointVel_(0),limitJointVel_(1),gv_(6+i),tempReward);
                barrierJointVel += tempReward;
            }
            /// Log Barrier - limit_target_vel
            relaxedLogBarrier(0.2,limitTargetVel_(0),limitTargetVel_(1),bodyLinearVel_(0)-command_(0),tempReward);
            barrierTargetVel += tempReward;
            relaxedLogBarrier(0.2,limitTargetVel_(0),limitTargetVel_(1),bodyLinearVel_(1)-command_(1),tempReward);
            barrierTargetVel += tempReward;
            relaxedLogBarrier(0.2,limitTargetVel_(0),limitTargetVel_(1),bodyAngularVel_(2)-command_(2),tempReward);
            barrierTargetVel += tempReward;
            /// Log Barrier - limit_foot_contact
            for (int i=0;i<4;i++){
                relaxedLogBarrier(0.1,limitFootContact_(0),limitFootContact_(1),footContactDouble_(i),tempReward);
                barrierFootContact += tempReward;
            }
            /// Log Barrier - limit_foot_clearance
            for (int i=0;i<4;i++){
                relaxedLogBarrier(0.01,limitFootClearance_(0),limitFootClearance_(1),footClearance_(i),tempReward);
                barrierFootClearance += tempReward;
            }

//      if (barrierFootClearance < -40) {
////          std::cout << "barrierJointPos : " <<  barrierJointPos << std::endl;
////          std::cout << "barrierBodyHeight : " <<  barrierBodyHeight << std::endl;
////          std::cout << "barrierBaseMotion : " <<  barrierBaseMotion << std::endl;
////          std::cout << "barrierJointVel : " <<  barrierJointVel << std::endl;
////          std::cout << "barrierTargetVel : " <<  barrierTargetVel << std::endl;
////          std::cout << "barrierFootContact : " <<  barrierFootContact << std::endl;
//          std::cout << "barrierFootClearance : " <<   barrierFootClearance << std::endl;
////                std::cout << "foot clearance : " << footClearance_.transpose() << std::endl;
//      }

            double logClip = -500.0;
            barrierJointPos = fmax(barrierJointPos,logClip);           /// 여기 밖 부분은 gradient 안 받겠다
//      barrierBodyHeight = fmax(barrierBodyHeight,logClip);
//      barrierBaseMotion = fmax(barrierBaseMotion,logClip);
//      barrierJointVel = fmax(barrierJointVel,logClip);
//      barrierTargetVel = fmax(barrierTargetVel,logClip);
//      barrierFootContact = fmax(barrierFootContact,logClip);
//      barrierFootClearance = fmax(barrierFootClearance,logClip);
            rewards_.record("barrierJointPos", barrierJointPos);
            rewards_.record("barrierBodyHeight", barrierBodyHeight);
            rewards_.record("barrierBaseMotion", barrierBaseMotion);
            rewards_.record("barrierJointVel", barrierJointVel);
            rewards_.record("barrierTargetVel", barrierTargetVel);
            rewards_.record("barrierFootContact", barrierFootContact);
            rewards_.record("barrierFootClearance", barrierFootClearance);


            float logBarReward =  (float)(1e-1*(barrierJointPos + barrierBodyHeight + barrierBaseMotion + barrierJointVel + barrierTargetVel + barrierFootContact + barrierFootClearance));
            rewards_.record("relaxedLog", logBarReward); /// relaxed log barrier
            return  logBarReward;
        }

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

        void updateObservation() {
            /// update previous footVel
            preJointVel_ = gv_.tail(12);
            /// update state
            hound_->getState(gc_, gv_);
            raisim::Vec<4> quat;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot_);
            bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);
            for(int i = 0; i < 4; i++) {
                hound_->getFramePosition(footFrames_[i], footPos_[i]);
                hound_->getFrameVelocity(footFrames_[i], footVel_[i]);
                hound_->getFramePosition(rollJointFrames_[i], rollJointPos_[i]);
            }

            /// foot contact update
            footContact_.setZero();
            for(auto& contact: hound_->getContacts()){
                for (size_t i=0; i<4; i++){
                    if(contact.getlocalBodyIndex() == footIndices_[i]){
                        footContact_(i) = 1;
                    }
                }
            }

            /// update foot terrain
            updateFootToTerrain();
        }

        void updateFootToTerrain(){
            Eigen::Matrix<double, 3, 5> sample_point;
            double point = 0.05; /// foot size
            sample_point.col(0) << point, 0.0, 0.0;
            sample_point.col(1) << 0.0, point, 0.0;
            sample_point.col(2) << -point, 0.0, 0.0;
            sample_point.col(3) << 0.0, -point, 0.0;
            sample_point.col(4).setZero();
            for (int i = 0; i < 4; i++) {
                sample_point.col(i) = rot_.e().transpose() * sample_point.col(i).eval();
            }
            Eigen::Matrix<double, 5, 1> temp_foot;
            Eigen::Matrix<double, 3, 1> temp3;
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < 5; i++) {
                    temp3 = footPos_[k].e() + sample_point.col(i);
                    footToTerrain_(5 * k + i) = footPos_[k].e()(2) - heightMap_->getHeight(temp3(0), temp3(1));
                }
            }
            if (abs(footToTerrain_.minCoeff()) >  3.0){ /// print error !!!
                std::cout << "gc_[2] : " << gc_[2] << std::endl;
                std::cout << "height map : " << heightMap_->getHeight(temp3(0), temp3(1)) <<  std::endl;
                std::cout << "Error too big here : " << footToTerrain_.transpose() << std::endl;
            }
        }

        void visualizeCommand(){
            Eigen::Matrix<double,3,3> rot_robot, rot_pitch_90, rot_command;
            Eigen::Quaterniond quaternion;
            Eigen::Vector3d command;
            Eigen::VectorXd gc_head_7(7);
            double theta_command;
            Eigen::Matrix<double,3,1> arrow_pos_offset;

            command = command_;
            gc_head_7 = gc_.head(7);

            quaternion.coeffs() << gc_head_7.tail(3),gc_head_7(3);
            rot_robot = quaternion;

            rot_pitch_90 << 0,0,1,0,1,0,-1,0,0;
            theta_command = -atan2(command(1),command(0));
            rot_command << 1,0,0,0,cos(theta_command),-sin(theta_command),0,sin(theta_command),cos(theta_command);

            arrow_pos_offset << 0,0,0.15;
            arrow_pos_offset = rot_robot * arrow_pos_offset.eval();
            quaternion = rot_robot.eval() * rot_pitch_90 * rot_command;

            arrows_[0]->setCylinderSize(0.2,command.head(2).norm()*0.3);
            arrows_[0]->setPosition(gc_head_7.head(3) + arrow_pos_offset);
            arrows_[0]->setOrientation(quaternion.w(),quaternion.x(),quaternion.y(),quaternion.z());

            arrows_[1]->setCylinderSize(0.2,command(2)*0.3);
            arrows_[1]->setPosition(gc_head_7.head(3) + arrow_pos_offset);
            arrows_[1]->setOrientation(gc_head_7.segment(3,4));
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            if (standingMode_){
//          footContactPhase_.setZero();
                phaseSin_.setZero();
            }
            obDouble_ << rot_.e().row(2).transpose(),                               /// body orientation. 3
                    bodyAngularVel_,                                                      /// body angular velocity. 3
                    gc_.tail(12),                                                      /// joint pos 12
                    gv_.tail(12),                                                      /// joint velocity 12

                    prevTarget_,                                                          /// previous action 12
                    prevPrevTarget_,                                                      /// preprevious action 12
                    jointPosErrorHist_[0], jointPosErrorHist_[6], jointPosErrorHist_[12], /// joint History 36 (0.18, 0.12, 0.6)
                    jointVelHist_[0], jointVelHist_[6], jointVelHist_[12],                /// joint History 36 (0.18, 0.12, 0.6)
                    rot_.e().transpose() * (footPos_[0].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[1].e() - gc_.head(3)),
                    rot_.e().transpose() * (footPos_[2].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[3].e() - gc_.head(3)),
                    /// relative foot position with respect to the body COM, expressed in the body frame 12
                    command_,                                                             /// command 3
//          footContactPhase_.head(2), /// footContactPhase 2
                    phaseSin_, /// phase encoding 2
                    static_cast<double>(standingMode_);  /// standingMode 1

            double noise = 0.0;
            for (int i=0; i<obDim_; i++){
                if (i<3)       {noise = 0.03;}  /// body orientation
                else if(i<6)   {noise = 0.1;}   /// body angular velocity (rad/sec)
                else if(i<18)  {noise = 0.05;}  /// joint pos             (rad)
                else if(i<30)  {noise = 0.5;}   /// joint vel             (rad/sec)
                else if(i<54)  {noise = 0.01;}  /// action related
                else if(i<90)  {noise = 0.0;}   /// action related
                else if(i<126) {noise = 0.1;}  /// vel history
//          else if(i<138) {noise = 0.02;} /// relative foot pos (2 cm)
                else if(i<138) {noise = footObsNoise_(i-126);} /// relative foot pos (2 cm)
                else           {noise = 0.0;}

                obDouble_(i) += uniDist_(gen_) * noise;
            }

            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void valueObserve(Eigen::Ref<EigenVec> ob) final { /// obs + (true) estimated_state
            if (standingMode_){
                phaseSin_.setZero();
            }
            valueObDouble_ << rot_.e().row(2).transpose(),                               /// body orientation. 3
                    bodyAngularVel_,                                                      /// body angular velocity. 3
                    gc_.tail(12),                                                      /// joint pos 12
                    gv_.tail(12),                                                      /// joint velocity 12

                    prevTarget_,                                                          /// previous action 12
                    prevPrevTarget_,                                                      /// preprevious action 12
                    jointPosErrorHist_[0], jointPosErrorHist_[6], jointPosErrorHist_[12], /// joint History 36 (0.18, 0.12, 0.6)
                    jointVelHist_[0], jointVelHist_[6], jointVelHist_[12],                /// joint History 36 (0.18, 0.12, 0.6)
                    rot_.e().transpose() * (footPos_[0].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[1].e() - gc_.head(3)),
                    rot_.e().transpose() * (footPos_[2].e() - gc_.head(3)), rot_.e().transpose() * (footPos_[3].e() - gc_.head(3)),
                    /// relative foot position with respect to the body COM, expressed in the body frame 12
                    command_,                                                             /// command 3
                    phaseSin_, /// phase sin cos 2
                    static_cast<double>(standingMode_),                                   /// standingMode 1

                    bodyLinearVel_,                                                       /// body linear velocity. 3
                    footClearance_,                                                       /// min foot z
                    footContact_.cast<double>();

            /// convert it to float
            ob = valueObDouble_.cast<float>();
        }

        bool isTerminalState(float& terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);


            /// if the contact body is not feet
//    if (iter_>4200 and (iter_%4==2 or iter_%4==3)){
//        for (int i=0; i<4; i++){
//            if (gc_(9+i*3)>-0.0)  {return true;}
//        }
//        if ((pTarget_-actionMean_).squaredNorm() > 1e2)   {return true;}
//    }else{
            for(auto& contact: hound_->getContacts())
                if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end()) {
//            return true;
                }
//    if ((pTarget_-actionMean_).squaredNorm() > 1e2)   {return true;}
//    }

            terminalReward = -0.f;
            return false;
        }

        void curriculumUpdate() {
            /// for each iteration
            iter_ ++;
            if (curriculum_<2.0){
                curriculum_ = (double)iter_ * (1.0/200.0); /// 500 iter -> 1.0
            }else{
                curriculum_ = (double)(iter_-400) * (1.0/1000.0) + 2.7; /// 1500 iter -> 1.0
                curriculum_ = (curriculum_ > 4.7) ? 4.7 : curriculum_;
            }

            world_->removeObject(heightMap_);
            heightMap_ = HeightMapSample(world_.get(),5,curriculum_,gen_,uniDist_);
        }

        void setSeed(int seed) {gen_.seed(seed);}


        /// for tester.py
        void setCommand(Eigen::Vector3d command){
            command_ = command;
            if (command_.norm()<0.2){
                standingMode_ = true;
                command_.setZero();
            }else{
                standingMode_ = false;
            }
        }
        void setTerrain(int type, double curriculum, double mu){
            world_->removeObject(heightMap_);
            heightMap_ = HeightMapSample(world_.get(),type,curriculum,gen_,uniDist_);
            world_->setDefaultMaterial(mu, 0, 0);
        }
        void setInitial(int type){
            if (type == 0){
                hound_->setState(gcInit_,gvInit_);
                double heightShift = 1e2, temp = 0.0;
                for (int i = 0; i < 4; i++){
                    hound_->getFramePosition(footFrames_[i], footPos_[i]);
                    temp = footPos_[i](2) - heightMap_->getHeight(footPos_[i](0), footPos_[i](1));
                    if (temp < heightShift){heightShift = temp;}
                }
                gcNoise_(2) -= heightShift;

                /// reset
                hound_->setState(gcNoise_, gvInit_);
                updateObservation();
            }
        }

    private:
        int gcDim_, gvDim_;
        bool visualizable_ = false;
        double terminalRewardCoeff_ = -10.0;
        raisim::ArticulatedSystem* hound_;

        Eigen::VectorXd gc_, gv_;
        Eigen::Vector<double,19> gcInit_, gcNoise_, gcDes_;
        Eigen::Vector<double,18> gvInit_, gvNoise_, gvDes_;
        Eigen::Vector<double,12> pTarget_, prevTarget_, prevPrevTarget_, preJointVel_, jointFrictions_;
        Eigen::Vector<double,18> jointPgain_, jointDgain_;

        raisim::Mat<3,3> rot_;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_, valueObDouble_, estDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::vector<size_t> footIndices_;
        /// additional
        Eigen::Vector3d command_;                     // vx, vy, w
        std::vector<std::string> footFrames_;
        std::vector<std::string> rollJointFrames_;
        Eigen::Vector4i footContact_;
        std::vector<raisim::Vec<3>> footPos_,footVel_, rollJointPos_, refBodyToFoot_;
        double phase_;
        double gait_hz_;
        Eigen::Matrix<double,4,1> footContactDouble_; // gait
        Eigen::Matrix<double,4,1> footContactPhase_;  // gait hz
        Eigen::Matrix<double,4,1> footClearance_;     // foot clearance
        Eigen::Matrix<double,4,1> footSlip_;     // foot clearance
        Eigen::Matrix<double,20,1> footToTerrain_; // 5 sample point for each footfo
        Eigen::Matrix<double,2,1> phaseSin_;  // sin cos representation of phase
        Eigen::Matrix<double,12,1> footObsNoise_;
        Eigen::Matrix<double,2,1> bodyFrameHeight_; // for front legs and hind legs
        bool standingMode_;
        double standingRegulation_;
        double heightDifferenceWeight_;
        double bodyOriWeight;
        Eigen::VectorXd jointPosWeight_;
        Eigen::Vector3d footPosWeight_;
        /// log barrier function
        Eigen::Matrix<double,12,2> limitJointPos_;
        Eigen::Matrix<double,1,2> limitBodyHeight_;
        Eigen::Matrix<double,1,2> limitBaseMotion_; // z vel, roll,pitch vel
        Eigen::Matrix<double,1,2> limitJointVel_;
        Eigen::Matrix<double,1,2> limitTargetVel_;
        Eigen::Matrix<double,1,2> limitFootClearance_;
        Eigen::Matrix<double,1,2> limitFootContact_; // for gait enforcing
        ///
        std::vector<Eigen::Vector<double,12>> jointPosErrorHist_, jointVelHist_;
        std::vector<Eigen::Vector<double,18>> genForceTargetHist_;
        double standingSmoothness_;

        /// initialize
        Eigen::Matrix<double,3,3> rotYawNoise_,rotTotalNoise_;
        Eigen::Quaterniond quat_;
        double yawNoise_;
        ///
        std::vector<raisim::Visuals*> arrows_;
        raisim::HeightMap* heightMap_;
        /// curriculum
        double curriculum_;
        int iter_;
        double mu_;
        /// for barrier
        float barrierReward_;

        thread_local static std::mt19937 gen_;
        thread_local static std::normal_distribution<double> normDist_;
        thread_local static std::uniform_real_distribution<double> uniDist_;

    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
    thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0.0,1.0);
    thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1.0,1.0);
}

