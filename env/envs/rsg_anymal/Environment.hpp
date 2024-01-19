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
            pTarget_.setZero(); prevTarget_.setZero(); prevPrevTarget_.setZero(); preJointVel_.setZero();

            /// this is nominal configuration of anymal
            gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

            /// set pd
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 154;
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
            footIndices_.push_back(anymal_->getBodyIdx("RH_SHANK"));
            footIndices_.push_back(anymal_->getBodyIdx("LH_SHANK"));
            footIndices_.push_back(anymal_->getBodyIdx("RF_SHANK"));
            footIndices_.push_back(anymal_->getBodyIdx("LF_SHANK"));
            footFrames_.push_back("RH_ADAPTER_TO_FOOT");
            footFrames_.push_back("LH_ADAPTER_TO_FOOT");
            footFrames_.push_back("RF_ADAPTER_TO_FOOT");
            footFrames_.push_back("LF_ADAPTER_TO_FOOT");
            rollJointFrames_.push_back("RH_HAA");
            rollJointFrames_.push_back("LH_HAA");
            rollJointFrames_.push_back("RF_HAA");
            rollJointFrames_.push_back("LF_HAA");

            phase_ = 0.0;
            gait_hz_ = 0.72;

            rot.setZero();
            command_.setZero();
            footContact_.setZero();
            jointPosWeight_.setZero(12);
            footPosWeight_.setZero();
            footContactPhase_.setZero();
            footClearance_.setZero();
            phaseSin_.setZero();
            footSlip_.setZero();
            footContactDouble_.setZero();
            footToTerrain_.setZero();
            footVel_.resize(4); footPos_.resize(4), rollJointPos_.resize(4), refBodyToFoot_.resize(4);
            jointPosErrorHist_ = std::vector<Eigen::Vector<double,12>>(18,Eigen::Vector<double,12>::Zero());
            jointVelHist_ = std::vector<Eigen::Vector<double,12>>(18,Eigen::Vector<double,12>::Zero());
            genForceTargetHist_ = std::vector<Eigen::Vector<double,18>>(3,Eigen::Vector<double,18>::Zero());


            for (int i=0;i<4;i++){
                if(i == 1 || i == 3){
                    limitJointPos_.row(i*3+0) << -0.03-0.523599,-0.03+0.523599;
                }  else {
                    limitJointPos_.row(i*3+0) << 0.03-0.523599,0.03+0.523599;
                }

                if(i < 2){
                    limitJointPos_.row(i*3+1) << 0.4-0.785398,0.4+0.785398; // hip
                }  else {
                    limitJointPos_.row(i*3+1) << -0.4-0.785398,-0.4+0.785398; // hip
                }

                if(i < 2){
                    limitJointPos_.row(i*3+2) << -0.8-1.04,-0.8+1.04; // knee
                }  else {
                    limitJointPos_.row(i*3+2) << 0.8-1.04,0.8+1.04; // knee
                }
            }

//            limitBodyHeight_ << 0.48, 0.68;
//            limitBodyHeight_ << 0.23, 0.71;
            limitBaseMotion_ << -0.3,0.3;
            limitJointVel_ << -8,8;
            limitTargetVel_ << -0.4,0.4;
            limitFootContact_ << -0.3,3.0;
            limitFootClearance_ << -0.08,1.0;

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(anymal_);
                arrows_xy = server_->addVisualArrow("command_xy",0.1,0.3,0,1,0,1);
                arrows_yaw = server_->addVisualArrow("command_yaw",0.1,0.3,1,0,0,1);
//                visual_target = server_->addVisualSphere("visual_target",0.05,1,0,0,0.4);
//                external_force = server_->addVisualArrow("visual_force",0.25,0.5,1,0,0);
            }

            /// initial body to foot pos
            anymal_->setState(gc_init_,gv_init_);

            for(int i = 0; i < 4; i++) {
                anymal_->getFrameVelocity(footFrames_[i], footVel_[i]);
                anymal_->getFramePosition(footFrames_[i], footPos_[i]);
                anymal_->getFramePosition(rollJointFrames_[i], rollJointPos_[i]);
                refBodyToFoot_[i] =   footPos_[i] - rollJointPos_[i];
//          std::cout << i<<"th initial body to foot pos : " << refBodyToFoot_[i].e().transpose() << std::endl;
            }
        }

        void init() final { }


        void reset() final {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();

            do {
//            double maxCommand = (iter_ % 4 == 0) ? (1.0 + comCurriculum * 1.0) : (1.0 + comCurriculum * 0.5); // 평지 lin x max 2.0, other 1.5
                double maxCommand = 1.8;
                command_ << maxCommand * uniDist_(gen_), 0.6 * uniDist_(gen_), 0.6 * uniDist_(gen_);     // [lix x max, 0.6, 0.6]
                command_(0) = (command_(0) < -1.0) ? command_(0)+1.2 : command_(0);           // 뒤로가는 건 max -1.0
//            command_ << maxCommand/2 * (uniDist_(gen_)+1), 0, 0;     // [lix x max, 0.6, 0.6]
            } while (command_.norm() < 0.2);

//            std::cout << command_.transpose() << std::endl;

            for (auto& vec : genForceTargetHist_) { vec.setZero(); }
            phase_ = 0.0;
            pTarget_ = gc_.tail(12);
            prevTarget_ = pTarget_; prevPrevTarget_ = pTarget_; preJointVel_.setZero();

            for (auto& vec : genForceTargetHist_) { vec.setZero(); }
            for (auto& vec : jointPosErrorHist_) { vec.setZero(); }
            for (auto& vec : jointVelHist_) { vec.setZero(); }

            footContactPhase_.setZero();
            footClearance_.setZero();
        }


        float step(const Eigen::Ref<EigenVec>& action) final {
            /// action scaling
            pTarget_ = action.cast<double>();
            pTarget_ = pTarget_.cwiseProduct(actionStd_);
            pTarget_ += actionMean_;                                  /// joint P target

            double avgReward = 0.0;
            barrierReward_ = 0.0;

            for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
                /// compute target torque
                computeTorque();
                anymal_->setGeneralizedForce(genForceTargetHist_[0]); /// 2ms delay (torque command in PC -> actual torque in real robot)
                /// simpulation
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
                updateObservation();
                avgReward += getNegPosReward();
                barrierReward_+= getLogBarReward();
                visualizeCommand();
            }

            avgReward /= (control_dt_ / simulation_dt_ + 1e-10);
            barrierReward_ /= (control_dt_ / simulation_dt_ + 1e-10);
            updateHistory();

            return rewards_.sum();
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
            raisim::Vec<4> quat;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            for(int i = 0; i < 4; i++) {
                anymal_->getFramePosition(footFrames_[i], footPos_[i]);
                anymal_->getFrameVelocity(footFrames_[i], footVel_[i]);
                anymal_->getFramePosition(rollJointFrames_[i], rollJointPos_[i]);
            }

            // foot contact update
            footContact_.setZero();
            for(auto& contact: anymal_->getContacts()){
                for (size_t i=0; i<4; i++){
                    if(contact.getlocalBodyIndex() == footIndices_[i]){
                        footContact_(i) = true;
                    }
                }
            }

            updateFootToTerrain();
            /// for gait enforcing & foot clearance
            phase_ += simulation_dt_;
            footContactPhase_(0) = sin(phase_/gait_hz_ * 2*3.141592); // RR
            footContactPhase_(1) = -footContactPhase_(0); // RL
            footContactPhase_(2) = -footContactPhase_(0); // FR
            footContactPhase_(3) = footContactPhase_(0); // FL

            phaseSin_(0) = sin(phase_/gait_hz_ * 2*3.141592); // for observation
            phaseSin_(1) = cos(phase_/gait_hz_ * 2*3.141592); // for observation
//
            if (1) { /// walking
                /// footContactDouble_ -> limit_foot_contact 에 있도록 (-0.3,3) -> Gait Enforcing (요 -0.3 이 벗어나도 되는 범위)
                for (int i = 0; i < 4; i++) {
                    if (footContact_(i)) { footContactDouble_(i) = 1.0 * footContactPhase_(i); }
                    else { footContactDouble_(i) = -1.0 * footContactPhase_(i); }
                }
                /// footClearance_ -> limit_foot_clearance 에 있도록 (-0.12,0.12) -> foot 드는 거 enforcing
                double desiredFootZPosition = 0.15;
                for (int i = 0; i < 4; i++) {
                    if (footContactPhase_(i) < -0.6) { /// during swing, 전체시간의 33 %
                        footClearance_(i) = footToTerrain_.segment(i * 5, 5).minCoeff() -desiredFootZPosition;
                        // 대략, 0.17 sec, 0 보다 크거나 같으면 됨 (enforcing clearance)
                    } else { footClearance_(i) = 0.0; } // max reward (not enforcing clearance)
                }
            }
//            } else { /// under standingMode_
//                /// standingMode_ 는 zero command 로 부터 유추 가능, command 는 obs 이기 때문에, robot 은 standingMode_인지 아닌지 충분히 알 수 있음
//                for (int i=0; i<4; i++){
//                    footContactDouble_(i) = 1.0; // around max reward, where this value should go under (-0.3,3)
//                    footClearance_(i) = 0.0; // max reward (not enforcing clearance)
//                }
//            }

            obDouble_ << rot.e().row(2).transpose(), /// body orientation : 3
                    gc_.tail(12), /// joint angles : 12
                    gv_.tail(12), /// joint velocity 12
                    bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity : 6
                    prevTarget_,                                                          /// previous action 12
                    prevPrevTarget_,                                                      /// preprevious action 12
                    jointPosErrorHist_[0], jointPosErrorHist_[6], jointPosErrorHist_[12], /// joint History 36 (0.18, 0.12, 0.6)
                    jointVelHist_[0], jointVelHist_[6], jointVelHist_[12],                /// joint History 36 (0.18, 0.12, 0.6)
                    rot.e().transpose() * (footPos_[0].e() - gc_.head(3)), rot.e().transpose() * (footPos_[1].e() - gc_.head(3)),
                    rot.e().transpose() * (footPos_[2].e() - gc_.head(3)), rot.e().transpose() * (footPos_[3].e() - gc_.head(3)), /// relative foot position with respect to the body COM, expressed in the body frame 12
                    command_, /// command 3
                    phaseSin_(0),phaseSin_(1), //// phase encoding 2
                    footClearance_,  //// footClearance_ 4
                    footContact_.cast<double>(); ////footContact_ 4
        }

        void computeTorque(){
            genForceTargetHist_.erase(genForceTargetHist_.begin());
            Eigen::Vector<double,18> tempGenForce; tempGenForce.head(6).setZero();
            tempGenForce.tail(12) = jointPgain.tail(12).cwiseProduct(pTarget_-gc_.tail(12))
                                    + jointDgain.tail(12).cwiseProduct(-gv_.tail(12));

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

        void updateHistory(){
            prevPrevTarget_ = prevTarget_;
            prevTarget_ = pTarget_;

            jointVelHist_.erase(jointVelHist_.begin());
            jointVelHist_.push_back(gv_.tail(12));

            jointPosErrorHist_.erase(jointPosErrorHist_.begin());
            jointPosErrorHist_.push_back(pTarget_ - gc_.tail(12));
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

            arrow_pos_offset << 0,0,0.4;
            arrow_pos_offset = rot_robot * arrow_pos_offset.eval();
            quaternion = rot_robot.eval() * rot_pitch_90 * rot_command;

            if (visualizable_) {
                arrows_xy->setCylinderSize(0.3, command.head(2).norm() * 0.4);
                arrows_xy->setPosition(gc_head_7.head(3) + arrow_pos_offset);
                arrows_xy->setOrientation(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());

                arrows_yaw->setCylinderSize(0.3, command(2) * 0.4);
                arrows_yaw->setPosition(gc_head_7.head(3) + arrow_pos_offset);
                arrows_yaw->setOrientation(gc_head_7.segment(3, 4));
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
//            rewards_.record("bodyOri", std::acos(rot(8)) * std::acos(rot(8)) * bodyOriWeight);
            jointPosWeight_ << 1.0, 0.6,0.6,1.,0.6,0.6,1.,0.6,0.6,1.,0.6,0.6;
            rewards_.record("smoothness2", ((pTarget_ - 2 * prevTarget_ + prevPrevTarget_).cwiseProduct(jointPosWeight_)).squaredNorm());
            rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());

            /// joint pos regulation -> not used
            Eigen::VectorXd jointPosTemp(12); jointPosTemp.setZero();
//      jointPosTemp = gc_.tail(12) - gcInit_.tail(12);
//      jointPosTemp = jointPosWeight_.cwiseProduct(jointPosTemp.eval());
            rewards_.record("jointPos", jointPosTemp.squaredNorm());

            /// task space foot pos regulation -> used
            Eigen::Vector3d  tempVec;
            footPosWeight_ << 0.6,1.0,0.4;
            double tempReward = 0.0;
            for(int index_leg = 0; index_leg < 4; index_leg++){
                tempVec = (footPos_[index_leg].e() - rollJointPos_[index_leg].e());
                tempVec = rot.e().transpose() *  tempVec.eval();
                tempReward += footPosWeight_.cwiseProduct(tempVec-refBodyToFoot_[index_leg].e()).squaredNorm();
            }
            rewards_.record("footPos", tempReward);

//            /// body height difference -> used
//            bodyFrameHeight_.setZero();
//            for (int i=0; i<2; i++) {
//                for (int j = 0; j < 2; j++) {
//                    int index_leg = i * 2 + j;
//                    tempVec = (footPos_[index_leg].e() - rollJointPos_[index_leg].e());
//                    tempVec(2) = heightMap_->getHeight(footPos_[index_leg](0), footPos_[index_leg](1)) -
//                                 rollJointPos_[index_leg](2);
//                    tempVec = rot_.e().transpose() * tempVec.eval();
//                    bodyFrameHeight_(i) -= tempVec(2);
//                }
//            }
//            bodyFrameHeight_ = bodyFrameHeight_.eval()/2.0;
//            rewards_.record("bodyHeightDifference", pow(bodyFrameHeight_(0)-bodyFrameHeight_(1),2.0) * heightDifferenceWeight_);
////            std::cout << "bodyHeightDifference : " << rewards_.getReward("bodyHeightDifference") << std::endl;

//            rewards_.record("jointVel", gv_.tail(12).squaredNorm()  * (double)(standingMode_));                 /// only for standingMode_
//            rewards_.record("jointAcc", (gv_.tail(12) - preJointVel_).squaredNorm() * (double)(standingMode_)); /// only for standingMode_

            /// sum
            float posReward, negReward;
            posReward = (float)(rewards_.getReward("comAngularVel") + rewards_.getReward("comLinearVel"));
            negReward = (float)(rewards_.getReward("bodyHeightDifference")+ rewards_.getReward("footPos")+rewards_.getReward("jointPos") + rewards_.getReward("jointVel") + rewards_.getReward("jointAcc") + rewards_.getReward("torque") + rewards_.getReward("footSlip") + rewards_.getReward("smoothness2"));
//            rewards_.record("negReward2", negReward); /// only for recording
            return (float)(std::exp(0.2 * negReward) * posReward);
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float& terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            /// if the contact body is not feet
//            for(auto& contact: anymal_->getContacts())
//                if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
//                    return true;
            for(auto& contact: anymal_->getContacts())
                if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end()) {
                    return true;
                }

            terminalReward = 0.f;

            return false;
        }

        void curriculumUpdate() { };

        float getLogBarReward(){
            /// compute barrier reward
            double barrierJointPos = 0.0, barrierBodyHeight = 0.0, barrierBaseMotion = 0.0, barrierJointVel = 0.0, barrierTargetVel = 0.0, barrierFootContact = 0.0, barrierFootClearance = 0.0;
            double tempReward = 0.0;

            /// Log Barrier - limit_joint_pos
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    int index_joint = i * 3 + j;
                    //         relaxedLogBarrier(0.09,limitJointPos_(index_leg,0),limitJointPos_(index_leg,1),gc_(7+index_leg),tempReward);
                    relaxedLogBarrier(0.08, limitJointPos_(index_joint, 0), limitJointPos_(index_joint, 1),
                                      gc_(7 + index_joint), tempReward);
                    barrierJointPos += tempReward;
                    //        std::cout << index_leg<<" th joint : " << tempReward << std::endl;
                }
            }
            // Log Barrier - limit_body_height
//      for (int i = 0; i < 2; i++) {
//          relaxedLogBarrier(0.04, limitBodyHeight_(0), limitBodyHeight_(1), bodyFrameHeight_(i), tempReward);
//          barrierBodyHeight += tempReward;
//      }

            // Log Barrier - limit_base_motion
            relaxedLogBarrier(0.2, limitBaseMotion_(0, 0), limitBaseMotion_(0, 1), bodyLinearVel_(2), tempReward);
            barrierBaseMotion += tempReward;

            for (int i = 0; i < 2; i++) {
                relaxedLogBarrier(0.3, limitBaseMotion_(1, 0), limitBaseMotion_(1, 1), bodyAngularVel_(i), tempReward);
                barrierBaseMotion += tempReward;
            }

            // Log Barrier - limit_joint_vel
            for (int i = 0; i < 12; i++) {
                relaxedLogBarrier(2.0, limitJointVel_(0), limitJointVel_(1), gv_(6 + i), tempReward);
                barrierJointVel += tempReward;
            }

            // Log Barrier - limit_target_vel
            relaxedLogBarrier(0.2, limitTargetVel_(0), limitTargetVel_(1), bodyLinearVel_(0) - command_(0), tempReward);
            barrierTargetVel += tempReward;
            relaxedLogBarrier(0.2, limitTargetVel_(0), limitTargetVel_(1), bodyLinearVel_(1) - command_(1), tempReward);
            barrierTargetVel += tempReward;
            relaxedLogBarrier(0.2, limitTargetVel_(0), limitTargetVel_(1), bodyAngularVel_(2) - command_(2), tempReward);
            barrierTargetVel += tempReward;

            // Log Barrier - limit_foot_contact
            for (int i = 0; i < 4; i++) {
                relaxedLogBarrier(0.1, limitFootContact_(0), limitFootContact_(1), footContactDouble_(i), tempReward);
                barrierFootContact += tempReward;
            }
            // Log Barrier - limit_foot_clearance
            for (int i = 0; i < 4; i++) {
                relaxedLogBarrier(0.004
                                  , limitFootClearance_(0), limitFootClearance_(1), footClearance_(i), tempReward);
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
            barrierJointPos = fmax(barrierJointPos, logClip);           /// 여기 밖 부분은 gradient 안 받겠다
            barrierBaseMotion = fmax(barrierBaseMotion,logClip);
            barrierJointVel = fmax(barrierJointVel,logClip);
            barrierTargetVel = fmax(barrierTargetVel,logClip);
            barrierFootContact = fmax(barrierFootContact,logClip);
            barrierFootClearance = fmax(barrierFootClearance,logClip);
            rewards_.record("barrierJointPos", barrierJointPos);
//      rewards_.record("barrierBodyHeight", barrierBodyHeight);
            rewards_.record("barrierBaseMotion", barrierBaseMotion);
            rewards_.record("barrierJointVel", barrierJointVel);
            rewards_.record("barrierTargetVel", barrierTargetVel);
            rewards_.record("barrierFootContact", barrierFootContact);
            rewards_.record("barrierFootClearance", barrierFootClearance);

            float logBarReward =  (float)(1e-1*(barrierJointPos + barrierBaseMotion + barrierJointVel + barrierTargetVel + barrierFootContact + barrierFootClearance));
//      rewards_.record("relaxedLog", logBarReward); /// relaxed log barrier
            return  logBarReward;
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
                sample_point.col(i) = rot.e().transpose() * sample_point.col(i).eval();
            }
            Eigen::Matrix<double, 5, 1> temp_foot;
            Eigen::Matrix<double, 3, 1> temp3;
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < 5; i++) {
                    temp3 = footPos_[k].e() + sample_point.col(i);
                    footToTerrain_(5 * k + i) = footPos_[k].e()(2) - 0.0;
//                   footToTerrain_(5 * k + i) = footPos_[k].e()(2) - heightMap_->getHeight(temp3(0), temp3(1));
                }
            }
//            if (abs(footToTerrain_.minCoeff()) >  3.0){ /// print error !!!
//               std::cout << "gc_[2] : " << gc_[2] << std::endl;
//               std::cout << "height map : " << heightMap_->getHeight(temp3(0), temp3(1)) <<  std::endl;
//               std::cout << "Error too big here : " << footToTerrain_.transpose() << std::endl;
//           }
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


    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* anymal_;
        raisim::Visuals* arrows_xy;
        raisim::Visuals* arrows_yaw;
//        raisim::Visuals* visual_EEpos;
//        raisim::Visuals* external_force;
//        raisim::Visuals* visual_target2;
//        std::vector<raisim::Visuals*> arrows_;

        raisim::Mat<3,3> rot;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
        double barrierReward_;
        double terminalRewardCoeff_ = -10.0;
        double phase_;
        double gait_hz_;
        Eigen::Vector<double,12> pTarget_, prevTarget_, prevPrevTarget_, preJointVel_;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_,command_; //// TEEpos_, posError_,baseError_,basepos_,exforce3d,exforceX,exforceY,exforceZ,
        Eigen::Vector4d footContact_;
        Eigen::VectorXd jointPosWeight_;
        Eigen::Vector3d footPosWeight_;
        Eigen::Vector<double,18> jointPgain, jointDgain;
        std::vector<Eigen::Vector<double,12>> jointPosErrorHist_, jointVelHist_;
        std::vector<Eigen::Vector<double,18>> genForceTargetHist_;
        Eigen::Matrix<double,4,1> footContactDouble_; // gait
        Eigen::Matrix<double,4,1> footContactPhase_;  // gait hz
        Eigen::Matrix<double,4,1> footClearance_;     // foot clearance
        Eigen::Matrix<double,4,1> footSlip_;     // foot clearance
        Eigen::Matrix<double,20,1> footToTerrain_; // 5 sample point for each footfo
        Eigen::Matrix<double,2,1> phaseSin_;  // sin cos representation of phase

        Eigen::Matrix<double,12,2> limitJointPos_;
//        Eigen::Matrix<double,1,2> limitBodyHeight_;
        Eigen::Matrix<double,1,2> limitBaseMotion_; // z vel, roll,pitch vel
        Eigen::Matrix<double,1,2> limitJointVel_;
        Eigen::Matrix<double,1,2> limitTargetVel_;
        Eigen::Matrix<double,1,2> limitFootClearance_;
        Eigen::Matrix<double,1,2> limitFootContact_; // for gait enforcing

        std::vector<size_t> footIndices_;
        std::vector<std::string> footFrames_;
        std::vector<std::string> rollJointFrames_;
        std::vector<raisim::Vec<3>> footPos_,footVel_, rollJointPos_, refBodyToFoot_;
//        size_t RR_footIndex,RL_footIndex,FR_footIndex,FL_footIndex;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        thread_local static std::mt19937 gen_;
        thread_local static std::normal_distribution<double> normDist_;
        thread_local static std::uniform_real_distribution<double> uniDist_;
    };
    thread_local std::mt19937  raisim::ENVIRONMENT::gen_;
    thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
    thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}