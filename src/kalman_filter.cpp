#include <math.h>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  MatrixXd F_t = F_.transpose();
  P_ = F_*P_*F_t + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_*x_;
  VectorXd y = z - z_pred;
  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_*P_*H_t + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = (P_*H_t)*S_i;
  
  //New Estimation
  x_ += K * y;
  float x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ -= K * H_ * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	float rho = sqrt(pow(x_(0), 2) + pow(x_(1), 2));
	float phi = atan2(x_(1), x_(0));
	float rho_dot;
	if(rho < 0.0001){
		rho_dot = 0;
	} else {
	rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;
	}
	
	VectorXd z_pred(3);
	z_pred << rho, phi, rho_dot;
	VectorXd y = z-z_pred;
	
	//Normalizing between -pi & pi
	y(1) = atan2(sin(double(y(1))),cos(double(y(1))));
	
	MatrixXd H_t = H_.transpose();
	MatrixXd S = H_*P_*H_t + R_;
	MatrixXd S_i = S.inverse();
	MatrixXd K = (P_*H_t)*S_i;
	  
	//New Estimation
	x_ += K * y;
	float x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ -= K * H_ *P_;
}
