#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() == 0 || estimations.size() != ground_truth.size())
	{
	    std::cout << "Invalid estimation of ground_truth data" << std::endl;
		return rmse;
	}
	
	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
    rmse = rmse/estimations.size();
	
	//calculate the squared root
    rmse = rmse.array().sqrt();
	
	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

    float den = pow(px,2)+pow(py,2);
	//check division by zero
	if (den < 0.0001)
	{
	    std::cout << "Function CalculateJacobian() Error - Division by 0" << std::endl;
		return Hj;
	}
	else{
		//compute the Jacobian matrix
		Hj << px/sqrt(den), py/sqrt(den), 0, 0,
			  -py/den, px/den, 0, 0,
			  py*(vx*py - vy*px)/pow(den,1.5), py*(vy*px - vx*py)/pow(den,1.5),
			  px/sqrt(den), py/sqrt(den); 
	    
		return Hj;
	}
	
}
