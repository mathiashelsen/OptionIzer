#ifndef _VANILLAOPTION_HPP
#define _VANILLAOPTION_HPP

#include "../Solvers/Solver.hpp"

class VanillaOption
{
    public:
	// Inputs
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	bool american;
	bool put;
	// Outputs
	double price, delta, gamma, theta;
    
	VanillaOption();
	VanillaOption(double _S0, double _K, double _sigma, double _r, double _T, bool _american, bool _put);
	void setUnderlying( double _S0 ){ S0 = _S0; };
	void setVol( double _s ) { sigma = _s; };
	void getPrice( double *_price ) { *_price = price; };
	void getGreeks( double *_delta, double *_gamma, double *_theta ){
	    *_delta = delta;
	    *_gamma = gamma;
	    *_theta = theta;
	};
	~VanillaOption();
};

#endif
