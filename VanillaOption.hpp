#ifndef _VANILLAOPTION_HPP
#define _VANILLAOPTION_HPP

#include "Solver.hpp"

class VanillaOption
{
    friend class BinomialSolver;
    protected:
	// Inputs
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	bool american;
	// Outputs
	double price, delta, gamma, theta;
    public:
	VanillaOption();
	VanillaOption(double _S0, double _K, double _sigma, double _r, double _T, bool _american);
	void setUnderlying( double _S0 ){ S0 = _S0; };
	void getPrice( double *_price ) { *_price = price; };
	void getGreeks( double *_delta, double *_gamma, double *_theta ){
	    *_delta = delta;
	    *_gamma = gamma;
	    *_theta = theta;
	};
	~VanillaOption();
};

#endif
