#ifndef _VANILLAOPTION_HPP
#define _VANILLAOPTION_HPP

#include "../Solvers/Solver.hpp"
#include "Option.hpp"

class VanillaOption : public Option
{
    protected:
	VanillaOption(double _S0, double _K, double _sigma, double _r, double _T, bool _put);
    public:
	VanillaOption() { return; };
	// Inputs
	double K;
	double sigma;
	double T;
	bool put;
    
	void setVol( double _s ) { sigma = _s; };
	void getGreeks( double *_delta, double *_gamma, double *_theta ){
	    *_delta = delta;
	    *_gamma = gamma;
	    *_theta = theta;
	};
	~VanillaOption();
};

#endif
