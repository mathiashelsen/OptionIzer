#ifndef _ASIANOPTION_HPP
#define _ASIANOPTION_HPP

#include "Option.hpp"

class AsianOption : public Option
{
    public:
	double S0;
	double K;
	double sigma;
	double r;
	double T;
	bool put;
	// Outputs
	double price, delta, gamma, theta;

	AsianOption(double _S0, double _K, double _sigma, double _r, double _T, bool _put);
	void setUnderlying( double _S0 ){ S0 = _S0; };
	void getPrice( double *_price ){ *_price = price; };
};

#endif
