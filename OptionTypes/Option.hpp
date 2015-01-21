#ifndef _OPTION_HPP
#define _OPTION_HPP

#include <iostream>

class Option
{
    public:
	// Outputs
	double S0;
	double r;
	double price, delta, gamma, theta;
	Option() {return; };
	~Option() { return; }; 
	void setUnderlying( double _S0 ){ S0 = _S0; };
	void getPrice( double *_price ) { *_price = price; };
};

#endif
