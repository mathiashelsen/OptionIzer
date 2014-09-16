#ifndef _EUROPEAN_HPP
#define _EUROPEAN_HPP

#include "Option.hpp"

class EuropeanOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
    public:
	EuropeanOption(double rate, double underlying, double strike, int bins);
	~EuropeanOption();
	virtual void evaluate();
};

#endif
