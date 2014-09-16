#ifndef _EUROPEAN_HPP
#define _EUROPEAN_HPP

#include "Option.hpp"

class EuropeanOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
    public:
	EuropeanOption(double _r, double _S, double _strike);
	virtual void getValueDistribution(TimeSeries *walk, MyPDF *callDist, MyPDF *putDist);
};

#endif
