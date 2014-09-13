#ifndef _OPTION_HPP
#define _OPTION_HPP

#include "TimeSeries.hpp"
#include "MyPDF.hpp"

class Option
{
    protected:	
	double rate; // The riskless rate
	double underlying; // The value of the underlying
    public:
	virtual void getValueDistribution(TimeSeries *walk, MyPDF *priceDistribution) {};
};

#endif
