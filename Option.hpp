#ifndef _OPTION_HPP
#define _OPTION_HPP

#include "TimeSeries.hpp"
#include "MyPDF.hpp"

class Option
{
    protected:	
	double r; // The riskless rate
    public:
	void getValueDistribution(TimeSeries *walk, MyPDF *priceDistribution);
};

#endif
