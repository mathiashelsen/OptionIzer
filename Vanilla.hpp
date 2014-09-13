#ifndef _VANILLA_HPP
#define _VANILLA_HPP

#include "Option.hpp"

class VanillaOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
    public:
	VanillaOption(double _r, double _S, bool _call, double _strike);
	virtual void getValueDistribution(TimeSeries *walk, MyPDF *callDist, MyPDF *putDist);
};

#endif
