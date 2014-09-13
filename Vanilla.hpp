#ifndef _VANILLACALL_HPP
#define _VANILLACALL_HPP

#include "Option.hpp"

class VanillaOption : public Option
{
    protected:
	bool call;
	double strike; // The strike price of the underlying
    public:
	VanillaOption(double _r, double _S, bool _call, double _strike);
	virtual void getValueDistribution(TimeSeries *walk, MyPDF *priceDistribution);
};

#endif
