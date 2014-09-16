#ifndef _AMERICAN_HPP
#define _AMERICAN_HPP

class AmericanOption : public Option
{
    protected:
	double strike; // The strike price of the underlying
    public:
	AmericanOption(double _r, double _S, double _strike);
	virtual void getValueDistribution(TimeSeries *walk, MyPDF *callDist, MyPDF *putDist);
}

#endif
