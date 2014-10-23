#ifndef _OPTION_HPP
#define _OPTION_HPP

#include <iostream>

using namespace std;

#include "TimeSeries.hpp"
#include "NIT_PDF.hpp"

class Option
{
    protected:	
	double rate; // The riskless rate
	double underlying; // The value of the underlying
	NIT_PDF *callDist;
	NIT_PDF *putDist;
	int nBins;
	TimeSeries *walk;
    public:
	virtual ~Option();
	virtual void evaluate() {};
	void setWalk( TimeSeries *_walk ) { walk = _walk; };
	NIT_PDF *getCallPriceDist() { return callDist; };
	NIT_PDF *getPutPriceDist() { return putDist; };
};

#endif
