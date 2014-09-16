#ifndef _OPTION_HPP
#define _OPTION_HPP

#include <iostream>

using namespace std;

#include "TimeSeries.hpp"
#include "MyPDF.hpp"

class Option
{
    protected:	
	double rate; // The riskless rate
	double underlying; // The value of the underlying
	MyPDF *callDist;
	MyPDF *putDist;
	int nBins;
	TimeSeries *walk;
    public:
	virtual ~Option();
	virtual void evaluate() {};
	void setWalk( TimeSeries *_walk ) { walk = _walk; };
	MyPDF* getCallPriceDist() { return callDist; };
	MyPDF* getPutPriceDist() { return putDist; };
};

#endif
