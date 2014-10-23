#include "European.hpp"

EuropeanOption::EuropeanOption(double _rate, double _underlying, double _strike, int bins)
{
    rate = _rate;
    underlying = _underlying;
    strike = _strike;
    nBins = bins;
}

EuropeanOption::~EuropeanOption()
{
    if( callDist != NULL ) delete callDist;
    if( putDist != NULL ) delete putDist;
}

void EuropeanOption::evaluate()
{
    vector<double> callOptionValues;
    vector<double> putOptionValues;
    callDist = new NIT_PDF(nBins);
    putDist = new NIT_PDF(nBins);
    double discount = pow((1.0 + rate/100.0), -(double)walk->nPoints / 360.0);
    for(int i = 0; i < walk->nSeries; i++)
    {
	double finalValue = walk->series[i][walk->nPoints -1];
	callOptionValues.push_back( (finalValue > strike) ? (finalValue - strike)*discount : 0.0 );
	putOptionValues.push_back( (strike > finalValue) ? (strike - finalValue)*discount : 0.0 );
    }
    callDist->generatePDF(&callOptionValues);
    putDist->generatePDF(&putOptionValues);
}

