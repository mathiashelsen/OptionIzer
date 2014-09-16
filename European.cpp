#include "European.hpp"

EuropeanOption::EuropeanOption(double _r, double _S, double _strike)
{
    rate = _r;
    underlying = _S;
    strike = _strike;
}

void EuropeanOption::getValueDistribution(TimeSeries *walk, MyPDF *callDist, MyPDF *putDist)
{
    vector<double> callOptionValues;
    vector<double> putOptionValues;
    double discount = pow((1.0 + rate/100.0), -(double)walk->nPoints / 360.0);
    for(int i = 0; i < walk->nSeries; i++)
    {
	double finalValue = underlying;
	for(int j = 0; j < walk->nPoints; j++)
	{
	    finalValue *= (1.0 + walk->series[i][j]);
	}
	callOptionValues.push_back( (finalValue > strike) ? (finalValue - strike)*discount : 0.0 );
	putOptionValues.push_back( (strike > finalValue) ? (strike - finalValue)*discount : 0.0 );
    }
    callDist->generatePDF(&callOptionValues, false);
    putDist->generatePDF(&putOptionValues, false);
}

