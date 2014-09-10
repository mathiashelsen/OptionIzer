#include "timeSeries.hpp"

timeSeries::timeSeries(int _nPoints, int _nSeries, myPDF *_PDF, boost::mt19937 *rng )
{
    nPoints = _nPoints;
    nSeries = _nSeries;
    series = new double*[nSeries];
    for(int i = 0; i < nSeries; i++)
    {
	series[i] = new double[nPoints];
	for(int j = 0; j < nPoints; j++)
	{
	    series[i][j] = _PDF->drawRandom(0.1);
	}
    }
}
