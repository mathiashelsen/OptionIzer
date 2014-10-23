#include "TimeSeries.hpp"

TimeSeries::TimeSeries(int _nPoints, int _nSeries, double _initial, Generic_PDF *_PDF)
{
    rng = new boost::mt19937();
    nPoints = _nPoints;
    nSeries = _nSeries;
    series = new double*[nSeries];
    static boost::uniform_01<boost::mt19937> generator(*rng);
    for(int i = 0; i < nSeries; i++)
    {
	series[i] = new double[nPoints];
	series[i][0] = _initial;
	for(int j = 1; j < nPoints; j++)
	{
	    series[i][j] = series[i][j-1]*(_PDF->drawRandom(generator()));
	}
    }
}

TimeSeries::~TimeSeries()
{
    delete rng;
    for(int i = 0 ; i < nSeries; i++ )
    {
	delete[] series[i];
    }
    delete[] series;
}
