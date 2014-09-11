#include "TimeSeries.hpp"

TimeSeries::TimeSeries(int _nPoints, int _nSeries, MyPDF *_PDF)
{
    rng = new boost::mt19937();
    nPoints = _nPoints;
    nSeries = _nSeries;
    series = new double*[nSeries];
    static boost::uniform_01<boost::mt19937> generator(*rng);
    for(int i = 0; i < nSeries; i++)
    {
	series[i] = new double[nPoints];
	for(int j = 0; j < nPoints; j++)
	{
	    series[i][j] = _PDF->drawRandom(generator());
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
