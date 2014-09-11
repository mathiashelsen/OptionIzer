#include "timeSeries.hpp"

timeSeries::timeSeries(int _nPoints, int _nSeries, myPDF *_PDF)
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

timeSeries::~timeSeries()
{
    delete rng;
    for(int i = 0 ; i < nSeries; i++ )
    {
	delete[] series[i];
    }
    delete[] series;
}
