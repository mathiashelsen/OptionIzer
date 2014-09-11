#ifndef _TIMESERIES_HPP
#define _TIMESERIES_HPP

#include "MyPDF.hpp"
#include <vector>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

class TimeSeries
{
private:
    int nPoints;
    int nSeries; 
    boost::mt19937 *rng;
public:
    double **series;
    TimeSeries(int _nPoints, int _nSeries, MyPDF *_PDF);
    ~TimeSeries();
};

#endif
