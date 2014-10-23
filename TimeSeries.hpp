#ifndef _TIMESERIES_HPP
#define _TIMESERIES_HPP

#include "MyPDF.hpp"
#include <vector>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

class TimeSeries
{
private:
    boost::mt19937 *rng;
public:
    int nPoints;
    int nSeries; 
    double **series;
    TimeSeries(int _nPoints, int _nSeries, double _initial, MyPDF *_PDF);
    ~TimeSeries();
};

#endif
