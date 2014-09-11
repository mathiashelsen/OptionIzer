#ifndef _TIMESERIES_HPP
#define _TIMESERIES_HPP

#include "myPDF.hpp"
#include <vector>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

class timeSeries
{
private:
    int nPoints;
    int nSeries; 
    boost::mt19937 *rng;
public:
    double **series;
    timeSeries(int _nPoints, int _nSeries, myPDF *_PDF);
    ~timeSeries();
};

#endif
