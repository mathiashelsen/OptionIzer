#include "myPDF.hpp"
#include <vector>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

class timeSeries
{
private:
    int nPoints;
    int nSeries; 
    double **series;
public:
    timeSeries(int _nPoints, int _nSeries, myPDF *_PDF, boost::mt19937 *rng );
    ~timeSeries();
};
