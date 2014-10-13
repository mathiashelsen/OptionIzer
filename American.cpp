
/*
The MIT License (MIT)

Copyright (c) 2014 Mathias Helsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "American.hpp"

static double WeighedLaguerre(double x, int n)
{
    switch(n)
    {
	case 0: return 1.0;
		break;
	case 1: return (1.0 - x);
		break;
	case 2: return (1.0 - 2.0*x + x*x/2.0);
		break;
	case 3: return (-x*x*x + 9.0*x*x -18.0*x + 6.0)/6.0;
		break;
	default: return 0;
		break;
    }
}

void AmericanOption::LSE_estimate(vector<double> *x, vector<double> *y, vector<double> *ybar)
{
    assert( x->size() == y->size() );
    double *X = new double[x->size()*4];
    double *Y = new double[y->size()];
    for(unsigned int i = 0; i < x->size(); i++)
    {
	for(int j = 0; j < 4; j++ )
	{
	    X[4*i+j] = WeighedLaguerre(x->at(i), j);
	}
	Y[i] = y->at(i);
    }

    alglib::real_1d_array newY;
    newY.setcontent( y->size(), Y );
    alglib::real_2d_array newX;
    newX.setcontent( x->size(), 4, X); 

    alglib::ae_int_t info;
    alglib::real_1d_array c;
    alglib::lsfitreport rep;

    //
    // Linear fitting without weights
    //
    lsfitlinear(newY, newX, info, c, rep);
    ybar->erase(ybar->begin(), ybar->end());
    for( unsigned int i = 0; i < x->size(); i++ )
    {
	double tmp = 0.0;
	for(int j = 0; j < 4; j++)
	{
	    tmp += WeighedLaguerre(x->at(i), j)*c[j];
	}
	ybar->push_back(tmp);
    }
    delete[] X;
    delete[] Y;
}

void AmericanOption::evaluate()
{
    // First calculate the final value for each of the random walks
    double *finalValues = new double[walk->nSeries];
    // The pay off at each point in time for an option
    double *payoffs = new double[walk->nSeries];
    // The risk free discounting rate for each time step (not limited to daily rate)
    double stepRate = pow((1.0 + rate*0.01), -1.0/360.0);
    int *exercise = new int[walk->nSeries];
    vector<double> x;
    vector<double> y;
    vector<double> estimate;
    vector<int> indices;
    putDist = new MyPDF(nBins);

    double avgFinal = 0.0;
    double avgPutVal = 0.0;
    std::cout << underlying << std::endl;
    // Calculate the final value for each path
    for(int i = 0; i < walk->nSeries; i++)
    {
	finalValues[i] = underlying;
    }

    for(int j = 0; j < walk->nPoints; j++)
    {
	avgFinal = 0.0;
	for(int i = 0; i < walk->nSeries; i++)
	{
	    finalValues[i] *= (1.0 + walk->series[i][j]);
	    avgFinal += finalValues[i];
	}
    }

    // First the last point in time
    double avgEx = 0.0;
    for(int j = 0 ; j < walk->nSeries; j++)
    {
	payoffs[j] = max(strike-finalValues[j], 0.0);
	if(payoffs[j] > 0.0)
	{
	    exercise[j] = 1;
	}
	avgEx += (double)exercise[j];
    }

    // Now going back in time, calculate the expected payoff and LS estimate
    for(int i = walk->nPoints-2; i > -1; i-- )
    {
	x.erase(x.begin(), x.end());
	y.erase(y.begin(), y.end());
	estimate.erase(estimate.begin(), estimate.end());
	indices.erase(indices.begin(), indices.end());
	double avgFinal = 0.0;
	for(int j = 0; j < walk->nSeries; j++)
	{
	    // Discount the final value one step back
	    finalValues[j] /= (1.0 + walk->series[i][j]); 
	    avgFinal += finalValues[j];
	    // Calculate if the option is in the money
	    // if it is...
	    if(max(strike-finalValues[j], 0.0) > 0.0)
	    {
		//calculate what payoff lies in the future if it is not exercised
		y.push_back(payoffs[j]);
		x.push_back(finalValues[j]);
		indices.push_back(j);
	    }
	}

	// perform LS estimate
	LSE_estimate(&x, &y, &estimate);

	// Check if the estimated value is larger or smaller than the current payoff
	int k = 0;
	avgEx = 0.0;
	avgPutVal = 0.0;
	for(int j = 0; j < walk->nSeries; j++)
	{
	    if( (k < (int)indices.size()) && (indices.at(k) == j) )
	    {
		if( max(strike-finalValues[j], 0.0) > estimate.at(k) )
		{
		    payoffs[j] = max(strike-finalValues[j], 0.0)*stepRate;
		    exercise[j] = 1;
		}
		k++;
	    }
	    else
	    {
		payoffs[j] *= stepRate; 
	    }
	    avgPutVal += payoffs[j];
	    avgEx += (double)exercise[j];
	}
	std::cout << avgEx/(double)walk->nSeries << "\t" << avgFinal/(double)walk->nSeries << "\t" << avgPutVal/(double)walk->nSeries << std::endl;
    }
    
    vector<double> putVals;
    for(int i = 0; i < walk->nSeries; i++)
    {
	if(payoffs[i] == 0.0)
	{
	    putVals.push_back( 0.0 );
	}
	else
	{
	    putVals.push_back( payoffs[i] );
	}
    }
    putDist->generatePDF(&putVals, false);
   
    delete[] exercise; 
    delete[] payoffs;
    delete[] finalValues;

}

AmericanOption::AmericanOption(double _rate, double _underlying, double _strike, int bins)
{
    rate = _rate;
    underlying = _underlying;
    strike = _strike;
    nBins = bins;
}

AmericanOption::~AmericanOption()
{
    delete putDist;
}   
