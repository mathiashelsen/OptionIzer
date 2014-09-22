#include "American.hpp"

static double WeighedLaguerre(double x, int n)
{
    switch(n)
    {
	case 0: return exp(-x/2.0);
		break;
	case 1: return (exp(-x/2.0)*(1.0 - x));
		break;
	case 2: return (exp(-x/2.0)*(1.0 - 2.0*x + x*x/2.0));
		break;
	case 3: return (exp(-x/2.0)*(-x*x*x + 9.0*x*x -18.0*x + 6.0)/6.0);
		break;
	default: return 0;
		break;
    }
}

void AmericanOption::evaluate()
{
    // First calculate the final value for each of the random walks
    double *finalValues = new double[walk->nSeries];
    // The pay off at each point in time for an option
    double *payoffs = new double[walk->nSeries];
    double *currentPayoff = new double[walk->nSeries], *futurePayoff = new double[walk->nSeries];;
    // The risk free discounting rate for each time step (not limited to daily rate)
    double stepRate = pow((1.0 + rate*0.001), -1.0/360.0);
    int *exercise = new int[walk->nSeries];
    vector<double> x;
    vector<double> y;
    vector<double> estimate;
    vector<int> indices;
    for(int i = 0; i < walk->nSeries; i++)
    {
	finalValues[i] = underlying;
	for(int j = 0; j < walk->nPoints; j++)
	{
	    finalValues[i] *= (1.0 + walk->series[i][j]);
	}
    }
    // First the last point in time
    for(int j = 0 ; j < walk->nSeries; j++)
    {
	payoffs[j] = max(strike-finalValues[walk->nPoints - 1], 0.0);
	futurePayoff[j] = payoffs[j];
	if(payoffs[j] > 0.0)
	{
	    exercise[j] = walk->nPoints - 1;
	}
    }

    // Now going back in time, calculate the expected payoff and LS estimate
    for(int i = walk->nPoints-2; i > -1; i-- )
    {
	x.erase(x.begin(), x.end());
	y.erase(y.begin(), y.end());
	estimate.erase(estimate.begin(), estimate.end());
	indices.erase(indices.begin(), indices.end());
	for(int j = 0; j < walk->nSeries; j++)
	{
	    // Calculate if the option is in the money
	    currentPayoff[j] = max(strike-finalValues[j], 0.0);
	    // if it is...
	    if(currentPayoff[j] > 0.0)
	    {
		//calculate what payoff lies in the future if it is not exercised
		y.push_back(futurePayoff[j]*stepRate);
		x.push_back(finalValues[j]);
		indices.push_back(j);
	    }
	}

	// perform LSE estimate
	LSE_estimate(&x, &y, &estimate);

	// Check if the estimated value is larger or smaller than the current payoff
	for(int j = 0; j < estimate.size(); j++)
	{
	    int pathIndex = indices.at(j);
	    
	    if( currentPayoff[pathIndex] > estimate.at(j) )
	    {
		payoffs[pathIndex] = currentPayoff;
		exercise[pathIndex] = i;
	    }
	}

	// Swap future and current payoff
	tmp = futurePayoff;
	futurePayoff = currentPayoff;
	currentPayoff = tmp;
    }
   
    delete[] exercise; 
    delete[] payoffs;
    delete[] finalValues;

}
