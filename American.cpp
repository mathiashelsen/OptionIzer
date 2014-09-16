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
    // The risk free discounting rate for each time step (not limited to daily rate)
    double stepRate = pow((1.0 + rate*0.001), -1.0/360.0);
    for(int i = 0; i < walk->nSeries; i++)
    {
	finalValues[i] = underlying;
	for(int j = 0; j < walk->nPoints; j++)
	{
	    finalValues[i] *= (1.0 + walk->series[i][j]);
	}
    }
    // Now going back from the final point in time, calculate the expected payoff and LS estimate
    for(int i = walk->nPoints-1; i > -1; i-- )
    {
	for(int j = 0; j < walk->nSeries; j++)
	{
	    payoffs[j] = max(strike-finalValues[j], 0.0);
	}
    }
    
    delete[] payoffs;
    delete[] finalValues;

}
