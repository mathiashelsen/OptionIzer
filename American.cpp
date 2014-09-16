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
    double *finalValues = new double[walk->nSeries];
    for(int i = 0; i < walk->nSeries; i++)
    {
	double finalValue = underlying;
	for(int j = 0; j < walk->nPoints; j++)
	{
	    finalValue *= (1.0 + walk->series[i][j]);
	}
    }

    delete[] finalValues;

}
