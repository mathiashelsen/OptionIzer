#include "Binomial.hpp"

Binomial::Binomial(double _underlying,
	    double _strike,
	    double _volatility,
	    double _riskless,
	    double _T,
	    int _N )
{
    S0 = _underlying;
    K = _strike;
    sigma = _volatility;
    T = _T;
    N = _N;

    dt = T/(double)N;

    u = exp(sigma*sqrt(dt));
    d = 1.0/u;
    p = (exp(r*dt) - d)/(u - d);
}

void Binomial::evaluate()
{
    // Start with the construction of the binomial tree
    double **assetValues = new double*[N];
    double **optionValues = new double*[N];

    // The root of the tree is the underlying value
    assetValues[0] = new double[1];
    assetValues[0][0] = S0;
    optionValues[0] = new double[1];
    optionValues[0][0] = 0.0;

    for(int i = 1; i < N; i++)
    {
	assetValues[i] = new double[i+1];
	optionValues[i] = new double[i+1];

	// This might lead to numerical inaccuracies, though they probably won't matter
	assetValues[i][0] = assetValues[i-1][0]*u;

	for(int j = 1; j < (i+1); j++)
	{
	    optionValues[i][j] = 0.0;
	    assetValues[i][j] = assetValues[i][j-1]/(u*u);
	}
    }

    // Now we work backwards to calculate the option payoff
    // First the payoff for the terminal nodes
    for(int j = 0; j < (N+1); j++)
    {
	optionValues[N-1][j] = std::max(K - assetValues[N-1][j], 0.0 );	
    }

    for(int i = (N-2); i >= 0; i--)
    {
	for(int j = 0; j < (i+1); j++)
	{
	    // The value of exercising the put at this time
	    double intrinsic = std::max(K - assetValues[i][j], 0.0);
	    // The value obtained by waiting to exercise
	    double continuation = p*optionValues[i+1][j] + (1.0 - p)*optionValues[i+1][j+1];
	    continuation *= exp(-dt*r);
	    optionValues[i][j] = std::max(intrinsic, continuation);
	}
    }

    price = optionValues[0][0];
    delta = (optionValues[1][0] - optionValues[1][1])/(assetValues[1][0]-assetValues[1][1]);
    gamma = (optionValues[2][0] + optionValues[2][2] - 2.0*optionValues[2][0]);
	// Clean up some crap
	delete[] assetValues[i];
	delete[] optionValues[i+1];
    delete[] optionValues[0];
    delete[] assetValues;
    delete[] optionValues;

}

void Binomial::calcPrice(double *put)
{
    *put = price;
}

void Binomial::recalc(void)
{
    dt = T/(double)N;

    u = exp(sigma*sqrt(dt));
    d = 1.0/u;
    p = (exp(r*dt) - d)/(u - d);
}
