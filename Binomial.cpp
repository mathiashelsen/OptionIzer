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
    r = _riskless;

    dt = T/(double)(N);

    u = exp(sigma*sqrt(dt));
    d = 1.0/u;
    p = (exp(r*dt) - d)/(u - d);
}

void Binomial::evaluate()
{

    // Start with the construction of the binomial tree
    double **assetValues = new double*[N+1];
    double **optionValues = new double*[N+1];

    // The root of the tree is the underlying value
    assetValues[0] = new double[1];
    assetValues[0][0] = S0;
    optionValues[0] = new double[1];
    optionValues[0][0] = 0.0;

    for(int i = 1; i < N+1; i++)
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
	optionValues[N][j] = std::max<double>(K - assetValues[N][j], 0.0 );	
    }

    for(int i = (N-1); i > 0; i--)
    {
	for(int j = 0; j < (i+1); j++)
	{
	    // The value of exercising the put at this time
	    double intrinsic = std::max(K - assetValues[i][j], 0.0);
	    // The value obtained by waiting to exercise
	    double continuation = p*optionValues[i+1][j] + (1.0 - p)*optionValues[i+1][j+1];
	    continuation *= exp(-dt*r);
	    optionValues[i][j] = std::max(intrinsic, continuation);
	    //optionValues[i][j] = continuation;
	}
    }

    price = p*optionValues[1][0] + (1.0 - p)*optionValues[1][1];
    price *= exp(-dt*r);
   // std::cout << price << std::endl;
    //delta = (optionValues[1][0] - optionValues[1][1])/(assetValues[1][0]-assetValues[1][1]);

    //double delta1 = (optionValues[2][0] - optionValues[2][1])/(assetValues[2][0]-assetValues[2][1]);
    /*
    double delta2 = (optionValues[2][1] - optionValues[2][2])/(assetValues[2][1]-assetValues[2][2]);
    double h = 0.5*(assetValues[2][0]-assetValues[2][2]);
    gamma = (delta1 - delta2)/h;

    theta = (optionValues[2][1] - optionValues[0][0])/(2.0*dt);
    */
    // Re-calculate the tree for a different volatility
    /*
    sigma *= 1.01;
    recalc();
    for(int i = 1; i < N; i++)
    {
	//assetValues[i] = new double[i+1];
	//optionValues[i] = new double[i+1];

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
    sigma /= 1.01;
    vega = (optionValues[0][0] - price)/(0.01*sigma);

    r *= 1.01;
    recalc();
    for(int i = 1; i < N; i++)
    {
	//assetValues[i] = new double[i+1];
	//optionValues[i] = new double[i+1];

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
    r /= 1.01;
    rho = (optionValues[0][0] - price)/(0.01*T);
    */
    // Clean up some crap
    for(int i = 0; i < N; i++ )
    {
	delete assetValues[i];
	delete optionValues[i];
    }
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
