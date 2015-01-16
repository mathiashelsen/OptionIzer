#include "BinomialSolver.hpp"

template<> void BinomialSolver<EuroOption>::operator()(EuroOption *option)
{
    double dt = option->T/(double)N;
    double u = exp(option->sigma*sqrt(dt));
    double d = 1.0/u;
    double p = (exp(option->r*dt) - d)/(u - d);
    double multiplier = 1.0;
    if ( ! option->put )
    {
	multiplier = -1.0;
    }

    // The root of the tree is the underlying value
    assetValues[0][0] = option->S0;
    optionValues[0][0] = 0.0;

    for(int i = 1; i < N+1; i++)
    {
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
	optionValues[N][j] = std::max<double>(multiplier*(option->K - assetValues[N][j]), 0.0 );	
    }

    for(int i = (N-1); i > 0; i--)
    {
	for(int j = 0; j < (i+1); j++)
	{
	    // The value obtained by waiting to exercise
	    double continuation = p*optionValues[i+1][j] + (1.0 - p)*optionValues[i+1][j+1]; 
	    continuation *= exp(-dt*option->r);
	    optionValues[i][j] = continuation;
	}
    }

    option->price = p*optionValues[1][0] + (1.0 - p)*optionValues[1][1];
    option->price *= exp(-dt*option->r);
    option->delta = (optionValues[1][0] - optionValues[1][1])/(assetValues[1][0]-assetValues[1][1]);

    double delta1 = (optionValues[2][0] - optionValues[2][1])/(assetValues[2][0]-assetValues[2][1]);
    double delta2 = (optionValues[2][1] - optionValues[2][2])/(assetValues[2][1]-assetValues[2][2]);
    double h = 0.5*(assetValues[2][0]-assetValues[2][2]);
    option->gamma = (delta1 - delta2)/h;

    option->theta = (optionValues[2][1] - optionValues[0][0])/(2.0*dt);
};

template<> void BinomialSolver<AmericanOption>::operator()(AmericanOption *option)
{
    double dt = option->T/(double)N;
    double u = exp(option->sigma*sqrt(dt));
    double d = 1.0/u;
    double p = (exp(option->r*dt) - d)/(u - d);
    double multiplier = 1.0;
    if ( ! option->put )
    {
	multiplier = -1.0;
    }

    // The root of the tree is the underlying value
    assetValues[0][0] = option->S0;
    optionValues[0][0] = 0.0;

    for(int i = 1; i < N+1; i++)
    {
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
	optionValues[N][j] = std::max<double>(multiplier*(option->K - assetValues[N][j]), 0.0 );	
    }

    for(int i = (N-1); i > 0; i--)
    {
	for(int j = 0; j < (i+1); j++)
	{
	    // The value obtained by waiting to exercise
	    double continuation = p*optionValues[i+1][j] + (1.0 - p)*optionValues[i+1][j+1]; 
	    continuation *= exp(-dt*option->r);
	    double intrinsic = std::max(multiplier*(option->K - assetValues[i][j]), 0.0);
	    optionValues[i][j] = std::max(intrinsic, continuation);
	}
    }

    option->price = p*optionValues[1][0] + (1.0 - p)*optionValues[1][1];
    option->price *= exp(-dt*option->r);
    option->delta = (optionValues[1][0] - optionValues[1][1])/(assetValues[1][0]-assetValues[1][1]);

    double delta1 = (optionValues[2][0] - optionValues[2][1])/(assetValues[2][0]-assetValues[2][1]);
    double delta2 = (optionValues[2][1] - optionValues[2][2])/(assetValues[2][1]-assetValues[2][2]);
    double h = 0.5*(assetValues[2][0]-assetValues[2][2]);
    option->gamma = (delta1 - delta2)/h;

    option->theta = (optionValues[2][1] - optionValues[0][0])/(2.0*dt);
};
