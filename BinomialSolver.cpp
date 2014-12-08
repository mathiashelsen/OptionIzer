#include "BinomialSolver.hpp"

BinomialSolver::BinomialSolver(int _N)
{
    N = _N;

    assetValues = new double*[N+1];
    optionValues = new double*[N+1];

    assetValues[0] = new double[1];
    optionValues[0] = new double[1];
    for(int i = 1; i < N+1; i++)
    {
	assetValues[i] = new double[i+1];
	optionValues[i] = new double[i+1];
    }
};

BinomialSolver::~BinomialSolver()
{
    for(int i = 0; i < N+1; i++)
    {
	delete assetValues[i];
	delete optionValues[i];
    }
    delete[] assetValues;
    delete[] optionValues;
};

void BinomialSolver::operator()(VanillaOption *option)
{
    double dt = option->T/(double)N;
    double u = exp(option->sigma*sqrt(dt));
    double d = 1.0/u;
    double p = (exp(option->r*dt) - d)/(u - d);

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
	optionValues[N][j] = std::max<double>(option->K - assetValues[N][j], 0.0 );	
	std::cout << optionValues[N][j] << ", ";
    }

    for(int i = (N-1); i > 0; i--)
    {
	for(int j = 0; j < (i+1); j++)
	{
	    // The value obtained by waiting to exercise
	    double continuation = p*optionValues[i+1][j] + (1.0 - p)*optionValues[i+1][j+1];
	    continuation *= exp(-dt*option->r);
	    if(option->american)
	    {
		// The value of exercising the put at this time
		double intrinsic = std::max(option->K - assetValues[i][j], 0.0);
		optionValues[i][j] = std::max(intrinsic, continuation);
	    }
	    else
	    {
		optionValues[i][j] = continuation;
	    }
	}
    }

    price = p*optionValues[1][0] + (1.0 - p)*optionValues[1][1];
    price *= exp(-dt*option->r);
    delta = (optionValues[1][0] - optionValues[1][1])/(assetValues[1][0]-assetValues[1][1]);

    double delta1 = (optionValues[2][0] - optionValues[2][1])/(assetValues[2][0]-assetValues[2][1]);
    double delta2 = (optionValues[2][1] - optionValues[2][2])/(assetValues[2][1]-assetValues[2][2]);
    double h = 0.5*(assetValues[2][0]-assetValues[2][2]);
    gamma = (delta1 - delta2)/h;

    theta = (optionValues[2][1] - optionValues[0][0])/(2.0*dt);
};
