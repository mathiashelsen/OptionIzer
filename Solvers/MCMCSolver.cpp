#include "MCMCSolver.hpp"

MCMCSolver::MCMCSolver(Generic_PDF *_p, int _Nbins, int _Nseries, int _Nsteps)
{
    p = _p;
    priceHistogram = gsl_histogram_alloc(_Nbins);
    Nbins = _Nbins;
    Nseries = _Nseries;
    Nsteps = _Nsteps;
    assetValues = new double*[Nseries];
    for(int i = 0; i < Nseries; i++ )
    {
	assetValues[i] = new double[Nsteps];
    }
    // The pay off at each point in time for an option
    payoffs = new double[Nseries];
};

MCMCSolver::~MCMCSolver()
{
    gsl_histogram_free(priceHistogram);
    for(int i = 0; i < Nseries; i++ )
    {
	delete assetValues[i];
    }
    delete[] assetValues;
    delete[] payoffs;
};

static double WeighedLaguerre(double x, int n);
static void LSE_estimate(vector<double> *x, vector<double> *y, vector<double> *ybar);

void MCMCSolver::operator()(VanillaOption *option)
{
    for(int i = 0; i < Nseries; i++ )
    {
	assetValues[i][0] = option->S0;
	for(int j = 1; j < Nsteps; j++)
	{
	    assetValues[i][j] = assetValues[i][j-1]*(1.0 + p->drawRandom());
	}
    }

    // The risk free discounting rate for each time step (not limited to daily rate)
    dt = option->T / (double)Nsteps;
    double discount = exp( -option->r*dt );

    vector<double> x;
    vector<double> y;
    vector<double> estimate;
    vector<int> indices;

    double multiplier = 1.0; 
    if( ! option->put )
    {
	multiplier = -1.0;
    }

    // First the last point in time
    for(int i = 0 ; i < Nseries; i++)
    {
	payoffs[i] = max(multiplier*(option->K - assetValues[i][Nsteps-1]), 0.0);
    }

    // Now going back in time, calculate the expected payoff and LS estimate
    if( option->american )
    {
	for(int i = Nsteps-2; i > 0; i-- )
	{
	    x.erase(x.begin(), x.end());
	    y.erase(y.begin(), y.end());
	    estimate.erase(estimate.begin(), estimate.end());
	    indices.erase(indices.begin(), indices.end());

	    for(int j = 0; j < Nseries; j++)
	    {
		// Calculate if the option is in the money
		// if it is...
		if(max(multiplier*(option->K - assetValues[j][i]), 0.0) > 0.0)
		{
		    //calculate what payoff lies in the future if it is not exercised
		    y.push_back(payoffs[j]);
		    x.push_back(assetValues[j][i]);
		    indices.push_back(j);
		}
	    }

	    // perform LS estimate
	    LSE_estimate(&x, &y, &estimate);


	    // Check if the estimated value is larger or smaller than the current payoff
	    int k = 0;
	    for(int j = 0; j < Nseries; j++)
	    {
		if( (k < (int)indices.size()) && (indices.at(k) == j) )
		{
		    if( max(multiplier*(option->K - assetValues[j][i]), 0.0) > estimate.at(k) )
		    {
			payoffs[j] = max(multiplier*(option->K - assetValues[j][i]), 0.0);
		    }
		    else
		    {
			payoffs[j] *= discount; 
		    }
		    k++;
		}
		else
		{
		    payoffs[j] *= discount; 
		}
	    }
	}
    }
    else
    {
	discount = exp( -option->r * option->T );
	for(int i = 0; i < Nseries; i++)
	{
	    payoffs[i] *= discount;
	}
    }
   
    double maxPrice = 0.0, minPrice = 0.0; 
    for(int i = 0; i < Nseries; i++)
    {
	maxPrice = (maxPrice > payoffs[i]*1.01) ? maxPrice : payoffs[i]*1.01; // Upper bound is exclusive
	minPrice = (minPrice < payoffs[i]) ? minPrice : payoffs[i];
    }
    if( minPrice < maxPrice )
    {
	gsl_histogram_reset(priceHistogram);
	gsl_histogram_set_ranges_uniform(priceHistogram, minPrice, maxPrice);
	for(int i = 0; i < Nseries; i++)
	{
	    gsl_histogram_increment( priceHistogram, payoffs[i] );
	}
	option->price = gsl_histogram_mean( priceHistogram );
    }
    else
    {
	option->price = 0.0;
    }

}


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

static void LSE_estimate(vector<double> *x, vector<double> *y, vector<double> *ybar)
{
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
