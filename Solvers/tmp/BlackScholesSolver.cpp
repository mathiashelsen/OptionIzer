#include "BlackScholesSolver.hpp"

static inline double NCDF(double mu, double sigma, double x)
{
    return (1.0 + erf((x-mu)/(sigma*SQRT2)))/2.0;
}

BlackScholesSolver::BlackScholesSolver()
{

};

BlackScholesSolver::~BlackScholesSolver()
{

};

void BlackScholesSolver::operator()(VanillaOption *option)
{
    if(! option->american )
    {
	double d1 = (log(option->S0/option->K) + (option->r + 0.5*option->sigma*option->sigma)*option->T)/(option->sigma*sqrt(option->T));
	double d2 = d1 - option->sigma*sqrt(option->T);

	double tmp = NCDF(0.0, 1.0, d1)*option->S0 - NCDF(0.0, 1.0, d2)*option->K*exp(-option->T*option->r);
	
	if( option->put )
	{
	    option->price = option->K*exp(-option->T*option->r) - option->S0 + tmp;
	    option->delta = NCDF(0.0, 1.0, d1) - 1.0;
	    option->theta = -option->S0*option->sigma*exp(-d1*d1*0.5)/(SQRT2PI*2.0*sqrt(option->T)) 
		+ option->r*option->K*exp(-option->r*option->T)*NCDF(0.0, 1.0, -d2);
	}
	else
	{
	    option->price = tmp;
	    option->delta = NCDF(0.0, 1.0, d1);
	    option->theta = -option->S0*option->sigma*exp(-d1*d1*0.5)/(SQRT2PI*2.0*sqrt(option->T)) 
		- option->r*option->K*exp(-option->r*option->T)*NCDF(0.0, 1.0, -d2);
	}
	option->gamma = exp(-d1*d1*0.5)/(option->S0*option->sigma*sqrt(option->T)*SQRT2PI);
    }
};
