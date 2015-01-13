#include "FiniteDiffSolver.hpp"

FiniteDiffSolver::FiniteDiffSolver(int _Nt, int _Nz)
{
    Nt = _Nt;
    Nz = _Nz;

    f = gsl_vector_calloc(Nz);
    g = gsl_vector_calloc(Nz);
    diag = gsl_vector_calloc(Nz);
    super = gsl_vector_calloc(Nz-1);
    sub = gsl_vector_calloc(Nz-1);

};

FiniteDiffSolver::~FiniteDiffSolver()
{
    gsl_vector_free(diag); gsl_vector_free(super); gsl_vector_free(sub); 
    gsl_vector_free(f); gsl_vector_free(g);
};

void FiniteDiffSolver::operator()(VanillaOption *option)
{
    double multiplier = 1.0;
    if ( ! option->put )
    {
	multiplier = -1.0;
    }
    double dt = option->T/(double)Nt;
    double dZ = option->sigma*sqrt(3.0*dt);

    gsl_vector *tmp = NULL;
    // At the endpoint in time, each option is worth its intrinsic value
    for(int i = 0; i < Nz; i++)
    {
	double S = option->S0*exp(dZ*(double)(i-Nz/2));
	gsl_vector_set(g, i, std::max( multiplier*(option->K - S), 0.0));
    }

    // Prepare the vectors of the linear equation (tridiag system)
    gsl_vector_set(diag, 0, 1.0);
    gsl_vector_set(diag, Nz-1, 1.0);
    double sigmasqrd = option->sigma*option->sigma;
    for(int j = 1; j < Nz-1; j++)
    {
	double beta = 1.0 + dt*sigmasqrd/(dZ*dZ) + option->r*dt;
	double alpha = 0.5*dt*(option->r - sigmasqrd*0.5)/dZ - 0.5*dt*sigmasqrd/(dZ*dZ);
	double gamma2 = -0.5*dt*(option->r - sigmasqrd*0.5)/dZ - 0.5*dt*sigmasqrd/(dZ*dZ);
	gsl_vector_set(diag, j, beta);
	gsl_vector_set(super, j, gamma2);
	gsl_vector_set(sub, j-1, alpha);
    }

    // Now we work backwards in time, starting at second to last point
    double delta1 = 0.0, delta2 = 0.0;
    for(int i = Nt-1; i >= 0; i--)
    {
	// Boundary conditions vary for put and calls
	if(option->put){
	    gsl_vector_set(g, Nz-1, option->K);
	    gsl_vector_set(g, 0, 0.0);

	    gsl_vector_set(f, Nz-1, option->K);
	    gsl_vector_set(f, 0, 0.0);
	}
	else
	{
	    gsl_vector_set(g, 0, option->K);
	    gsl_vector_set(g, Nz-1, 0.0);

	    gsl_vector_set(f, 0, option->K);
	    gsl_vector_set(f, Nz-1, 0.0);
	}

	// Solve the equations
	gsl_linalg_solve_tridiag(diag, super, sub, g, f);

	// Check for early exercise
	for(int j = 1; j < Nz-1; j++)
	{
	    double S = option->S0*exp(dZ*(double)(j-Nz/2));
	    double continuation = gsl_vector_get(f, j);
	    if( option->american )
	    {
		double exercise = std::max(multiplier*(option->K-S), 0.0);
		gsl_vector_set(f, j, std::max(continuation, exercise));
	    }
	    else
	    {
		gsl_vector_set(f, j, continuation);
	    }
	}

	// Swap the vectors and iterate
	tmp = f;
	f = g;
	g = tmp;

	delta1 = gsl_vector_get(g, Nz/2) - gsl_vector_get(g, Nz/2 - 1);
	delta1 /= option->S0*(1.0 - exp(-dZ));

	delta2 = gsl_vector_get(g, Nz/2) - gsl_vector_get(g, Nz/2 + 1);
	delta2 /= option->S0*(1.0 - exp(dZ));
    }
 
    option->price = gsl_vector_get(g, Nz/2);
    option->delta = gsl_vector_get(g, Nz/2) - gsl_vector_get(g, Nz/2 - 1);
    option->delta /= option->S0*(1.0 - exp(-dZ));
    option->gamma = 2.0*(delta1 - delta2)/(option->S0*(exp(-dZ) - exp(dZ)));
};
