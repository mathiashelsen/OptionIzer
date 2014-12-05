#include "FiniteDiff.hpp"

FiniteDiff::FiniteDiff(
	    double _S0,
	    double _K,
	    double _sigma,
	    double _r,
	    double _T,
	    double _Smax,
	    int _Nz,
	    int _Nt)
{
    S0 = _S0;
    K = _K;
    sigma = _sigma;
    r = _r;
    T = _T;
    Nz = _Nz;
    Nt = _Nt;

    dt = T/(double)Nt;
    dZ = sigma*sqrt(3.0*dt);
}

void FiniteDiff::evaluate()
{
    int N = Nz;
    gsl_vector *f = gsl_vector_calloc(N);
    gsl_vector *g = gsl_vector_calloc(N);
    gsl_vector *diag = gsl_vector_calloc(N);
    gsl_vector *super = gsl_vector_calloc(N-1);
    gsl_vector *sub = gsl_vector_calloc(N-1);

    gsl_vector *tmp = NULL;


    // At the endpoint in time, each option is worth its intrinsic value
    for(int i = 0; i < N; i++)
    {
	double S = S0*exp(dZ*(double)(i-N/2));
	gsl_vector_set(g, i, std::max(K - S, 0.0));
    }

    // Prepare the vectors of the linear equation (tridiag system)
    gsl_vector_set(diag, 0, 1.0);
    gsl_vector_set(diag, N-1, 1.0);
    double sigmasqrd = sigma*sigma;
    for(int j = 1; j < N-1; j++)
    {
	double beta = 1.0 + dt*sigmasqrd/(dZ*dZ) + r*dt;
	double alpha = 0.5*dt*(r - sigmasqrd*0.5)/dZ - 0.5*dt*sigmasqrd/(dZ*dZ);
	double gamma = -0.5*dt*(r - sigmasqrd*0.5)/dZ - 0.5*dt*sigmasqrd/(dZ*dZ);
	gsl_vector_set(diag, j, beta);
	gsl_vector_set(super, j, gamma);
	gsl_vector_set(sub, j-1, alpha);
    }

    // Now we work backwards in time, starting at second to last point
    for(int i = Nt-1; i >= 0; i--)
    {
	gsl_vector_set(g, 0, K);
	gsl_vector_set(g, N-1, 0.0);

	gsl_vector_set(f, 0, K);
	gsl_vector_set(f, N-1, 0.0);

	// Solve the equations
	gsl_linalg_solve_tridiag(diag, super, sub, g, f);

	// Check for early exercise
	for(int j = 1; j < N-1; j++)
	{
	    double S = S0*exp(dZ*(double)(j-N/2));
	    double continuation = gsl_vector_get(f, j);
	    double exercise = std::max(K-S, 0.0);
	    gsl_vector_set(f, j, std::max(continuation, exercise));
	}

	// Swap the vectors and iterate
	tmp = f;
	f = g;
	g = tmp;
    }
 
    price = gsl_vector_get(g, N/2);
    gsl_vector_free(diag); gsl_vector_free(super); gsl_vector_free(sub); 
    gsl_vector_free(f); gsl_vector_free(g);
}
