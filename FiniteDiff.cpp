#include "FiniteDiff.hpp"

FiniteDiff::FiniteDiff(
	    double _S0,
	    double _K,
	    double _sigma,
	    double _r,
	    double _T,
	    double _Smax,
	    int _Ns,
	    int _Nt)
{
    S0 = _S0;
    K = _K;
    sigma = _sigma;
    r = _r;
    T = _T;
    Smax = _Smax;
    Ns = _Ns;
    Nt = _Nt;

    dS = S0/(double)Ns;
    dt = T/(double)Nt;
}

void FiniteDiff::evaluate()
{
    int N = (int)(Smax/dS);
    std::cout << N << std::endl;
    gsl_vector *f = gsl_vector_calloc(N);
    gsl_vector *g = gsl_vector_calloc(N);
    gsl_vector *diag = gsl_vector_calloc(N);
    gsl_vector *super = gsl_vector_calloc(N-1);
    gsl_vector *sub = gsl_vector_calloc(N-1);

    gsl_vector *tmp = NULL;

    // At the endpoint in time, each option is worth its intrinsic value
    for(int i = 0; i < N; i++)
    {
	gsl_vector_set(g, i, std::max(K - dS*(double)i, 0.0));
    }

    double sigmasqrd = sigma*sigma;
    // Now we work backwards in time, starting at second to last point
    for(int i = Nt-2; i >= 0; i--)
    {
	gsl_vector_set(f, 0, K);
	gsl_vector_set(f, N-1, 0.0);

	gsl_vector_set(diag, 0, 1.0);
	gsl_vector_set(diag, N-1, 1.0);
	for(int j = 1; j < N-1; j++)
	{
	    gsl_vector_set(diag, j, 1.0 + sigmasqrd*(double)j*(double)j*dt + r*dt);
	    gsl_vector_set(super, j, -0.5*(r + sigmasqrd*(double)j)*dt*(double)j);
	    gsl_vector_set(sub, j-1, 0.5*(r - sigmasqrd*(double)j)*dt*(double)j);
	}

	// Solve the equations
	gsl_linalg_solve_tridiag(diag, super, sub, g, f);

	// Check for early exercise
	for(int j = 1; j < N-1; j++)
	{
	    gsl_vector_set(f, j, std::max(gsl_vector_get(f, j), K - dS*(double)j));
	}

	// Swap the vectors and iterate
	tmp = f;
	f = g;
	g = tmp;
    }
 
    std::cout << Ns << std::endl; 
    price = gsl_vector_get(g, Ns);
    std::cout << price << "\n";
    gsl_vector_free(diag); gsl_vector_free(super); gsl_vector_free(sub); 
    gsl_vector_free(f); gsl_vector_free(g);
}
