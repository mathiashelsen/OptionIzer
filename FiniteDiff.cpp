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
    Smax = _Smax
    Ns = _Ns;
    Nt = _Nt;

    dS = S0/(double)Ns;
    dt = T/(double)Nt;
}

void FiniteDiff::evaluate()
{
    int N = (int)(Smax/dS);
    double *f = new double[N];
    double *g = new double[N];
    double **A = new double*[N];
    for(int i = 0; i < N; i++)
    {
	A[i] = new double[N];
    }

    double *tmp = NULL;

    // At the endpoint in time, each option is worth its intrinsic value
    for(int i = 0; i < N; i++)
    {
	g[i] = std::max(K - dS*(double)i, 0.0);
    }

    double sigmasqrd = sigma*sigma;
    // Now we work backwards in time, starting at second to last point
    for(int i = Nt-2; i >= 0; i++)
    {
	f[i][0] = K;
	f[i][N-1] = 0.0;

	A[0][0] = 1.0;
	A[N-1][N-1] = 1.0;

	for(int j = 1; j < N-1; j++)
	{
	    A[j][j-1] = 0.5*(r - sigmasqrd*(double)j)*dt*(double)j;
	    A[j][j] = 1.0 + sigmasqrd*(double)j*(double)j*dt + r*dt;
	    A[j][j+1] = -0.5*(r + sigmasqrd*(double)j)*dt*(double)j;
	}	

	for(int j = 1; j < N-1; j++)
	{

	}

    }
    
    for(int i = 0; i < N; i++)
    {
	delete[] A[i];
    }
    delete[] A;
    delete f;
    delete g;
}
