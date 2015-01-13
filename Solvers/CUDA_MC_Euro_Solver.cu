#include "CUDA_MC_Euro_Solver.hpp"

typedef struct
{
    float r;
    float S0;
    float K;
    float *call;
    int Nseries;
    int Nsteps; 
} params;

__global__ void EuroKernel( float *_x, float *_assets, float *_payoffs,
    float r,
    float S0,
    float K,
    float *call,
    int Nseries,
    int Nsteps )
{
    int pathIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if( pathIndex >= Nseries )
	return;
    
    _assets[pathIndex*Nsteps] = S0;

    for(int i = 0; i < Nsteps; i++)
    {
	_assets[pathIndex*Nsteps + i + 1] = _assets[pathIndex*Nsteps + i]*(1.0 + _x[pathIndex*Nsteps + i]);
    }
    float tmp = _assets[pathIndex*(Nsteps + 1) - 1] - K;
    _payoffs[pathIndex] = (tmp > 0.0) ? tmp : 0.0;
    
}

CUDA_MC_Euro_Solver::CUDA_MC_Euro_Solver(int _Nseries, int _Nsteps)
{
    Nseries = _Nseries;
    Nsteps = _Nsteps; 
}

void CUDA_MC_Euro_Solver::operator()(VanillaOption *option)
{
    float *returns, *assets, *payoffs;
    float scale = (float)option->T/(float)Nsteps;

    curandGenerator_t gen;

    cudaMalloc( (void **) &returns, Nseries*Nsteps*sizeof(float) );
    cudaMalloc( (void **) &assets, Nseries*(Nsteps+1)*sizeof(float) );
    cudaMalloc( (void **) &payoffs, Nseries*sizeof(float) );

    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );
    curandGenerateNormal( gen, returns, Nseries*Nsteps, option->r*scale, option->sigma*sqrt(scale) );


    curandDestroyGenerator( gen );
    cudaFree( returns );
    cudaFree( assets );
};
