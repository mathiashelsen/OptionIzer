#ifndef _CUDA_MC_LOOKBACK_SOLVER_CUH
#define _CUDA_MC_LOOKBACK_SOLVER_CUH

#include "CUDA_MC_Solver.hpp"


template<LookBackType type> __global__ void LookBackKernel(float *_x, float *_assets, float *_payoffs,
    float r,
    float S0,
    float K,
    float sigma,
    float T,
    float call,
    int Nseries,
    int Nsteps)
{
    int pathIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if( pathIndex >= Nseries )
	return;
    
    _assets[pathIndex] = S0;
    float extr = S0;
    for(int i = 0; i < Nsteps; i++)
    {
	_assets[pathIndex] *= (1.0 + r + sigma*_x[pathIndex*Nsteps + i]);
	switch(type)
	{
	    case LookBackTypeMin:   extr = (_assets[pathIndex] < extr) ? _assets[pathIndex] : extr;
				    break;
	    case LookBackTypeMax:   extr = (_assets[pathIndex] > extr) ? _assets[pathIndex] : extr;
				    break;
	}
    }

    float tmp = call*(extr - K)*exp(-r*T);
    _payoffs[pathIndex] = (tmp > 0.0f) ? tmp : 0.0f;

    __syncthreads();

}

#endif
