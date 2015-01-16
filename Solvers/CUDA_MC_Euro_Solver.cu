#include "CUDA_MC_Solver.hpp"

__global__ void EuroKernel( float *_x, float *_assets, float *_payoffs,
    float r,
    float S0,
    float K,
    float sigma,
    float T,
    float call,
    int Nseries,
    int Nsteps )
{
    int pathIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if( pathIndex >= Nseries )
	return;
    
    _assets[pathIndex] = S0;

    for(int i = 0; i < Nsteps; i++)
    {
	_assets[pathIndex] *= (1.0 + r + sigma*_x[pathIndex*Nsteps + i]);
    }
    float tmp = call*(_assets[pathIndex] - K)*exp(-r*T);
    _payoffs[pathIndex] = (tmp > 0.0f) ? tmp : 0.0f;

    __syncthreads();
}

CUDA_MC_Solver<EuroOption>::CUDA_MC_Solver(int _Nseries, int _Nsteps)
{
    Nseries = _Nseries;
    Nsteps = _Nsteps; 

    std::cout << returns << std::endl;
    assert( cudaMalloc( (void **) &returns, Nseries*Nsteps*sizeof(float) ) == cudaSuccess);
    std::cout << returns << std::endl;
    assert( cudaMalloc( (void **) &assets, Nseries*sizeof(float) ) == cudaSuccess);
    assert( cudaMalloc( (void **) &payoffs, Nseries*sizeof(float) ) == cudaSuccess);

    assert( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_MTGP32 ) == CURAND_STATUS_SUCCESS);
    assert( curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ) == CURAND_STATUS_SUCCESS);
    assert( curandGenerateNormal( gen, returns, Nseries*Nsteps, 0.0, 1.0 ) == CURAND_STATUS_SUCCESS);
}

CUDA_MC_Solver<EuroOption>::~CUDA_MC_Solver()
{
    curandDestroyGenerator( gen );
    cudaFree( returns );
    cudaFree( assets );
};

void CUDA_MC_Solver<EuroOption>::operator()(EuroOption *option)
{

    float *localPayoffs = new float[Nseries];
    assert( localPayoffs );
    float scale = (float)option->T/(float)Nsteps;

    float call = 1.0;
    if(option->put)
	call = -1.0;

    int threadsPerBlock = 256;
    int nBlocks = Nseries/threadsPerBlock;
    EuroKernel<<<nBlocks, threadsPerBlock>>>(returns, 
					    assets, 
					    payoffs, 
					    (float)option->r, 
					    (float)option->S0, 
					    (float)option->K, 
					    (float) option->sigma*scale, 
					    (float) option->T,
					    call,
					    Nseries, 
					    Nsteps);
    
    cudaThreadSynchronize();
    cudaMemcpy( (void *) localPayoffs, payoffs, sizeof(float)*Nseries, cudaMemcpyDeviceToHost );

    // This ends all the calls to cuda, now just averaging over the payoffs
    double avg;
    for(int i = 0; i < Nseries; i++)
    {
	avg += (double) localPayoffs[i];
    }
    avg /= (double) Nseries;
    option->price = avg;
    delete[] localPayoffs;
};
