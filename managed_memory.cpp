/*
Author: Abby McCollam
Class: ECE4122 Section A
Last Date Modified: 11/09/23
Description: Managed memory file using CUDA taking in number of walkers and number of steps for each walker performing
calculations with three different types of memory models.
*/

//PREPROCESSOR DIRECTIVES-----------------------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

using namespace std;

//RANDOM WALK FUNCTION--------------------------------------------------------------------------------------------------
//arguments: array storing distances, number of steps, number of walkers, random seed
__global__ void randomWalk(float* steps, int numSteps, int numWalkers, unsigned int seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //indexes in each dimension

    if (tid < numWalkers)
    {
        curandState state; // initializes random number generator
        curand_init(seed, tid, 0, &state);

        float xPos = 0, yPos = 0; //current position of walker

        for (int i = 0; i < numSteps; ++i)
        {
            float random_float = curand_uniform(&state); // creates random float in [0.0 1.0]
            int direction = static_cast<int>(random_float * 4); //converts to integer in [0 3]
            if (direction == 0) yPos++; // 0: up
            else if (direction == 1) yPos--; //1: down
            else if (direction == 2) xPos--; //2: left
            else xPos++; //3: right
        }

        //Calculating distance of each walker from the origin
        steps[tid] = sqrt(xPos * xPos + yPos * yPos);
    }
}

// FUNCTION CALCULATING AVERAGE DISTANCE FROM ORIGIN--------------------------------------------------------------------
//array storing distances, average distance from origin calculated by kernel, number of walkers
__global__ void calculateAverageDistance(float* steps, float* avgDistance, int numWalkers)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Create a shared variable to accumulate the results
    __shared__ float sharedAvgDistance;
    sharedAvgDistance = 0.0;

    if (tid < numWalkers)
    {
        sharedAvgDistance += static_cast<float>(steps[tid]); //accumulating distances of individual walkers

        // Syncing threads
        __syncthreads();

        if (threadIdx.x == 0)
        {
            // first thread summing avg distances
            atomicAdd(avgDistance, sharedAvgDistance / numWalkers);
        }
    }
}

int main(int argc, char** argv)
{
    // INITIALIZING VARIABLES-------------------------------------------------------------------------------------------
    unsigned int seed = time(NULL);
    int numBlocks;
    int blockSize = 1;
    int numWalkers = 10000000;
    int numSteps = 10000;
    chrono::time_point<chrono::high_resolution_clock> start, stop;

    if (argc != 5) //INPUT CHECKING-------------------------------------------------------------------------------------
    {
        cerr << "Usage: " << argv[0] << " -W <num_walkers> -I <num_steps>" << endl;
        return 1;
    }

    if (numWalkers <= 0 || numSteps <= 0) //INPUT CHECKING--------------------------------------------------------------
    {
        cerr << "Not a correct input." << endl;
        return 1;
    }

    for (int i = 1; i < argc; i += 2) // taking in arguments for number of walkers and number of steps
    {
        if (argv[i][0] == '-' && argv[i][1] == 'W')
            numWalkers = atoi(argv[i + 1]);
        else if (argv[i][0] == '-' && argv[i][1] == 'I')
            numSteps = atoi(argv[i + 1]);
    }

    // calculating number of blocks
    numBlocks = (numWalkers + blockSize - 1) / blockSize;

    // SETTING UP AND CLEANING CUDA DEVICE------------------------------------------------------------------------------
    cudaSetDevice(0);
    cudaDeviceReset();

    //WARMING UP--------------------------------------------------------------------------------------------------------
    float* warmUp = nullptr;
    cudaMalloc((void**)&warmUp, numWalkers * sizeof(float));
    cudaFree(warmUp);

    // TIME CUDA MALLOC-------------------------------------------------------------------------------------------------
    //starting time
    start = chrono::high_resolution_clock::now();

    //declaring #steps and average distance pointers and initializing to null
    float* stepsMalloc = nullptr;
    float* avgDistanceDeviceMalloc = nullptr;

    //allocating device memory
    cudaMalloc((void**)&stepsMalloc, numWalkers * sizeof(int));
    cudaMalloc((void**)&avgDistanceDeviceMalloc, sizeof(float));

    //calling kernel functions to perform calculations
    randomWalk <<<numBlocks, blockSize>>> (stepsMalloc, numSteps, numWalkers, seed); //CUDA kernel launch
    calculateAverageDistance <<<numBlocks, blockSize>>> (stepsMalloc, avgDistanceDeviceMalloc, numWalkers);

    //copying data from device to host
    float avgDistanceMalloc;
    cudaMemcpy(&avgDistanceMalloc, avgDistanceDeviceMalloc, sizeof(float), cudaMemcpyDeviceToHost);

    //freeing allocated memory
    cudaFree(stepsMalloc);
    cudaFree(avgDistanceDeviceMalloc);

    //stopping time
    stop = chrono::high_resolution_clock::now();
    auto mallocTotalTime = chrono::duration_cast<chrono::microseconds>(stop - start);

    //printing results to terminal
    cout << "Lab4 -W 1000 -I 10000" << endl;
    cout << "Normal CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << mallocTotalTime.count() << endl;
    cout << "    Average distance from origin: " << avgDistanceMalloc << endl;

    // TIME CUDA MALLOC HOST (REPEAT LIKE ABOVE)-----------------------------------------------------------------------------------
    //starting time
    start = chrono::high_resolution_clock::now();

    //declaring #steps and average distance pointers and initializing to null
    float* stepsHostAlloc = nullptr;
    float* avgDistanceDeviceHostAlloc = nullptr;

    //allocating device memory
    cudaHostAlloc((void**)&stepsHostAlloc, numWalkers * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&avgDistanceDeviceHostAlloc, sizeof(float), cudaHostAllocDefault);

    //calling kernel functions to perform calculations
    randomWalk <<<numBlocks, blockSize>>> (stepsHostAlloc, numSteps, numWalkers, seed);
    calculateAverageDistance <<<numBlocks, blockSize>>> (stepsHostAlloc, avgDistanceDeviceHostAlloc, numWalkers);

    //copying data from device to host
    float avgDistanceHostAlloc;
    cudaMemcpy(&avgDistanceHostAlloc, avgDistanceDeviceHostAlloc, sizeof(float), cudaMemcpyDeviceToHost);

    //freeing allocated memory
    cudaFree(stepsHostAlloc);
    cudaFree(avgDistanceDeviceHostAlloc);

    //stopping time
    stop = chrono::high_resolution_clock::now();
    auto HostTotalTime = chrono::duration_cast<chrono::microseconds>(stop - start);

    //printing results to terminal
    cout << "Pinned CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << HostTotalTime.count() << endl;
    cout << "    Average distance from origin: " << avgDistanceHostAlloc << endl;

    // TIME CUDA MALLOC MANAGED (REPEAT LIKE ABOVE)--------------------------------------------------------------------------------
    //starting time
    start = chrono::high_resolution_clock::now();

    //declaring #steps and average distance pointers and initializing to null
    float* stepsMallocManaged = nullptr;
    float* avgDistanceDeviceMallocManaged = nullptr;

    //allocating device memory
    cudaMallocManaged((void**)&stepsMallocManaged, numWalkers * sizeof(int));
    cudaMallocManaged((void**)&avgDistanceDeviceMallocManaged, sizeof(float));

    //calling kernel functions to perform calculations
    randomWalk <<<numBlocks, blockSize>>> (stepsMallocManaged, numSteps, numWalkers, seed);
    calculateAverageDistance <<<numBlocks, blockSize>>> (stepsMallocManaged, avgDistanceDeviceMallocManaged, numWalkers);

    //copying data from device to host
    float avgDistanceMallocManaged;
    cudaMemcpy(&avgDistanceMallocManaged, avgDistanceDeviceMallocManaged, sizeof(float), cudaMemcpyDeviceToHost);

    //freeing allocated memory
    cudaFree(stepsMallocManaged);
    cudaFree(avgDistanceDeviceMallocManaged);

    //stopping time
    stop = chrono::high_resolution_clock::now();
    auto ManagedTotalTime = chrono::duration_cast<chrono::microseconds>(stop - start);

    //printing results to terminal
    cout << "Managed CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): " << ManagedTotalTime.count() << endl;
    cout << "    Average distance from origin: " << avgDistanceMallocManaged << endl;
    cout << "Bye" << endl;

    return 0;
}