#include <iostream>
#include <cmath>
#include <iomanip> 
#include <chrono>   // for timing
#include <cuda.h>   
#include <fstream>

#define M_PI 3.14276   // pi
#define c 299792458    // speed of light
#define mu0 M_PI*4e-7  // permeability of free space (vacuum)
#define eta0 c*mu0     // intrinsic impedance of free space

double** declare_array2D(int NX, int NY) {
    double** V = new double* [NX]; // Allocate memory for rows
    for (int x = 0; x < NX; x++) {
        V[x] = new double[NY];     // Allocate memory for columns in each row
    }

    for (int x = 0; x < NX; x++) { // Initialise all elements of array to zero
        for (int y = 0; y < NY; y++) {
            V[x][y] = 0;
        }   
    }
    return V;
}

__global__ void injectSource(double* V1, double* V2, double* V3, double* V4, 
                             int EinX, int EinY, int stride, double E0) {
    // Only 1 thread does injection
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int idx = EinX * stride + EinY;  // flattened index, V[x][y], x => row
        V1[idx] += E0;
        V2[idx] -= E0;
        V3[idx] -= E0;
        V4[idx] += E0;
    }
}

__global__ void scatter(double* V1, double* V2, double* V3, double* V4, 
                        double Z, double I, int stride, int flatArraySize) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= flatArraySize) return;

    __shared__ double sV1[1024], sV2[1024], sV3[1024], sV4[1024];

    sV1[threadIdx.x] = V1[idx];    
    sV2[threadIdx.x] = V2[idx];
    sV3[threadIdx.x] = V3[idx];    
    sV4[threadIdx.x] = V4[idx];

    __syncthreads();
    
    int x = idx / stride; // [x][] which row 
    int y = idx % stride; // [][y] column position in row
    double tempV = 0;     // temp voltage for swap process

    // calc total incident current at node
    I = (2 * sV1[(x*stride + y)%blockDim.x] + 2 * sV4[(x*stride + y)%blockDim.x] - 
         2 * sV2[(x*stride + y)%blockDim.x] - 2 * sV3[(x*stride + y)%blockDim.x]) / (4 * Z);

    // // update voltages based on scattering equations
    tempV = 2 * sV1[(x*stride + y)%blockDim.x] - I * Z;         //port1
            sV1[(x*stride + y)%blockDim.x] = tempV - sV1[(x*stride + y)%blockDim.x];
    tempV = 2 * sV2[(x*stride + y)%blockDim.x] + I * Z;         //port2
            sV2[(x*stride + y)%blockDim.x] = tempV - sV2[(x*stride + y)%blockDim.x];
    tempV = 2 * sV3[(x*stride + y)%blockDim.x] + I * Z;         //port3
            sV3[(x*stride + y)%blockDim.x] = tempV - sV3[(x*stride + y)%blockDim.x];
    tempV = 2 * sV4[(x*stride + y)%blockDim.x] - I * Z;         //port4
            sV4[(x*stride + y)%blockDim.x] = tempV - sV4[(x*stride + y)%blockDim.x];

    __syncthreads();

    V1[idx] = sV1[threadIdx.x];    
    V2[idx] = sV2[threadIdx.x];
    V3[idx] = sV3[threadIdx.x];    
    V4[idx] = sV4[threadIdx.x];
}

__global__ void connect(double* V1, double* V2, double* V3, double * V4, 
                        int stride, int flatArraySize) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= flatArraySize) return;

    int x = idx / stride; // [x][] which row 
    int y = idx % stride; // [][y] column position in row
    double tempV = 0;
    
    if(x >= 1) {
        tempV = V2[x*stride + y];                  // temporarily store V2
        V2[x*stride + y] = V4[((x-1)*stride) + y]; // replace V2 with left of V4
        V4[((x-1)*stride) + y] = tempV;            // update left of V4 with old V2
    }
    if(y >= 1) {
        tempV = V1[x*stride + y];                  // temporarily store V1
        V1[x*stride + y] = V3[(x*stride) + (y-1)]; // replace V1 with lower of V4
        V3[(x*stride) + (y-1)] = tempV;            // update lower of V3 with old V1
    }
}

__global__ void boundary(double* V1, double* V2, double* V3, double * V4, 
                         int stride, int NY, int flatArraySize, 
                         double rXmin, double rXmax, double rYmin, double rYmax) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= stride) return;

    int x = idx; // [x][] which row 
    int y = idx; // [][y] column position in row
    
    V3[(x*stride) + (NY-1)] = rYmax * V3[(x*stride) + (NY-1)]; // scale v3 for top boundary
    V1[x*stride] = rYmin * V1[x*stride];                       // scale v1 for bottom boundary
    
    V4[((stride-1)*stride) + y] = rXmax * V4[((stride-1)*stride) + y]; // scale v4 for right boundary
    V2[y] = rXmin * V2[y];                                             // scale v2 for left boundary
}

__global__ void saveOutput(int n, double dt, double* output, double* V2, double* V4,
                           int EoutX, int EoutY, int stride) {
    output[n*2] =  n * dt;
    output[(n*2)+1] = V2[EoutX*stride + EoutY] + V4[EoutX*stride + EoutY];
}
// --------------------------------------------------------------------------------------------------
// 
// --------------------------------------------------------------------------------------------------
int main () {
    std::clock_t start = std::clock();

    int NX = 100;  // mesh size x-axis
    int NY = 100;  // mesh size y-axis
    int NT = 8192; // time steps
    double dl = 1; // spatial step size - distance between adjacent nodes
    double dt = dl / (sqrt(2.) * c); // time step size

    // 2D mesh variables
    double I = 0;     // incident current 
    double E0 = 0;    // source excitation voltage

    double** h_V1 = declare_array2D(NX, NY);
    double** h_V2 = declare_array2D(NX, NY);
    double** h_V3 = declare_array2D(NX, NY);
    double** h_V4 = declare_array2D(NX, NY);

    double Z = eta0 / sqrt(2.); // characteristic impedance

    // boundary reflection coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    // input / output
    double width = 20 * dt * sqrt(2.); // wave pulse width
    double delay = 100 * dt * sqrt(2.); // wave peak amplitude delay
    int Ein[] = { 10,10 }; // coordinates of wave source injection on mesh
    int Eout[] = { 15,15 }; // coordinates of where measurements to be took on mesh
    
    double* h_output = (double*) malloc(sizeof(double)*NT*2); // for V2 and V4 output per NT
    double* d_output; // for V2 and V4 output per NT

    // h_V1, h_V2, h_V3 h_V4 -> h_flatArrayV1, h_flatArrayV2, h_flatArrayV3, h_flatArrayV4
    int flatArraySize = NX*NY;
    double* h_flatArrayV1 = (double*) malloc(sizeof(double)*flatArraySize);
    double* h_flatArrayV2 = (double*) malloc(sizeof(double)*flatArraySize);
    double* h_flatArrayV3 = (double*) malloc(sizeof(double)*flatArraySize);
    double* h_flatArrayV4 = (double*) malloc(sizeof(double)*flatArraySize);

    // memory for device 
    double* d_V1; 
    double* d_V2; 
    double* d_V3; 
    double* d_V4;

    if(cudaMalloc((void**)&d_V1, sizeof(double) * flatArraySize) != cudaSuccess) {
            std::cout << "Error allocating memory d_V1!" << std::endl;
            return 0;
    }
    cudaMemset(d_V1, 0, sizeof(double) * flatArraySize);

    if(cudaMalloc((void**)&d_V2, sizeof(double) * flatArraySize) != cudaSuccess) {
            std::cout << "Error allocating memory d_V2!" << std::endl;
            return 0;
    }
    cudaMemset(d_V2, 0, sizeof(double) * flatArraySize);

    if(cudaMalloc((void**)&d_V3, sizeof(double) * flatArraySize) != cudaSuccess) {
            std::cout << "Error allocating memory d_V3!" << std::endl;
            return 0;
    }
    cudaMemset(d_V3, 0, sizeof(double) * flatArraySize);

    if(cudaMalloc((void**)&d_V4, sizeof(double) * flatArraySize) != cudaSuccess) {
            std::cout << "Error allocating memory d_V4!" << std::endl;
            return 0;
    }
    cudaMemset(d_V4, 0, sizeof(double) * flatArraySize);

    if(cudaMalloc((void**)&d_output, sizeof(double) * NT*2) != cudaSuccess) {
            std::cout << "Error allocating memory d_output!" << std::endl;
            return 0;
    }
    cudaMemset(d_output, 0, sizeof(double) * NT*2);
    // --------------------------------------------------------------------------------------------------
    // 
    // --------------------------------------------------------------------------------------------------
    std::ofstream output("outputCu.out", std::ios::app);

    if (!output.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return 0;
    }
    // --------------------------------------------------------------------------------------------------
    // 
    // --------------------------------------------------------------------------------------------------
    for (int n = 0; n < NT; n++) { // for NT time steps
        // source injection
        // calc pulse value at the current time step 'n'
        E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
        
        injectSource<<<1,1>>>(d_V1, d_V2, d_V3, d_V4, Ein[0], Ein[1], NX, E0);

        cudaDeviceSynchronize(); // Synchronize the device

        int totalThreads = flatArraySize;
        int threadsPerBlock = 1024;        
        int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock; // integer division ceiling
        
        scatter<<<numBlocks, threadsPerBlock>>>(d_V1, d_V2, d_V3, d_V4, Z, I, NX, flatArraySize);

        cudaDeviceSynchronize(); // Synchronize the device

        connect<<<numBlocks, threadsPerBlock>>>(d_V1, d_V2, d_V3, d_V4, NX, flatArraySize);
         
        cudaDeviceSynchronize(); // Synchronize the device

        totalThreads = NX;
        numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock; // integer division ceiling

        boundary<<<numBlocks, threadsPerBlock>>>(d_V1, d_V2, d_V3, d_V4, 
                                                 NX, NY, flatArraySize, rXmin, rXmax, rYmin, rYmax);

        cudaDeviceSynchronize(); // Synchronize the device

        saveOutput<<<1,1>>>(n, dt, d_output, d_V2, d_V4, Eout[0], Eout[1], NX);

        if (n % 100 == 0)
            std::cout << n << std::endl;
    }
    // --------------------------------------------------------------------------------------------------
    // 
    // --------------------------------------------------------------------------------------------------
    // copy memeory from device to host
    if(cudaMemcpy(h_flatArrayV1, d_V1, sizeof(double) * flatArraySize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Error copying memory from d_V1!" << std::endl;
        cudaFree(d_V1);
        cudaFree(d_V2);
        cudaFree(d_V3);
        cudaFree(d_V4);
        cudaFree(d_output);
        return 0;
    }

    if(cudaMemcpy(h_flatArrayV2, d_V2, sizeof(double) * flatArraySize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Error copying memory from d_V2!" << std::endl;
        cudaFree(d_V1);
        cudaFree(d_V2);
        cudaFree(d_V3);
        cudaFree(d_V4);
        cudaFree(d_output);
        return 0;
    }

    if(cudaMemcpy(h_flatArrayV3, d_V3, sizeof(double) * flatArraySize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Error copying memory from d_V3!" << std::endl;
        cudaFree(d_V1);
        cudaFree(d_V2);
        cudaFree(d_V3);
        cudaFree(d_V4);
        cudaFree(d_output);
        return 0;
    }

    if(cudaMemcpy(h_flatArrayV4, d_V4, sizeof(double) * flatArraySize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Error copying memory from d_V4!" << std::endl;
        cudaFree(d_V1);
        cudaFree(d_V2);
        cudaFree(d_V3);
        cudaFree(d_V4);
        cudaFree(d_output);
        return 0;
    }

    if(cudaMemcpy(h_output, d_output, sizeof(double)*NT*2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Error copying memory from d_output!" << std::endl;
        cudaFree(d_V1);
        cudaFree(d_V2);
        cudaFree(d_V3);
        cudaFree(d_V4);
        cudaFree(d_output);
        return 0;
    }

    // copy 1dArray to 2dArray 
    for(int x = 0; x < NX; ++x) {
        for(int y = 0; y < NY; ++y) {
            h_V1[x][y] = h_flatArrayV1[x*NX + y];
            h_V2[x][y] = h_flatArrayV2[x*NX + y];
            h_V3[x][y] = h_flatArrayV3[x*NX + y];
            h_V4[x][y] = h_flatArrayV4[x*NX + y];
        }
    }

    // copy output from array to file
    for(int i = 0; i < NT; ++i) {
        output << h_output[i*2] << "  " << h_output[i*2+1] << std::endl;
    }
    // --------------------------------------------------------------------------------------------------
    // 
    // --------------------------------------------------------------------------------------------------
    output.close();
    std::cout << "Done";
    std::cout << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << '\n';

    free(h_V1);
    free(h_V2);
    free(h_V3);
    free(h_V4);

    free(h_flatArrayV1);
    free(h_flatArrayV2);
    free(h_flatArrayV3);
    free(h_flatArrayV4);
    free(h_output);

    cudaFree(d_V1);
    cudaFree(d_V2);
    cudaFree(d_V3);
    cudaFree(d_V4);
    cudaFree(d_output);

    return 0;
}
/*// --------------------------------------------------------------------------------------------------
// Testing - A successful run with current parameters outputs:
// --------------------------------------------------------------------------------------------------
0
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
2000
2100
2200
2300
2400
2500
2600
2700
2800
2900
3000
3100
3200
3300
3400
3500
3600
3700
3800
3900
4000
4100
4200
4300
4400
4500
4600
4700
4800
4900
5000
5100
5200
5300
5400
5500
5600
5700
5800
5900
6000
6100
6200
6300
6400
6500
6600
6700
6800
6900
7000
7100
7200
7300
7400
7500
7600
7700
7800
7900
8000
8100
Done3.716
// --------------------------------------------------------------------------------------------------
// 
// -------------------------------------------------------------------------------------------------*/