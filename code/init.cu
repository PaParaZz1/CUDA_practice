const int Nx = 12;
const int Ny = 6;

dim3 threadsPerBlock(4, 3, 1);
dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y, 1)

// kernel definition
__global__ void matrixAdd(float A[Ny][Nx], float B[Ny][Nx], float C[Ny][Nx]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    C[j][i] = A[j][i] + B[j][i]
}
// __global__ denote a CUDA kernel function

__device__ float doubleValue(float x) {
    return 2 * x;
}
// __device__ SPMD execution on GPU

Nx = 11;
Ny = 6;
dim3 numBlocks((Nx+threadsPerBlock.x-1)/threadsPerBlock.x,
               (Ny+threadsPerBlock.y-1)/threadsPerBlock.y, 1);
__global__ void matrixAddDoubleB(float A[Ny][Nx], float B[Ny][Nx], float C[Ny][Nx]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<Nx && j<Ny) {
        C[j][i] = A[j][i] + B[j][i]
    }
}

const int N = 100;
float* A = new float[N];
for (int i=0; i<N; ++i) {
    A[i] = (float)i;
}

int bytes = sizeof(float) * N;
float* deviceA;
cudaMalloc(&deviceA, bytes);
cudaMemcpy(deviceA, A, bytes, cudaMemcpyHostToDevice);
// deviceA[i] is an invalid op, deviceA is not a pointer into the host's address space


// 3 distince types of address space
// device global memory(all threads)
// per-block shared memory(all threads in block)
// per-thread private memory(thread)

#define THREADS_PER_BLK 128
__global__ void convolve(int N, float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i=0; i<3; ++i) {
        result += input[index + i];
    }
    output[index] = result / 3.f;
}

int N = 1024*1024;
cudaMalloc(&devInput, sizeof(float) * (N+2));
cudaMalloc(&devOutput, sizeof(float) * N);

//convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput)
__global__ void convolve_shared(int N, float *input, float *output) {
    __shared__ float support[THREADS_PER_BLK+2];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    support[threadIdx.x] = input[index];
    if (threadIdx.x < 2) {
        support[THREADS_PER_BLK + threadIdx.x] = input[index + THREADS_PER_BLK];
    }
    __syncthreads();
    
    float result = 0.0f;
    for (int i=0; i<3; ++i) {
        result += support[threadIdx.x + i];
    }
    output[index] = result / 3.f;
}
// __syncthreads: wait for all the threads in the same block to arrive at this point

//float atomicAdd(float* addr, float amount);
//atomic op on both global memory and per-block shared memory

//major CUDA assumption: thread block execution can be carried out in any order
//GPU implemenation map thread blocks to cores using a dynamic scheduling policy
//warps are an important GPU implemenation detail, but not a CUDA abstraction
