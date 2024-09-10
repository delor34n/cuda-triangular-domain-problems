#ifndef KERNELS_H
#define KERNELS_H

__global__ void inverseSquareRoot_Method(float* matrix, int N);
__global__ void boundingBox_Method(float* matrix, int N, int n, int offsetx, int offsety);
__global__ void dynamicParallelism_Method(float* matrix, int N, int n, int offsetx, int offsety, int depth, int MAXDEPTH, int BLOCKSIZE);
__global__ void flatRecursive(float* matrix, int N, int offset );
__global__ void flatRecursiveBasic(float* matrix, int N, int col );
__device__ int cost_function();

__device__ int cost_function() {
    return 1;
}

/**
*   Método flat-recursive - @Author:CristobalNavarro
*/
__global__ void flatRecursive(float* matrix, int N, int offset ){
    // los bloques que contienen parte de la diagonal, no es lo mismo que la diagonal pura, O(n)
    if(blockIdx.y == 0){
        if(threadIdx.y >= threadIdx.x){
            const int tx = blockIdx.x*blockDim.x + threadIdx.x;
            const int ty = blockIdx.x*blockDim.x + threadIdx.y;
            matrix[ty*N + tx] = matrix[(ty+offset)*N + tx+offset] = cost_function();
        }
        return;
    }
    // todo el resto O(n^2)
    uint2 w = {blockIdx.x, blockIdx.y};
    unsigned int b = 1 << (31 - __clz(w.y)); 
    unsigned int q = w.x/b; 
    uint2 m = {w.x + q*b, w.y + ((q*b) << 1)};
    matrix[(threadIdx.y + m.y*blockDim.y)*N + threadIdx.x + m.x*blockDim.x] = cost_function();
}

/**
*   Método flat-recursive basic: primera implementación
*/
__global__ void flatRecursiveBasic(float* matrix, int N, int col ){
    if (blockIdx.y == 0){
        if(threadIdx.y >= threadIdx.x){
            const int tx = blockIdx.x*blockDim.x + threadIdx.x;
            const int ty = blockIdx.x*blockDim.x + threadIdx.y;
            matrix[ty*N + tx] = cost_function();
            matrix[(ty+col)*N + tx+col] = cost_function();
        }
        return;
    }

    int b, q, wx, wy;
    
    b =  1 << (int) trunc(log2f(blockIdx.y));
    float alpha = blockIdx.x/b;
    q = (int) trunc(alpha);

    wx = blockIdx.x + q*b;
    wy = blockIdx.y + 2*q*b;

    int index = ((wx * blockDim.x)+threadIdx.x) + ((wy *blockDim.y)+ threadIdx.y)*N;
    matrix[index] = cost_function();
}

/**
*   Método DynamicParallelism (DP)
*/
__global__ void dynamicParallelism_Method(float* matrix, int N, int n, int offsetx, int offsety, int depth, int MAXDEPTH, int BLOCKSIZE) {
    int tidx  = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy  = blockIdx.y * blockDim.y + threadIdx.y;

    if(tidx <= n && tidy <= n) {
        matrix[(tidy+offsety)*N+(tidx+offsetx)] = cost_function();
    }

    if((tidx + tidy) == 0){
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        int halfn = n>>1;
        int gdim = (halfn+BLOCKSIZE-1)/BLOCKSIZE;
        dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 dimGrid(gdim, gdim, 1);
        if(  ((depth >= MAXDEPTH) || (n <= NCORTE)) ){
            dimGrid.x = dimGrid.x << 1;
            dimGrid.y = dimGrid.y << 1;

            boundingBox_Method <<< dimGrid, dimBlock, 0, s1 >>> (matrix, N, n, offsetx, offsety - n);
            boundingBox_Method <<< dimGrid, dimBlock, 0, s2 >>> (matrix, N, n, offsetx + n, offsety);
        } else {
            //Invocación hijo izquierdo
            dynamicParallelism_Method <<< dimGrid, dimBlock, 0, s1>>> (matrix, N, halfn, offsetx, offsety - halfn, depth+1, MAXDEPTH, BLOCKSIZE);

            //Invocación hijo derecho
            dynamicParallelism_Method <<< dimGrid, dimBlock, 0, s2>>> (matrix, N, halfn, offsetx + n, offsety + halfn, depth+1, MAXDEPTH, BLOCKSIZE);
        }
    }
}


/**
*   Método Raiz cuadrada inversa (rqsqrt) - @Author:CristobalNavarro
*/
__global__ void inverseSquareRoot_Method(float* matrix, int N){  
    unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
  
    float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);

    unsigned int bi = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
    unsigned int bj = bc - (bi*(bi+1) >> 1);

    unsigned int i = bi * blockDim.y + threadIdx.y;
    unsigned int j = bj * blockDim.x + threadIdx.x;

    unsigned int index = (i*N)+j;
    if(i >= j && index < N*N){
        matrix[index] = cost_function();
    }
}

/**
*   Método BoundingBox (BB) - Fuerza Bruta
*/
__global__ void boundingBox_Method(float* matrix, int N, int n, int offsetx, int offsety){
    if( blockIdx.x > blockIdx.y )
        return;

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= j){
        unsigned int index = (i + offsetx)*N + (j + offsety);
        matrix[index] = cost_function();
    }
}

#endif