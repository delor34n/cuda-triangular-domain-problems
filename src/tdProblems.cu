#include <stdio.h>
#include <cuda.h>
#include "constantes.h"
#include "utils.h"
#include "kernels.cuh"

void dynamicParallelism_Warmup(float * matrix_device, float * matrix_host);
void dynamicParallelism_BenchMark(float * matrix_device, float * matrix_host);

void boundingBox_Warmup(float * matrix_device, float * matrix_host);
void boundingBox_BenchMark(float * matrix_device, float * matrix_host);

void inverseSquareRoot_Warmup(float * matrix_device, float * matrix_host);
void inverseSquareRoot_BenchMark(float * matrix_device, float * matrix_host);

void flatRecursive_Warmup(float * matrix_device, float * matrix_host);
void flatRecursive_BenchMark(float * matrix_device, float * matrix_host);

void flatRecursiveBasic_Warmup(float * matrix_device, float * matrix_host);
void flatRecursiveBasic_BenchMark(float * matrix_device, float * matrix_host);

int main(int argc, char** argv) {
    srand(time(NULL));

    //1: DP; 2: BB; 3: IRS; 4: FR; 5: FRB;
    int metodo;
    if (argc != 4) {
        fprintf(stderr, "run as ./prog N BLOCKSIZE METODO\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }
    N = atoi(argv[1]);
    BLOCKSIZE = atoi(argv[2]);
    if(BLOCKSIZE < 1 || BLOCKSIZE > 32){
        fprintf(stderr, "BLOCKSIZE debe ser múltiplo de 2.\nNo puede ser menor a 2.\nNo puede ser mayor a 32\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }

    metodo = atoi(argv[3]);

    int vol = N*N;
    MAXDEPTH = log2f(N) * ALPHA;
    infoPrintf("MAXDEPTH = %d; NCORTE = %d; ALPHA = %f\n", MAXDEPTH, NCORTE, ALPHA); fflush(stdout);

    //Creamos espacios de memoria para la matriz
    float *matrix_host, *matrix_device;

    size_matrix = sizeof(float)*(vol);

    matrix_host = (float*)malloc(vol * sizeof(float));
    fillMatrix(matrix_host, 0.0f);
    gpuErrchk(cudaMalloc((void **) &matrix_device, size_matrix));
    gpuErrchk(cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    size_t limitSize, pending_limit;
    cudaDeviceGetLimit(&limitSize, cudaLimitDevRuntimeSyncDepth);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 18);
    cudaDeviceGetLimit(&limitSize, cudaLimitDevRuntimeSyncDepth);
    cudaDeviceSynchronize();

    cudaDeviceGetLimit(&pending_limit, cudaLimitDevRuntimePendingLaunchCount);
    infoPrintf("pending_limit = %i\n", pending_limit);

    switch(metodo){
        case 1:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Dynamic Parallelism Stream...\n"); fflush(stdout);
            dynamicParallelism_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Dynamic Parallelism Stream...\n"); fflush(stdout);

            dynamicParallelism_BenchMark(matrix_device, matrix_host);

            exportResults(timeDynamicParallelism, errorDynamicParallelism);
            break;
        case 2:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Bounding Box...\n"); fflush(stdout);
            boundingBox_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Bounding Box...\n"); fflush(stdout);

            boundingBox_BenchMark(matrix_device, matrix_host);

            exportResults(timeBoundingBox, errorBoundingBox);
            break;
        case 3:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Inverse Squeare Root method...\n"); fflush(stdout);
            inverseSquareRoot_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Inverse Squeare Root method...\n"); fflush(stdout);

            inverseSquareRoot_BenchMark(matrix_device, matrix_host);

            exportResults(timeInverseSquareRoot, errorInverseSquareRoot);
            break;
        case 4:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Flat Recursive method...\n"); fflush(stdout);
            flatRecursive_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Flat Recursive method...\n"); fflush(stdout);

            flatRecursive_BenchMark(matrix_device, matrix_host);

            exportResults(timeFlatRecursive, errorFlatRecursive);
            break;
        case 5:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Flat Recursive basic method...\n"); fflush(stdout);
            flatRecursiveBasic_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Flat Recursive basic method...\n"); fflush(stdout);

            flatRecursiveBasic_BenchMark(matrix_device, matrix_host);

            exportResults(timeFlatRecursiveBasic, errorFlatRecursiveBasic);
            break;
        default:
            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Dynamic Parallelism Stream...\n"); fflush(stdout);
            dynamicParallelism_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Dynamic Parallelism Stream...\n"); fflush(stdout);

            dynamicParallelism_BenchMark(matrix_device, matrix_host);

            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Bounding Box...\n"); fflush(stdout);
            boundingBox_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Bounding Box...\n"); fflush(stdout);

            boundingBox_BenchMark(matrix_device, matrix_host);

            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Inverse Squeare Root method...\n"); fflush(stdout);
            inverseSquareRoot_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Inverse Squeare Root method...\n"); fflush(stdout);

            inverseSquareRoot_BenchMark(matrix_device, matrix_host);

            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Flat Recursive method...\n"); fflush(stdout);
            flatRecursive_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Flat Recursive method...\n"); fflush(stdout);

            flatRecursive_BenchMark(matrix_device, matrix_host);

            infoPrintf("\n"); fflush(stdout);
            infoPrintf("Inicio Warmup Flat Recursive basic method...\n"); fflush(stdout);
            flatRecursiveBasic_Warmup(matrix_device, matrix_host);
            infoPrintf("Fin Warmup Flat Recursive basic method...\n"); fflush(stdout);

            flatRecursiveBasic_BenchMark(matrix_device, matrix_host);

            exportResultsAll();
            break;
    }

    cudaFree(matrix_device);
    free(matrix_host);

    cudaThreadExit();
    cudaDeviceReset();

    return 0;
}

void dynamicParallelism_Warmup(float * matrix_device, float * matrix_host) {
    debugPrintf("Iniciando\n"); fflush(stdout);

    //Valores de los parametros del kernel
    int nr = N/2;
    dim3 dimGrid((nr + BLOCKSIZE - 1)/BLOCKSIZE, (nr + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("nr %d, BLOCKSIZE %d; N %d\n", nr, BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < MEASURES; i++){
        dynamicParallelism_Method <<< dimGrid, dimBlock >>> (matrix_device, N, N/2, 0, N/2, 0, MAXDEPTH, BLOCKSIZE);
        cudaDeviceSynchronize();
    }

    debugPrintf("Fin\n"); fflush(stdout);
}

void dynamicParallelism_BenchMark(float * matrix_device, float * matrix_host) {
    infoPrintf("Iniciando\n"); fflush(stdout);

    timeDynamicParallelism = 0;
    errorDynamicParallelism = 0;

    cudaEvent_t measureStart, measureStop;

    cudaEventCreate(&measureStart);
    cudaEventCreate(&measureStop);

    float *times;
    times = (float*)malloc(CANTIDAD_ITERACIONES * sizeof(float));

    float accum, elapsedTime, varianza, desviacionEstandar;
    accum = 0.0f;
    elapsedTime = 0.0f;
    varianza = 0.0f;
    desviacionEstandar = 0.0f;

    //Valores de los parametros del kernel
    int nr = N/2;
    dim3 dimGrid((nr + BLOCKSIZE - 1)/BLOCKSIZE, (nr + BLOCKSIZE - 1)/BLOCKSIZE, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("nr %d, BLOCKSIZE %d; N %d\n", nr, BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < CANTIDAD_ITERACIONES; i++){
        cudaEventRecord(measureStart, 0);
        for(int j=0; j < MEASURES; j++){
            dynamicParallelism_Method <<< dimGrid, dimBlock >>> (matrix_device, N, N/2, 0, N/2, 0, MAXDEPTH, BLOCKSIZE);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        cudaEventRecord(measureStop, 0);
        cudaEventSynchronize(measureStop);
        cudaEventElapsedTime(&elapsedTime, measureStart, measureStop);
        times[i] = elapsedTime/(float)MEASURES;
        accum += times[i];

        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(matrix_host, matrix_device, size_matrix, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        verbosePrintf("Revisando matriz...\n"); fflush(stdout);
        if(!checkMatrix(matrix_host)){
            printf("error Dynamic Parallelism method\n"); fflush(stdout);
            exit(-1);
        }
        verbosePrintf("Matriz ok...\n"); fflush(stdout);

        verbosePrintf("Limpiando matriz...\n"); fflush(stdout);
        fillMatrix(matrix_host, 0);
        cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        verbosePrintf("Matriz limpia...\n"); fflush(stdout);
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaEventDestroy(measureStart);
    cudaEventDestroy(measureStop);
    timeDynamicParallelism = accum/(float)CANTIDAD_ITERACIONES;
    varianza = calculoVarianza(times);
    desviacionEstandar = sqrtf(varianza);
    errorDynamicParallelism = desviacionEstandar/timeDynamicParallelism;
    infoPrintf("timeDynamicParallelism %f, accum %f, desviacionEstandar %f, varianza %f, errorDynamicParallelism %f\n", timeDynamicParallelism, accum, desviacionEstandar, varianza, errorDynamicParallelism);

    infoPrintf("Fin\n"); fflush(stdout);
}

void boundingBox_Warmup(float * matrix_device, float * matrix_host) {
    debugPrintf("Iniciando\n"); fflush(stdout);

    //Valores de los parametros del kernel
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(N/dimBlock.x + (N%dimBlock.x == 0 ? 0:1), N/dimBlock.y + (N%dimBlock.y == 0 ? 0:1), 1);
    infoPrintf("BLOCKSIZE %d; N %d\n", BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < MEASURES; i++){
        boundingBox_Method <<< dimGrid, dimBlock >>> (matrix_device, N, N/2, 0, 0);
        cudaDeviceSynchronize();
    }

    debugPrintf("Fin\n"); fflush(stdout);
}

void boundingBox_BenchMark(float * matrix_device, float * matrix_host) {
    infoPrintf("Iniciando\n"); fflush(stdout);

    timeBoundingBox = 0;
    errorBoundingBox = 0;

    cudaEvent_t measureStart, measureStop;

    cudaEventCreate(&measureStart);
    cudaEventCreate(&measureStop);

    float *times;
    times = (float*)malloc(CANTIDAD_ITERACIONES * sizeof(float));

    float accum, elapsedTime, varianza, desviacionEstandar;
    accum = 0.0f;
    elapsedTime = 0.0f;
    varianza = 0.0f;
    desviacionEstandar = 0.0f;

    //Valores de los parametros del kernel
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(N/dimBlock.x + (N%dimBlock.x == 0 ? 0:1), N/dimBlock.y + (N%dimBlock.y == 0 ? 0:1), 1);
    infoPrintf("BLOCKSIZE %d; N %d\n", BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < CANTIDAD_ITERACIONES; i++){
        cudaEventRecord(measureStart, 0);
        for(int j=0; j < MEASURES; j++){
            boundingBox_Method <<< dimGrid, dimBlock >>> (matrix_device, N, N/2, 0, 0);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        cudaDeviceSynchronize();

        cudaEventRecord(measureStop, 0);
        cudaEventSynchronize(measureStop);
        cudaEventElapsedTime(&elapsedTime, measureStart, measureStop);
        times[i] = elapsedTime/(float)MEASURES;
        accum += times[i];

        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(matrix_host, matrix_device, size_matrix, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        verbosePrintf("Revisando matriz...\n"); fflush(stdout);
        if(!checkMatrix(matrix_host)){
            printf("error Bounding Box method\n"); fflush(stdout);
            exit(-1);
        }
        verbosePrintf("Matriz ok...\n"); fflush(stdout);

        verbosePrintf("Limpiando matriz...\n"); fflush(stdout);
        fillMatrix(matrix_host, 0);
        cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        verbosePrintf("Matriz limpia...\n"); fflush(stdout);
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaEventDestroy(measureStart);
    cudaEventDestroy(measureStop);
    timeBoundingBox = accum/(float)CANTIDAD_ITERACIONES;
    varianza = calculoVarianza(times);
    desviacionEstandar = sqrtf(varianza);
    errorBoundingBox = desviacionEstandar/timeBoundingBox;
    infoPrintf("timeBoundingBox %f, accum %f, desviacionEstandar %f, varianza %f, errorBoundingBox %f\n", timeBoundingBox, accum, desviacionEstandar, varianza, errorBoundingBox);

    infoPrintf("Fin\n"); fflush(stdout);
}

void inverseSquareRoot_Warmup(float * matrix_device, float * matrix_host) {
    debugPrintf("Iniciando\n"); fflush(stdout);

    //Valores de los parametros del kernel
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    int sn = (N+dimBlock.x-1)/dimBlock.x;
    int sd = sn*(sn+1)/2;
    int s = ceil(sqrt((double)sd)); 
    dim3 dimGrid(s, s, 1);

    infoPrintf("BLOCKSIZE %d; N %d\n", BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < MEASURES; i++){
        inverseSquareRoot_Method <<< dimGrid, dimBlock >>> (matrix_device, N);
        cudaDeviceSynchronize();
    }
    debugPrintf("Fin\n"); fflush(stdout);
}

void inverseSquareRoot_BenchMark(float * matrix_device, float * matrix_host) {
    infoPrintf("Iniciando\n"); fflush(stdout);

    timeInverseSquareRoot = 0;
    errorInverseSquareRoot = 0;

    cudaEvent_t measureStart, measureStop;

    cudaEventCreate(&measureStart);
    cudaEventCreate(&measureStop);

    float *times;
    times = (float*)malloc(CANTIDAD_ITERACIONES * sizeof(float));

    float accum, elapsedTime, varianza, desviacionEstandar;
    accum = 0.0f;
    elapsedTime = 0.0f;
    varianza = 0.0f;
    desviacionEstandar = 0.0f;

    //Valores de los parametros del kernel
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    int sn = (N+dimBlock.x-1)/dimBlock.x;
    int sd = sn*(sn+1)/2;
    int s = ceil(sqrt((double)sd)); 
    dim3 dimGrid(s, s, 1);

    infoPrintf("Limpiando matriz...\n"); fflush(stdout);
    fillMatrix(matrix_host, 0);
    cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    infoPrintf("Matriz limpia...\n"); fflush(stdout);

    infoPrintf("BLOCKSIZE %d; N %d\n", BLOCKSIZE, N); fflush(stdout);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < CANTIDAD_ITERACIONES; i++){
        cudaEventRecord(measureStart, 0);
        for(int j=0; j < MEASURES; j++){
            inverseSquareRoot_Method <<< dimGrid, dimBlock >>> (matrix_device, N);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        cudaEventRecord(measureStop, 0);
        cudaEventSynchronize(measureStop);
        cudaEventElapsedTime(&elapsedTime, measureStart, measureStop);
        times[i] = elapsedTime/(float)MEASURES;
        accum += times[i];

        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(matrix_host, matrix_device, size_matrix, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        verbosePrintf("Revisando matriz...\n"); fflush(stdout);
        if(!checkMatrix(matrix_host)){
            printf("error Inverse Squeare Root method\n"); fflush(stdout);
            exit(-1);
        }
        verbosePrintf("Matriz ok...\n"); fflush(stdout);

        verbosePrintf("Limpiando matriz...\n"); fflush(stdout);
        fillMatrix(matrix_host, 0);
        cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        verbosePrintf("Matriz limpia...\n"); fflush(stdout);
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaEventDestroy(measureStart);
    cudaEventDestroy(measureStop);
    timeInverseSquareRoot = accum/(float)CANTIDAD_ITERACIONES;
    varianza = calculoVarianza(times);
    desviacionEstandar = sqrtf(varianza);
    errorInverseSquareRoot = desviacionEstandar/timeInverseSquareRoot;
    infoPrintf("timeInverseSquareRoot %f, accum %f, desviacionEstandar %f, varianza %f, errorInverseSquareRoot %f\n", timeInverseSquareRoot, accum, desviacionEstandar, varianza, errorInverseSquareRoot);

    infoPrintf("Fin\n"); fflush(stdout);
}

void flatRecursive_Warmup(float * matrix_device, float * matrix_host) {
    debugPrintf("Iniciando\n"); fflush(stdout);

    //Valores de los parametros del kernel
    int Nb = ceil((N+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 dimGrid(Nb/2,Nb, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < MEASURES; i++){
        flatRecursive <<< dimGrid, dimBlock >>> (matrix_device, N, N/2);
        cudaDeviceSynchronize();
    }
    debugPrintf("Fin\n"); fflush(stdout);
}

void flatRecursive_BenchMark(float * matrix_device, float * matrix_host) {
    infoPrintf("Iniciando\n"); fflush(stdout);

    timeFlatRecursive = 0;
    errorFlatRecursive = 0;

    cudaEvent_t measureStart, measureStop;

    cudaEventCreate(&measureStart);
    cudaEventCreate(&measureStop);

    float *times;
    times = (float*)malloc(CANTIDAD_ITERACIONES * sizeof(float));

    float accum, elapsedTime, varianza, desviacionEstandar;
    accum = 0.0f;
    elapsedTime = 0.0f;
    varianza = 0.0f;
    desviacionEstandar = 0.0f;

    //Valores de los parametros del kernel
    int Nb = ceil((N+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 dimGrid(Nb/2,Nb, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < CANTIDAD_ITERACIONES; i++){
        cudaEventRecord(measureStart, 0);
        for(int j=0; j < MEASURES; j++){
            flatRecursive <<< dimGrid, dimBlock >>> (matrix_device, N, N/2);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        cudaEventRecord(measureStop, 0);
        cudaEventSynchronize(measureStop);
        cudaEventElapsedTime(&elapsedTime, measureStart, measureStop);
        times[i] = elapsedTime/(float)MEASURES;
        accum += times[i];

        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(matrix_host, matrix_device, size_matrix, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        verbosePrintf("Revisando matriz...\n"); fflush(stdout);
        if(!checkMatrix(matrix_host)){
            printf("error Flat Recursive method\n"); fflush(stdout);
            exit(-1);
        }
        verbosePrintf("Matriz ok...\n"); fflush(stdout);

        verbosePrintf("Limpiando matriz...\n"); fflush(stdout);
        fillMatrix(matrix_host, 0);
        cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        verbosePrintf("Matriz limpia...\n"); fflush(stdout);
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaEventDestroy(measureStart);
    cudaEventDestroy(measureStop);
    timeFlatRecursive = accum/(float)CANTIDAD_ITERACIONES;
    varianza = calculoVarianza(times);
    desviacionEstandar = sqrtf(varianza);
    errorFlatRecursive = desviacionEstandar/timeFlatRecursive;
    infoPrintf("timeFlatRecursive %f, accum %f, desviacionEstandar %f, varianza %f, errorFlatRecursive %f\n", timeFlatRecursive, accum, desviacionEstandar, varianza, errorFlatRecursive);

    infoPrintf("Fin\n"); fflush(stdout);
}

void flatRecursiveBasic_Warmup(float * matrix_device, float * matrix_host) {
    debugPrintf("Iniciando\n"); fflush(stdout);

    //Valores de los parametros del kernel
    int Nb = ceil((N+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 dimGrid(Nb/2,Nb, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < MEASURES; i++){
        flatRecursiveBasic <<< dimGrid, dimBlock >>> (matrix_device, N, N/2);
        cudaDeviceSynchronize();
    }
    debugPrintf("Fin\n"); fflush(stdout);
}

void flatRecursiveBasic_BenchMark(float * matrix_device, float * matrix_host) {
    infoPrintf("Iniciando\n"); fflush(stdout);

    timeFlatRecursiveBasic = 0;
    errorFlatRecursiveBasic = 0;

    cudaEvent_t measureStart, measureStop;

    cudaEventCreate(&measureStart);
    cudaEventCreate(&measureStop);

    float *times;
    times = (float*)malloc(CANTIDAD_ITERACIONES * sizeof(float));

    float accum, elapsedTime, varianza, desviacionEstandar;
    accum = 0.0f;
    elapsedTime = 0.0f;
    varianza = 0.0f;
    desviacionEstandar = 0.0f;

    //Valores de los parametros del kernel
    int Nb = ceil((N+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 dimGrid(Nb/2,Nb, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    infoPrintf("dimGrid.x %d, dimGrid.y %d, dimGrid.z %d\n", dimGrid.x, dimGrid.y, dimGrid.z); fflush(stdout);
    infoPrintf("dimBlock.x %d, dimBlock.y %d, dimBlock.z %d\n\n", dimBlock.x, dimBlock.y, dimBlock.z); fflush(stdout);

    for(int i=0; i < CANTIDAD_ITERACIONES; i++){
        cudaEventRecord(measureStart, 0);
        for(int j=0; j < MEASURES; j++){
            flatRecursiveBasic <<< dimGrid, dimBlock >>> (matrix_device, N, N/2);
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        cudaEventRecord(measureStop, 0);
        cudaEventSynchronize(measureStop);
        cudaEventElapsedTime(&elapsedTime, measureStart, measureStop);
        times[i] = elapsedTime/(float)MEASURES;
        accum += times[i];

        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(matrix_host, matrix_device, size_matrix, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        verbosePrintf("Revisando matriz...\n"); fflush(stdout);
        if(!checkMatrix(matrix_host)){
            printf("error Flat Recursive Basic method\n"); fflush(stdout);
            exit(-1);
        }
        verbosePrintf("Matriz ok...\n"); fflush(stdout);

        verbosePrintf("Limpiando matriz...\n"); fflush(stdout);
        fillMatrix(matrix_host, 0);
        cudaMemcpy(matrix_device, matrix_host, size_matrix, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        verbosePrintf("Matriz limpia...\n"); fflush(stdout);
    }

    gpuErrchk(cudaPeekAtLastError());

    cudaEventDestroy(measureStart);
    cudaEventDestroy(measureStop);
    timeFlatRecursiveBasic = accum/(float)CANTIDAD_ITERACIONES;
    varianza = calculoVarianza(times);
    desviacionEstandar = sqrtf(varianza);
    errorFlatRecursiveBasic = desviacionEstandar/timeFlatRecursiveBasic;
    infoPrintf("timeFlatRecursiveBasic %f, accum %f, desviacionEstandar %f, varianza %f, errorFlatRecursiveBasic %f\n", timeFlatRecursiveBasic, accum, desviacionEstandar, varianza, errorFlatRecursiveBasic);

    infoPrintf("Fin\n"); fflush(stdout);
}