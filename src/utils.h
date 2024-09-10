unsigned int BLOCKSIZE;
unsigned int N;
unsigned int MAXDEPTH;
size_t size_matrix;

float timeDynamicParallelism, errorDynamicParallelism;
float timeBoundingBox, errorBoundingBox;
float timeInverseSquareRoot, errorInverseSquareRoot;
float timeFlatRecursive, errorFlatRecursive;
float timeFlatRecursiveBasic, errorFlatRecursiveBasic;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void fillMatrix(float *matrix, float fillValue){
    for(int i=0; i < (N); i++){
        matrix[i] = fillValue;
    }
}

void printMatrix(float * matrix){
    for(int i=0; i<N; i++) {
        for(int j=(N*i); j<(N*(i+1)); j++) {
            printf("%.0f ", matrix[j]); fflush(stdout);
        }
        printf("\n"); fflush(stdout);
    }
}

int check_sin_diagonal(float* m, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; ++j){
			if( i <= j && m[i*n + j] != 0.0f ){
				fprintf(stderr, "error m[%i, %i] = %f\n", i, j, m[i*n + j]);
				return 0;
			}
			else if(i > j && m[i*n+j] != 1.0f){
				fprintf(stderr, "error m[%i, %i] = %f\n", i, j, m[i*n + j]);
				return 0;
			}
        }
    }
    return 1;
}

int checkMatrix(float* matrix) {
    int* elementosPorFila  = (int*)malloc(N * sizeof(int*));
    for(int i=0; i<N; i++) {
        elementosPorFila[i] = 0;
        for(int j=(N*i); j<(N*(i+1)); j++) {
            if(elementosPorFila[i] <= i && matrix[j] != 1.0f){
                return 0;
            }

            if(elementosPorFila[i] > i && matrix[j] != 0.0f){
                return 0;
            }

            elementosPorFila[i] += 1;
        }
    }

    return 1;
}

void exportResultsAll(){
    FILE *file;
    file = fopen("perf.dat", "a");
    //#N  BLOCKSIZE  DP  BB  RRSQRT  FLAT  FLAT_BASIC
    //fprintf(file, "%i %f %f %f %f %f\n", N, timeDynamicParallelism, timeBoundingBox, timeInverseSquareRoot, timeFlatRecursive, timeFlatRecursiveBasic);
    //fprintf(file, "%i %f %f %f %f %f %f %f %f %f %f\n", BLOCKSIZE, timeDynamicParallelism, errorDynamicParallelism, timeBoundingBox, errorBoundingBox, timeInverseSquareRoot, errorInverseSquareRoot, timeFlatRecursive, errorFlatRecursive, timeFlatRecursiveBasic, errorFlatRecursiveBasic);
    fprintf(file, "%i %i %f %f %f %f %f %f %f %f %f %f\n", N, BLOCKSIZE, timeDynamicParallelism, errorDynamicParallelism, timeBoundingBox, errorBoundingBox, timeInverseSquareRoot, errorInverseSquareRoot, timeFlatRecursive, errorFlatRecursive, timeFlatRecursiveBasic, errorFlatRecursiveBasic);
    //fprintf(file, "%i %f %f %f %f %f\n", N, timeDynamicParallelism, timeBoundingBox, timeInverseSquareRoot, timeFlatRecursive, timeFlatRecursiveBasic);
    //fprintf(file, "%i %i %f %f %f %f %f\n", N, BLOCKSIZE, timeDynamicParallelism, timeBoundingBox, timeInverseSquareRoot, timeFlatRecursive, timeFlatRecursiveBasic);
    fclose(file);
}

void exportResults(float time, float error){
    FILE *file;
    file = fopen("perf.dat", "a");
    //#N  BLOCKSIZE  DP  BB  RRSQRT  FLAT  FLAT_BASIC
    fprintf(file, "%i %f %f\n", BLOCKSIZE, time, error);
    fclose(file);
}

float calculoVarianza(float *samples){
    float M, S, x, oldM;
    M = 0;
    S = 0;

    for(int k=1; k<CANTIDAD_ITERACIONES; k++){
        x = samples[k];
        oldM = M;
        M = M + (x - M)/k;
        S = S + (x - M) * (x - oldM);
    }
    return S/(CANTIDAD_ITERACIONES-1);
}
