nvcc=/usr/local/cuda/bin/nvcc
PARAMS=-arch=sm_35 -rdc=true -lineinfo -lcudadevrt -use_fast_math
INPUT_CODE=tdProblems.cu
OUTPUT_CODE=tdProblems.out

all:
	${nvcc} ${PARAMS} ${INPUT_CODE} -DVERBOSE=true -DDEBUG=true -DINFO=true -o ${OUTPUT_CODE}

clean:
	${nvcc} ${PARAMS} ${INPUT_CODE} -DVERBOSE=false -DDEBUG=false -DINFO=false -o ${OUTPUT_CODE}

fast:
	${nvcc} ${PARAMS} -prec-sqrt=false ${INPUT_CODE} -DVERBOSE=false -DDEBUG=false -DINFO=false -o fast_${OUTPUT_CODE}

ptx:
	${nvcc} ${PARAMS} -ptx -DVERBOSE=false -DDEBUG=false -DINFO=false ${INPUT_CODE}

verbose:
	${nvcc} ${PARAMS} -DVERBOSE=true -DDEBUG=false -DINFO=false ${INPUT_CODE} -o ${OUTPUT_CODE}

debug:
	${nvcc} ${PARAMS} -DVERBOSE=false -DDEBUG=true -DINFO=false ${INPUT_CODE} -o ${OUTPUT_CODE}

info:
	${nvcc} ${PARAMS} -DVERBOSE=false -DDEBUG=false -DINFO=true ${INPUT_CODE} -o ${OUTPUT_CODE}
