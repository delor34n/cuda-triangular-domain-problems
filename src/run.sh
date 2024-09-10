#!/bin/bash

echo "ejecutando: make clean..."

make clean > /dev/null 2>&1

echo "compilacion ok..."

ITERACION_MAX=100;
for ((ITERACION=0; ITERACION<=ITERACION_MAX; ITERACION+=1)); do
	for ((BLOCKSIZE=8; BLOCKSIZE<=32; BLOCKSIZE*=2)); do
		for ((N=128; N<=16384; N*=2)); do
			echo "ejecutando[$ITERACION]: N: $N, BLOCKSIZE: $BLOCKSIZE";
		  	./tdProblems.out $N $BLOCKSIZE 10;
		done
	done
done
