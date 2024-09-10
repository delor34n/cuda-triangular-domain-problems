#!/bin/bash

echo "ejecutando: make clean..."

make all > /dev/null 2>&1

echo "compilacion ok..."

for ((N=1024; N<=32768; N*=2)); do
	echo "ejecutando: BLOCKSIZE = 32; N = $N";
  	./tdProblems.out $N 32 10;
	echo -e "\n";
done

echo -e "\n\n";

for ((N=1024; N<=16384; N*=2)); do
	echo "ejecutando: BLOCKSIZE = 16; N = $N";
  	./tdProblems.out $N 16 10;
	echo -e "\n";
done