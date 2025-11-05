all:
	futhark run radix_sort.fut < input.in

bench: 
	futhark bench radix_sort.fut --runs=500 --backend=cuda < input.in

clean: 
	rm -rf *.c radix_sort