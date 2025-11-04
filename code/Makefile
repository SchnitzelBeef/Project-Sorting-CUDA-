FUTHARK_PATH := futhark
NUM_ELEMENTS := 100
MAX_VAL := 4294967295
TYPE := u32

.PHONY: futhark

all:
	# Create dataset for testing
	make dataset
	# Run Futhark Radix Sort
	make futhark

dataset:
	# Creating dataset with above values
	futhark dataset --${TYPE}-bounds 0:${MAX_VAL} -g [${NUM_ELEMENTS}]${TYPE} > input.txt

futhark:
	futhark cuda $(FUTHARK_PATH)/radix_sort-${TYPE}.fut -o radix_sort-${TYPE}
	./radix_sort-${TYPE} -t /dev/stderr < input.txt > /dev/null 2> timing.txt
	@printf "Futhark Radix Sort Time: "
	@tr -d '\n' < timing.txt
	@printf " microseconds\n"

cuda:
