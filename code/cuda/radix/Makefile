CXX        = nvcc -O3 
#-arch=sm_60

SRC = radix.cu
EXECUTABLE = radix

RADIX_Q ?= 22
RADIX_B ?= 256
N ?= 100000

default: compile run

.cu.o: $(SRC)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) -DRADIX_Q=$(RADIX_Q) -DRADIX_B=$(RADIX_B) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) $(N) 1

clean:
	rm -f $(EXECUTABLE)

