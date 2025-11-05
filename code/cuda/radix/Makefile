CXX        = nvcc -O3 
#-arch=sm_60

SRC = radix.cu
EXECUTABLE = radix

RADIX_Q ?= 22
RADIX_B ?= 256

default: compile run

.cu.o: $(SRC)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) -DRADIX_Q=$(Q) -DRADIX_B=$(B) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 100

clean:
	rm -f $(EXECUTABLE)

