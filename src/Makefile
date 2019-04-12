CC = nvcc
CCFLAG = -lcublas
OPTION = -O3 --ptxas-options=-v
OUT = main
INPUT = spamatrixmul.cu
main: $(INPUT)
	$(CC) $(OPTION) -o $(OUT) $(INPUT) $(CCFLAG)
clean:
	rm -f $(OUT)