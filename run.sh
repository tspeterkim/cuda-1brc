python data/create_measurements.py 1000000000
nvcc -o fast fast.cu -O2
./fast data/measurements.txt 1000000 600000