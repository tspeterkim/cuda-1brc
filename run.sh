# Create the 1 billion row file as measurements.txt (~14GB)
python data/create_measurements.py 1000000000

# Compile and run the c++ baseline.
g++ -o base -O2 base.cpp
time ./base data/measurements.txt  # ~17 mins

# Compile and run my cuda solution.
nvcc -o fast -O2 fast.cu
# 1 million threads to execute in batches of max size 600000.
# TODO: Increase the max size by using a GPU with more memory.
time ./fast data/measurements.txt 1000000 600000  # ~17s on V100

# Sanity check
# TODO: The cuda floats are off by 0.1 for some cities. Check why.
diff cuda_measurements.out measurements.out
