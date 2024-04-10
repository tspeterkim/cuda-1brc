#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <cstring>

#define MAX_CITY_BYTE 100  // City names are at most 100 bytes.
#define MAX_THREADS_PER_BLOCK 1024

// File split metadata of the end offset, and length, all in bytes.
struct Part {
    long long offset; long long length;
};

// Hash table entry of a given city, keeping track of the temperature statistics.
struct Stat {
    char city[MAX_CITY_BYTE];
    float min = INFINITY; float max = -INFINITY; float sum = 0;
    int count = 0;
    Stat() {}
    Stat(const std::string& init_city) {
        strncpy(city, init_city.c_str(), init_city.size());
        city[init_city.size()] = '\0';
    }
};

// CUDA's atomicMin/Max only work with ints
__device__ static float atomicMin(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// ChatGPT's working solution.
// Probably could be made more accurate using doubles like the actual strtod.c
__device__ float cuda_atof(char* str) {
    float result = 0.0f;
    int sign = 1; int decimal = 0; int digits = 0;

    if (*str == '-') {
        sign = -1;
        ++str;
    }

    while (*str >= '0' && *str <= '9') {
        result = result * 10.0f + (*str - '0');
        ++str;
        ++digits;
    }

    if (*str == '.') {
        ++str;
        while (*str >= '0' && *str <= '9') {
            result = result * 10.0f + (*str - '0');
            ++str;
            ++digits;
            ++decimal;
        }
    }
    result *= sign;

    while (decimal > 0) {
        result /= 10.0f;
        --decimal;
    }
    return result;
}

// Identical to glibc's strcmp.c
__device__ int cuda_strcmp(const char* p1, const char* p2) {
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;
    do {
        c1 = (unsigned char) *s1++;
        c2 = (unsigned char) *s2++;
        if (c1 == '\0')
            return c1 - c2;
    } while (c1 == c2);
    return c1 - c2;
}

// Returns the pre-defined index of a city using binary search.
__device__ int get_index(char* cities, char* city_target, int n_city) {
    int left = 0;
    int right = n_city - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        const char* city_query = cities + mid * MAX_CITY_BYTE;

        int cmp = cuda_strcmp(city_query, city_target);
        if (cmp == 0)
            return mid;
        else if (cmp < 0)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

__global__ void process_buffer(char* buffer, Part* parts, Stat* stats, char* cities, int n_city, long long buffer_offset, int part_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x * blockDim.x + tx;
    int index = 0;
    bool parsing_city = true;

    char city[MAX_CITY_BYTE];
    char floatstr[5];  // longest temperature float str is -99.9 i.e. 5 bytes

    if (bx >= part_size)
        return;

    // An ugly way to do string processing in CUDA.
    // I could probably use more helper functions here.
    for (int i = 0; i < parts[bx].length; i++) {
        char c = buffer[parts[bx].offset-buffer_offset + i];
        if (parsing_city) {  // City characters
            if (c == ';') {
                city[index] = '\0';
                index = 0;
                parsing_city = false;
            } else {
                city[index] = c;
                index++;
            }
        } else {  // Float characters
            if (c == '\n') {
                floatstr[index] = '\0';
                float temp = cuda_atof(floatstr);

                int stat_index = get_index(cities, city, n_city);

                // The heart of the CUDA kernel.
                // Update (atomically) the temperature statistics.
                // Identical in spirit to the simple C++ version.
                atomicMin(&stats[stat_index].min, temp);
                atomicMax(&stats[stat_index].max, temp);
                atomicAdd(&stats[stat_index].sum, temp);
                atomicAdd(&stats[stat_index].count, 1);

                // reset for next line read
                parsing_city = true;
                index = 0;
                floatstr[0] = '\0'; city[0] = '\0';
            } else {
                floatstr[index] = c;
                index++;
            }
        }
    }
}

// Adapted from https://github.com/benhoyt/go-1brc/blob/master/r8.go#L124
std::vector<Part> split_file(std::string input_path, int num_parts) {
    std::ifstream file(input_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Using long long is necessary to avoid overflow of file size.
    // e.g. 15B for a 1B row file
    long long split_size = size / num_parts;

    std::cout << "Total file size: " << size << ", split size: " << split_size << std::endl;

    long long offset = 0;
    std::vector<Part> parts;
    while (offset < size) {
        long long seek_offset = std::max(offset + split_size - MAX_CITY_BYTE, 0LL);
        if (seek_offset > size) {
            parts.back().length += size-offset;
            break;
        }
        file.seekg(seek_offset, std::ios::beg);
        char buf[MAX_CITY_BYTE];
        file.read(buf, MAX_CITY_BYTE);

        std::streamsize n = file.gcount();
        std::streamsize newline = -1;
        for (int i = n - 1; i >= 0; --i) {
            if (buf[i] == '\n') {
                newline = i;
                break;
            }
        }
        int remaining = n - newline - 1;
        long long next_offset = seek_offset + n - remaining;
        parts.push_back({offset, next_offset-offset});
        offset = next_offset;
    }
    file.close();
    return parts;
}

std::set<std::string> get_cities() {
    std::ifstream weather_file("data/weather_stations.csv");
    std::string line;
    std::set<std::string> all_cities;

    while (getline(weather_file, line)) {
        std::istringstream iss(line);
        if (line[0] == '#')
            continue;
        std::string station;
        std::getline(iss, station, ';');
        all_cities.insert(station);
    }
    weather_file.close();
    return all_cities;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <file path> <num parts> <batch size>" << std::endl;
        return 1;
    }

    // Bending the rules of the challenge here.
    // I'm assuming a file like data/weather_stations.csv is given.
    // This file lists all possible cities that could appear in the input file.
    std::set<std::string> all_cities = get_cities();

    int n_city = all_cities.size();
    Stat* stats = new Stat[n_city];
    int index = 0;
    char cities[MAX_CITY_BYTE * n_city] = {'\0'};

    for (const auto& city : all_cities) {
        stats[index] = Stat(city);
        strcpy(cities + (index * MAX_CITY_BYTE), city.c_str());
        index++;
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string input_path = argv[1];
    int num_parts = atoi(argv[2]); int batch_size = atoi(argv[3]);

    std::vector<Part> parts = split_file(input_path, num_parts);
    num_parts = parts.size();

    std::cout << "Required GPU RAM Size (GB): " <<  parts[0].length * batch_size / 1'000'000'000.0 << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken finding parts: " << elapsed.count() << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    Stat* d_stats;
    cudaMalloc(&d_stats, n_city * sizeof(Stat));
    cudaMemcpy(d_stats, stats, n_city * sizeof(Stat), cudaMemcpyHostToDevice);

    char* d_buffer;  // Hold a subset of the raw text char buffer.
    cudaMalloc((void**) &d_buffer, 10'000'000'000 * sizeof(char));

    Part* d_parts;
    cudaMalloc(&d_parts, parts.size() * sizeof(Part));

    char* d_cities;
    cudaMalloc(&d_cities, MAX_CITY_BYTE * n_city * sizeof(char));
    cudaMemcpy(d_cities, cities, MAX_CITY_BYTE * n_city * sizeof(char), cudaMemcpyHostToDevice);

    // Launch CUDA kernels that processes different splits of the file.
    // Will do it in sequential batches, if GPU RAM is limited.
    std::ifstream file(input_path, std::ios::binary);
    for (int b = 0; b < num_parts; b += batch_size) {
        long long batch_file_size = 0;
        for (int bi = b; bi < std::min(b + batch_size, num_parts); bi++)
            batch_file_size += parts[bi].length;

        file.seekg(parts[b].offset, std::ios::beg);

        char* buffer = new char[batch_file_size];
        file.read(buffer, batch_file_size);

        cudaMemcpy(d_buffer, buffer, batch_file_size * sizeof(char), cudaMemcpyHostToDevice);

        int part_size = batch_size;
        if (b + batch_size > num_parts)
            part_size = num_parts - b;
        cudaMemcpy(d_parts, parts.data() + b, part_size * sizeof(Part), cudaMemcpyHostToDevice);

        int grid_blocks = std::ceil((float) part_size / MAX_THREADS_PER_BLOCK);

        process_buffer<<<grid_blocks, MAX_THREADS_PER_BLOCK>>>(d_buffer, d_parts, d_stats, d_cities, n_city, parts[b].offset, part_size);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;

        delete[] buffer;
    }

    cudaDeviceSynchronize();  // for accurate profiling (cuda calls are async)
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken in cuda kernel: " << elapsed.count() << " seconds" << std::endl;

    // Write out the results.
    cudaMemcpy(stats, d_stats, n_city * sizeof(Stat), cudaMemcpyDeviceToHost);
    std::ofstream measurements("measurements.out");
    for (int i = 0; i < n_city; i++) {
        if (stats[i].count != 0) {
            float mean = stats[i].sum / stats[i].count;
            measurements << stats[i].city << "=" << stats[i].min << "/";
            measurements << std::fixed << std::setprecision(1) << mean << "/";
            measurements << stats[i].max << std::endl;
        }
    }

    return 0;
}
