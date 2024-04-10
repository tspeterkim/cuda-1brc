#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <cstring>

#define N_CITY 44'691
#define N_ROW 100'000'000
#define MAX_CITY_BYTE 100

struct Part {
    long long offset;
    long long length;
};

struct Stat {
    char city[MAX_CITY_BYTE];
    float min; float max; float sum;
    int count;
    int index;
    Stat(const char* dcity = "", float dmin = INFINITY, float dmax = -INFINITY, float dsum = 0.0f, int dcount = 0)
        : min(dmin), max(dmax), sum(dsum), count(dcount) {
            strncpy(city, dcity, sizeof(dcity));
            city[sizeof(city)-1] = '\0';
            index = -1;
        }
};

bool compareStat(const Stat& a, const Stat& b) {
    return std::strcmp(a.city, b.city) < 0;
}

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

__device__ float parse_float(char* str) {
    float result = 0.0f;
    int sign = 1; int decimal = 0; int digits = 0;
    // Handling sign
    if (*str == '-') {
        sign = -1;
        ++str;
    } else if (*str == '+') {
        ++str;
    }
    // Parsing integer part
    while (*str >= '0' && *str <= '9') {
        result = result * 10.0f + (*str - '0');
        ++str;
        ++digits;
    }
    // Parsing decimal part
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
    // Adjusting the result based on the decimal position
    while (decimal > 0) {
        result /= 10.0f;
        --decimal;
    }
    return result;
}

__device__ unsigned int hash(char* str) {
    unsigned int val = 5381;
    int c;
    while ((c = *str++)) {
        val = ((val << 5) + val) + c;
    }
    return val;
    unsigned int hash = 0;
    while (*str) {
        hash += *str++;
        hash += hash << 10;
        hash ^= hash >> 6;
    }
    hash += hash << 3;
    hash ^= hash >> 11;
    hash += hash << 15;
    return hash;
}


__device__ int get_index(char* all_city, char* str) {
    #pragma unroll 1
    for (int i = 0; i < N_CITY; i++) {
        #pragma unroll 1
        for (int j = 0; j < MAX_CITY_BYTE; j++) {
            if (all_city[i * MAX_CITY_BYTE + j] == str[j]) {
                if (str[j] == '\0')
                    return i;
            } else {
                break;
            }
        }
    }
    return 0;
}

__global__ void process_buffer(char* buffer, Part* parts, Stat* stats, long long buffer_offset, int part_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x * blockDim.x + tx;
    int index = 0;
    int read_city = 0;

    char city[MAX_CITY_BYTE];
    char floatstr[100];

    if (bx >= part_size)
        return;

    for (int i = 0; i < parts[bx].length; i++) {
        char c = buffer[parts[bx].offset-buffer_offset+i];
        if (read_city == 0) {
            if (c == ';') {
                city[index] = '\0';
                //printf("%s \n", city);
                index = 0;
                read_city = 1;
            } else {
                city[index] = c;
                index++;
            }
        } else {
            if (c == '\n') {
                floatstr[index] = '\0';
                //printf("%s \n", floatstr);
                float temp = parse_float(floatstr);

                // do interesting stuff TODO: deal with collisions

                unsigned int stat_index = hash(city) % N_ROW; // hash city

                //int stat_index = get_index(all_city, city); // cheating

                //printf("%s, %.2f, hash=%d \n", city, temp, stat_index);
                /*
                if stats[stat_index]'s city is first insert:
                    proceed as usual
                if stats[stat_index]'s city is not equal to current city name
                    collision detected
                else
                    proceed as usual
                */
                atomicMin(&stats[stat_index].min, temp);
                atomicMax(&stats[stat_index].max, temp);
                atomicAdd(&stats[stat_index].sum, temp);
                atomicAdd(&stats[stat_index].count, 1);

                for (int j = 0; j < MAX_CITY_BYTE; j++) {
                    stats[stat_index].city[j] = city[j];
                    if (city[j] == '\0')
                        break;
                }

                stats[stat_index].index = stat_index;

                // reset
                index = 0;
                read_city = 0;
                floatstr[0] = '\0';
                city[0] = '\0';
            } else {
                floatstr[index] = c;
                index++;
            }
        }
    }
}


std::vector<Part> split_file(std::string input_path, int num_parts) {
    std::ifstream file(input_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    long long split_size = size / num_parts;

    std::cout << size << " " << split_size << std::endl;

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
        for (int i = n-1; i>=0; --i) {
            if (buf[i] == '\n') {
                newline = i;
                break;
            }
        }
        if (newline < 0) {
            throw std::runtime_error("newline not found!");
        }
        int remaining = n - newline - 1;
        long long next_offset = seek_offset + n - remaining;
        parts.push_back({offset, next_offset-offset});
        offset = next_offset;
    }
    /*
    for (auto& part: parts) {
        file.clear();
        file.seekg(part.offset, std::ios::beg);

        char* buffer = new char[part.length];
        file.read(buffer, part.length);
        std::cout.write(buffer, part.length);
        std::cout << std::endl;
    }
    */
    file.close();
    return parts;
}


std::vector<std::string> get_all_city() {
    std::vector<std::string> cityNames;
    std::ifstream file("data/weather_stations.csv");

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') {
            continue; // Skip lines that start with '#'
        }
        std::istringstream iss(line);
        std::string cityName;
        std::getline(iss, cityName, ';');
        cityNames.push_back(cityName);
    }
    file.close();
    return cityNames;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file path>" << std::endl;
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();

    std::string input_path = argv[1];

    int num_parts = 1024 * 1000;
    std::vector<Part> parts = split_file(input_path, num_parts);
    num_parts = parts.size();

    std::cout << parts.size() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken finding parts: " << elapsed.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    Stat* stats = new Stat[N_ROW];
    for (int i = 0; i < N_ROW; i++){
        stats[i] = Stat();
    }
    Stat* d_stats;
    cudaMalloc(&d_stats, N_ROW * sizeof(Stat));
    cudaMemcpy(d_stats, stats, N_ROW * sizeof(Stat), cudaMemcpyHostToDevice);

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken memcpy stats: " << elapsed.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    char* d_buffer; // holds the entire read char buffer, raw text
    cudaMalloc((void**)&d_buffer, 2'000'000'000 * sizeof(char));

    Part* d_parts;
    cudaMalloc(&d_parts, parts.size() * sizeof(Part));

    std::ifstream file(input_path, std::ios::binary);

    /*
    char all_city[100 * N_CITY] = {'\0'};
    int nc = 0;
    for (auto& city : get_all_city()) {
        strcpy(all_city + (nc * 100), city.c_str());
        nc++;
    }
    char* d_city;
    cudaMalloc(&d_city, 100 * N_CITY * sizeof(char));
    cudaMemcpy(d_city, all_city, 100 * N_CITY * sizeof(char), cudaMemcpyHostToDevice);
    */

    int batch_size = 1024 * 100;

    //std::ofstream tmp("temp.out");
    for (int b = 0; b < num_parts; b += batch_size) {
        long long batch_file_size = 0;
        for (int bi = b; bi < std::min(b+batch_size, num_parts); bi++) {
            batch_file_size += parts[bi].length;
        }

        file.seekg(parts[b].offset, std::ios::beg);

        char* buffer = new char[batch_file_size];
        file.read(buffer, batch_file_size);

        //tmp.write(buffer, batch_file_size);

        cudaMemcpy(d_buffer, buffer, batch_file_size * sizeof(char), cudaMemcpyHostToDevice);

        int part_size = batch_size;
        if (b+batch_size > num_parts) {
            part_size = num_parts-b;
        }
        cudaMemcpy(d_parts, parts.data()+b, part_size * sizeof(Part), cudaMemcpyHostToDevice);

        int threads_per_block = 1024;
        int grid_blocks = std::ceil((float) part_size / threads_per_block);

        //std::cout << part_size << " " << grid_blocks << " " << threads_per_block << std::endl;

        process_buffer<<<grid_blocks, threads_per_block>>>(d_buffer, d_parts, d_stats, parts[b].offset, part_size);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        }

        //cudaDeviceSynchronize();
        //std::cout << std::endl;
        delete[] buffer;
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken cuda kernel: " << elapsed.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(stats, d_stats, N_ROW * sizeof(Stat), cudaMemcpyDeviceToHost);

    // TODO: can we push this to the gpu?
    std::sort(stats, stats + N_ROW, compareStat);

    std::ofstream measurements("cuda_measurements.out");
    for (int i = 0; i < N_ROW; i++) {
        if (stats[i].count != 0) {
            float mean = stats[i].sum / stats[i].count;
            measurements << stats[i].city << "=" << stats[i].min << "/";
            measurements << std::fixed << std::setprecision(1) << mean << "/";
            measurements << stats[i].max << std::endl;
            //measurements << stats[i].max; //  << std::endl;
            //measurements << " " << stats[i].index << " count=" << stats[i].count << std::endl;
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken sort and print: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
