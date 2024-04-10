#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <cmath>

using namespace std;

struct Stat {
    float min, max, sum;
    int count;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <file path>" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    string line;
    map<string, Stat> stationStats;

    while (getline(file, line)) {
        istringstream iss(line);
        string station;
        float temp;
        getline(iss, station, ';');
        iss >> temp;

        auto it = stationStats.find(station);
        if (it == stationStats.end()) {
            stationStats[station] = {temp, temp, temp, 1};
        } else {
            Stat& s = it->second;
            s.min = min(s.min, temp);
            s.max = max(s.max, temp);
            s.sum += temp;
            s.count++;
        }
    }

    ofstream measurements("measurements.out");
    for (auto& pair : stationStats) {
        const Stat& s = pair.second;
        float mean = s.sum / s.count;
        measurements << pair.first << "=" << s.min << "/";
        measurements << fixed << setprecision(1) << mean << "/";
        measurements << s.max << endl;
    }
    return 0;
}
