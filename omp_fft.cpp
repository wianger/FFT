#include <omp.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

using Complex = complex<double>;
using Clock = chrono::high_resolution_clock;

const int m = 1638400;  // DO NOT CHANGE!!
const int K = 100000;   // DO NOT CHANGE!!

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int num,
                      const double disturb0);
bool verifyResults(const std::string& resultPath, const std::string& checkPath, double tolerance = 1e-5);

int main(int argc, char* argv[]) {
    vector<Complex> dat(m);
    vector<Complex> pri(m);
    vector<double> ctf(m);
    vector<double> sigRcp(m);
    vector<double> disturb(K);

    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    ifstream fin;
    fin.open("./data/input.dat");
    if (!fin.is_open()) {
        cout << "Error opening file input.dat" << endl;
        return EXIT_FAILURE;
    }

    int i = 0;
    while (!fin.eof()) {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        if (!fin) {
            break;
        }
        dat[i] = Complex(dat0, dat1);
        pri[i] = Complex(pri0, pri1);
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        ++i;
        if (i == m) {
            break;
        }
    }
    fin.close();

    fin.open("./data/K.dat");
    if (!fin.is_open()) {
        cout << "Error opening file K.dat" << endl;
        return EXIT_FAILURE;
    }
    i = 0;
    while (!fin.eof()) {
        fin >> disturb[i];
        if (!fin) {
            break;
        }
        ++i;
        if (i == K) {
            break;
        }
    }
    fin.close();

    auto startTime = Clock::now();

    ofstream fout;
    fout.open("./data/result.dat");
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        return EXIT_FAILURE;
    }

    for (unsigned int t = 0; t < K; ++t) {
        double result = logDataVSPrior(dat.data(), pri.data(), ctf.data(), sigRcp.data(), m, disturb[t]);
        fout << t + 1 << ": " << result << endl;
    }
    fout.close();

    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    const bool verified = verifyResults("./data/result.dat", "./data/check.dat");
    if (verified) {
        cout << "Result verification passed." << endl;
    } else {
        cout << "Result verification failed." << endl;
    }

    return EXIT_SUCCESS;
}

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int num,
                      const double disturb0) {
    double result = 0.0;

#pragma omp parallel for reduction(+ : result) schedule(static)
    for (int i = 0; i < num; ++i) {
        const Complex diff = dat[i] - ctf[i] * pri[i];
        result += norm(diff) * sigRcp[i];
    }
    return result * disturb0;
}

bool verifyResults(const std::string& resultPath, const std::string& checkPath, double tolerance) {
    ifstream resultFile(resultPath);
    ifstream checkFile(checkPath);

    if (!resultFile.is_open() || !checkFile.is_open()) {
        cout << "Verification skipped: unable to open result or check file." << endl;
        return false;
    }

    string resultLine;
    string checkLine;
    int lineNumber = 0;

    while (true) {
        const bool resultOk = static_cast<bool>(getline(resultFile, resultLine));
        const bool checkOk = static_cast<bool>(getline(checkFile, checkLine));

        if (!resultOk || !checkOk) {
            if (resultOk != checkOk) {
                cout << "Verification failed: line count mismatch." << endl;
                return false;
            }
            break;
        }

        ++lineNumber;
        istringstream resultStream(resultLine);
        istringstream checkStream(checkLine);

        int resultIndex = 0;
        int checkIndex = 0;
        char colon;
        double resultValue = 0.0;
        double checkValue = 0.0;

        resultStream >> resultIndex >> colon >> resultValue;
        checkStream >> checkIndex >> colon >> checkValue;

        if (!resultStream || !checkStream) {
            cout << "Verification failed: parse error at line " << lineNumber << "." << endl;
            return false;
        }

        if (resultIndex != checkIndex) {
            cout << "Verification failed: index mismatch at line " << lineNumber << "." << endl;
            return false;
        }

        const double diff = fabs(resultValue - checkValue);
        if (diff > tolerance) {
            cout << "Verification failed at line " << lineNumber << ": diff=" << diff << " exceeds tolerance " << tolerance << "." << endl;
            return false;
        }
    }

    return true;
}
