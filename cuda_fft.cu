#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

const int m = 1638400;  // DO NOT CHANGE!!
const int K = 100000;   // DO NOT CHANGE!!

struct Complex {
    double real;
    double imag;

    __host__ __device__ Complex() : real(0.0), imag(0.0) {}
    __host__ __device__ Complex(double r, double i) : real(r), imag(i) {}
};

namespace {
struct DeviceBuffers {
    Complex* dat = nullptr;
    Complex* pri = nullptr;
    double* ctf = nullptr;
    double* sigRcp = nullptr;
    double* blockSums = nullptr;
    int num = 0;
    int blockCount = 0;
    int threadsPerBlock = 256;
} deviceBuffers;

inline void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__device__ inline Complex scaleComplex(const Complex& c, const double factor) {
    return Complex(c.real * factor, c.imag * factor);
}

__device__ inline Complex subtractComplex(const Complex& a, const Complex& b) {
    return Complex(a.real - b.real, a.imag - b.imag);
}

__device__ inline double normComplex(const Complex& c) {
    return c.real * c.real + c.imag * c.imag;
}

__global__ void computeDiffKernel(const Complex* dat,
                                  const Complex* pri,
                                  const double* ctf,
                                  const double* sigRcp,
                                  double* blockSums,
                                  const int num) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int globalThreadId = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;

    double localSum = 0.0;
    for (int idx = globalThreadId; idx < num; idx += stride) {
        const Complex ctfPri = scaleComplex(pri[idx], ctf[idx]);
        const Complex diff = subtractComplex(dat[idx], ctfPri);
        localSum += normComplex(diff) * sigRcp[idx];
    }

    sdata[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

void initDeviceMemory(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int num) {
    if (deviceBuffers.dat != nullptr && deviceBuffers.num == num) {
        return;
    }

    if (deviceBuffers.dat != nullptr) {
        cudaFree(deviceBuffers.dat);
        cudaFree(deviceBuffers.pri);
        cudaFree(deviceBuffers.ctf);
        cudaFree(deviceBuffers.sigRcp);
        cudaFree(deviceBuffers.blockSums);
    }

    deviceBuffers.num = num;
    const size_t complexBytes = static_cast<size_t>(num) * sizeof(Complex);
    const size_t doubleBytes = static_cast<size_t>(num) * sizeof(double);

    checkCuda(cudaMalloc(&deviceBuffers.dat, complexBytes), "Failed to allocate device dat");
    checkCuda(cudaMalloc(&deviceBuffers.pri, complexBytes), "Failed to allocate device pri");
    checkCuda(cudaMalloc(&deviceBuffers.ctf, doubleBytes), "Failed to allocate device ctf");
    checkCuda(cudaMalloc(&deviceBuffers.sigRcp, doubleBytes), "Failed to allocate device sigRcp");

    checkCuda(cudaMemcpy(deviceBuffers.dat, dat, complexBytes, cudaMemcpyHostToDevice),
              "Failed to copy dat to device");
    checkCuda(cudaMemcpy(deviceBuffers.pri, pri, complexBytes, cudaMemcpyHostToDevice),
              "Failed to copy pri to device");
    checkCuda(cudaMemcpy(deviceBuffers.ctf, ctf, doubleBytes, cudaMemcpyHostToDevice),
              "Failed to copy ctf to device");
    checkCuda(cudaMemcpy(deviceBuffers.sigRcp, sigRcp, doubleBytes, cudaMemcpyHostToDevice),
              "Failed to copy sigRcp to device");

    const int maxBlocks = 4096;
    const int blocksNeeded = (num + deviceBuffers.threadsPerBlock - 1) / deviceBuffers.threadsPerBlock;
    deviceBuffers.blockCount = std::min(blocksNeeded, maxBlocks);
    if (deviceBuffers.blockCount < 1) {
        deviceBuffers.blockCount = 1;
    }

    checkCuda(cudaMalloc(&deviceBuffers.blockSums,
                         static_cast<size_t>(deviceBuffers.blockCount) * sizeof(double)),
              "Failed to allocate block sums");
}

void releaseDeviceMemory() {
    if (deviceBuffers.dat == nullptr) {
        return;
    }
    cudaFree(deviceBuffers.dat);
    cudaFree(deviceBuffers.pri);
    cudaFree(deviceBuffers.ctf);
    cudaFree(deviceBuffers.sigRcp);
    cudaFree(deviceBuffers.blockSums);
    deviceBuffers = DeviceBuffers{};
}
}  // namespace

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int num,
                      const double disturb0);
bool verifyResults(const std::string& resultPath, const std::string& checkPath, double tolerance = 1e-5);

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int num,
                      const double disturb0) {
    (void)dat;
    (void)pri;
    (void)ctf;
    (void)sigRcp;
    (void)num;

    if (deviceBuffers.blockCount == 0) {
        std::cerr << "Device buffers are not initialised" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int threads = deviceBuffers.threadsPerBlock;
    const int blocks = deviceBuffers.blockCount;
    const size_t sharedMem = static_cast<size_t>(threads) * sizeof(double);

    computeDiffKernel<<<blocks, threads, sharedMem>>>(deviceBuffers.dat,
                                                      deviceBuffers.pri,
                                                      deviceBuffers.ctf,
                                                      deviceBuffers.sigRcp,
                                                      deviceBuffers.blockSums,
                                                      num);
    checkCuda(cudaGetLastError(), "Kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");

    std::vector<double> hostBlockSums(deviceBuffers.blockCount);
    checkCuda(cudaMemcpy(hostBlockSums.data(),
                         deviceBuffers.blockSums,
                         static_cast<size_t>(deviceBuffers.blockCount) * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "Failed to copy block sums to host");

    double result = 0.0;
    for (double value : hostBlockSums) {
        result += value;
    }
    return result * disturb0;
}

bool verifyResults(const std::string& resultPath, const std::string& checkPath, double tolerance) {
    std::ifstream resultFile(resultPath);
    std::ifstream checkFile(checkPath);

    if (!resultFile.is_open() || !checkFile.is_open()) {
        std::cout << "Verification skipped: unable to open result or check file." << std::endl;
        return false;
    }

    std::string resultLine;
    std::string checkLine;
    int lineNumber = 0;

    while (true) {
        const bool resultOk = static_cast<bool>(std::getline(resultFile, resultLine));
        const bool checkOk = static_cast<bool>(std::getline(checkFile, checkLine));

        if (!resultOk || !checkOk) {
            if (resultOk != checkOk) {
                std::cout << "Verification failed: line count mismatch." << std::endl;
                return false;
            }
            break;
        }

        ++lineNumber;
        std::istringstream resultStream(resultLine);
        std::istringstream checkStream(checkLine);

        int resultIndex = 0;
        int checkIndex = 0;
        char colon;
        double resultValue = 0.0;
        double checkValue = 0.0;

        resultStream >> resultIndex >> colon >> resultValue;
        checkStream >> checkIndex >> colon >> checkValue;

        if (!resultStream || !checkStream) {
            std::cout << "Verification failed: parse error at line " << lineNumber << "." << std::endl;
            return false;
        }

        if (resultIndex != checkIndex) {
            std::cout << "Verification failed: index mismatch at line " << lineNumber << "." << std::endl;
            return false;
        }

        const double diff = std::fabs(resultValue - checkValue);
        if (diff > tolerance) {
            std::cout << "Verification failed at line " << lineNumber << ": diff=" << diff
                      << " exceeds tolerance " << tolerance << "." << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::vector<Complex> dat(m);
    std::vector<Complex> pri(m);
    std::vector<double> ctf(m);
    std::vector<double> sigRcp(m);
    std::vector<double> disturb(K);

    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    std::ifstream fin;
    fin.open("./data/input.dat");
    if (!fin.is_open()) {
        std::cout << "Error opening file input.dat" << std::endl;
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
        std::cout << "Error opening file K.dat" << std::endl;
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

    initDeviceMemory(dat.data(), pri.data(), ctf.data(), sigRcp.data(), m);

    auto startTime = Clock::now();

    std::ofstream fout;
    fout.open("./data/result.dat");
    if (!fout.is_open()) {
        std::cout << "Error opening file for result" << std::endl;
        releaseDeviceMemory();
        return EXIT_FAILURE;
    }

    for (unsigned int t = 0; t < K; ++t) {
        double result = logDataVSPrior(dat.data(), pri.data(), ctf.data(), sigRcp.data(), m, disturb[t]);
        fout << t + 1 << ": " << result << std::endl;
    }
    fout.close();

    auto endTime = Clock::now();
    auto compTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "Computing time=" << compTime.count() << " microseconds" << std::endl;

    const bool verified = verifyResults("./data/result.dat", "./data/check.dat");
    if (verified) {
        std::cout << "Result verification passed." << std::endl;
    } else {
        std::cout << "Result verification failed." << std::endl;
    }

    releaseDeviceMemory();

    return EXIT_SUCCESS;
}
