//
// Created by ltc on 2021/11/1.
//

#ifndef TINYDL_KERNEL_CUH
#define TINYDL_KERNEL_CUH

#include <cuda_runtime.h>
#include <eigen3/Eigen/Core>
#include <cfloat>

using Eigen::Matrix;
using Eigen::Map;
using Eigen::RowMajor;
using Eigen::Dynamic;

#define BLOCK_SIZE 32
#define CALCULATE_NUM 4

__global__ void kernelAdd(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelSub(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelNumMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						     double num,
						     Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelDot(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelLog(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelMaxPool(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
							  int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
							  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelMaxPoolBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                                int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m3,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelExp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelPow(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  double num,
						  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelConvToImg(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int num,
								Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelConvToImgBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int num,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelImgToConv(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelImgToConvBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelSetValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
							   double num);

__global__ void kernelGetValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                               int row, int col, double* r);

__global__ void kernelInfo(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

__global__ void kernelTranspose(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

__global__ void kernelMax(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
						  double* r);

__global__ void kernelMin(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          double* r);

__global__ void kernelRelu(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

__global__ void kernelReluBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

#endif //TINYDL_KERNEL_CUH
