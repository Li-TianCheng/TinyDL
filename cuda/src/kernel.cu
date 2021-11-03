//
// Created by ltc on 2021/11/1.
//
#include "kernel.cuh"

__global__ void kernelAdd(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = m1(row, col) + m2(row, col);
	}
}

__global__ void kernelSub(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = m1(row, col) - m2(row, col);
	}
}

__global__ void kernelMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x * CALCULATE_NUM + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ double sharedM1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double sharedM2[BLOCK_SIZE][BLOCK_SIZE];
	double re[CALCULATE_NUM] = {0};
	int num = (m1.cols()-1) / BLOCK_SIZE + 1;
	for (int i = 0; i < num; ++i) {
#pragma unroll 8
		for (int t = 0; t < CALCULATE_NUM; ++t) {
			double tmp = 0;
			if (row+t*BLOCK_SIZE/CALCULATE_NUM < m1.rows() && i*BLOCK_SIZE+threadIdx.y < m1.cols()) {
				tmp = m1(row+t*BLOCK_SIZE/CALCULATE_NUM, i*BLOCK_SIZE+threadIdx.y);
			}
			sharedM1[threadIdx.x+t*BLOCK_SIZE/CALCULATE_NUM][threadIdx.y] = tmp;
			tmp = 0;
			if (i*BLOCK_SIZE+threadIdx.x+t*BLOCK_SIZE/CALCULATE_NUM < m2.rows() && col < m2.cols()) {
				tmp = m2(i*BLOCK_SIZE+threadIdx.x+t*BLOCK_SIZE/CALCULATE_NUM, col);
			}
			sharedM2[threadIdx.x+t*BLOCK_SIZE/CALCULATE_NUM][threadIdx.y] = tmp;
		}
		__syncthreads();
#pragma unroll 32
		for (int j = 0; j < BLOCK_SIZE; ++j) {
#pragma unroll 8
			for (int t = 0; t < CALCULATE_NUM; ++t) {
				re[t] += sharedM1[threadIdx.x+t*BLOCK_SIZE/CALCULATE_NUM][j] * sharedM2[j][threadIdx.y];
			}
		}
		__syncthreads();
	}
#pragma unroll 8
	for (int t = 0; t < CALCULATE_NUM; ++t) {
		if (row+t*BLOCK_SIZE/CALCULATE_NUM < r.rows() && col < r.cols()) {
			r(row+t*BLOCK_SIZE/CALCULATE_NUM, col) = re[t];
		}
	}
}

__global__ void kernelNumMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                             double num,
                             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = m1(row, col) * num;
	}
}

__global__ void kernelDot(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = m1(row, col) * m2(row, col);
	}
}

__global__ void kernelLog(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = log(m1(row, col));
	}
}

__global__ void kernelMaxPool(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                              int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                              Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int x = threadIdx.x / dataCol;
	int y = threadIdx.x % dataCol;
	for (int k = 0; k < kernelRow*kernelCol; ++k) {
		int kx = k / kernelCol;
		int ky = k % kernelCol;
		int kx0 = x - kx;
		int ky0 = y - ky;
		if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
			int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
			for (int c = 0; c < channel; ++c) {
				r(blockIdx.x, n+c*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)) = max(r(blockIdx.x, n+c*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)), m1(blockIdx.x, threadIdx.x+c*dataRow*dataCol));
			}
		}
	}
}

__global__ void kernelMaxPoolBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                                int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m3,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int x = threadIdx.x / dataCol;
	int y = threadIdx.x % dataCol;
	for (int k = 0; k < kernelRow*kernelCol; ++k) {
		int kx = k / kernelCol;
		int ky = k % kernelCol;
		int kx0 = x - kx;
		int ky0 = y - ky;
		if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
			int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
			for (int c = 0; c < channel; ++c) {
				if (m2(blockIdx.x, n+c*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)) == m1(blockIdx.x, threadIdx.x+c*dataRow*dataCol)) {
					r(blockIdx.x, threadIdx.x+c*dataRow*dataCol) += m3(blockIdx.x, n+c*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1));
				}
			}
		}
	}
}

__global__ void kernelExp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = exp(m1(row, col));
	}
}

__global__ void kernelPow(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          double num,
                          Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = pow(m1(row, col), num);
	}
}

__global__ void kernelConvToImg(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int num,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m1.rows() && col < m1.cols()) {
		r(row/num, col*num+row%num) = m1(row, col);
	}
}

__global__ void kernelConvToImgBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                  int num,
                                  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) += m1(row/num, col*num+row%num);
	}
}

__global__ void kernelImgToConv(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int x = threadIdx.x / dataCol;
	int y = threadIdx.x % dataCol;
	for (int k = 0; k < kernelRow*kernelCol; ++k) {
		int kx = k / kernelCol;
		int ky = k % kernelCol;
		int kx0 = x - kx;
		int ky0 = y - ky;
		if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
			int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
			for (int c = 0; c < channel; ++c) {
				r(blockIdx.x*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)+n, k+c*kernelRow*kernelCol) = m1(blockIdx.x, threadIdx.x+c*dataRow*dataCol);
			}
		}
	}
}

__global__ void kernelImgToConvBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                  int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int x = threadIdx.x / dataCol;
	int y = threadIdx.x % dataCol;
	for (int k = 0; k < kernelRow*kernelCol; ++k) {
		int kx = k / kernelCol;
		int ky = k % kernelCol;
		int kx0 = x - kx;
		int ky0 = y - ky;
		if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
			int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
			for (int c = 0; c < channel; ++c) {
				r(blockIdx.x, threadIdx.x+c*dataRow*dataCol) += m1(blockIdx.x*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)+n, k+c*kernelRow*kernelCol);
			}
		}
	}
}

__global__ void kernelSetValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                               double num) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m1.rows() && col < m1.cols()) {
		if (num == -1) {
			if (row == col) {
				m1(row, col) = 1;
			} else {
				m1(row, col) = 0;
			}
		} else {
			m1(row, col) = num;
		}
	}
}

__global__ void kernelSetValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                               int row, int col, double value) {
	m1(row, col) = value;
}

__global__ void kernelGetValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                               int row, int col, double* r) {
	*r = m1(row, col);
}

__global__ void kernelInfo(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	for (int i = 0; i < m1.rows(); ++i) {
		for (int j = 0; j < m1.cols(); ++j) {
			printf("%f ", m1(i, j));
		}
		printf("\n");
	}
}

__global__ void kernelTranspose(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < r.rows() && col < r.cols()) {
		r(row, col) = m1(col, row);
	}
}

__global__ void kernelMax(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          double* r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ double mem[BLOCK_SIZE*BLOCK_SIZE];
	double tmp = row < m1.rows() && col < m1.cols() ? m1(row, col) : DBL_MIN;
	int idx = threadIdx.x*BLOCK_SIZE + threadIdx.y;
	mem[idx] = tmp;
	__syncthreads();
#pragma unroll 9
	for (int stride = BLOCK_SIZE*BLOCK_SIZE/2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (idx < stride) {
			if (mem[idx] < mem[idx+stride]) {
				mem[idx] = mem[idx+stride];
			}
		}
	}
	if (idx == 0) {
		r[blockIdx.x*gridDim.y+blockIdx.y] = mem[0];
	}
}

__global__ void kernelMin(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                          double* r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ double mem[BLOCK_SIZE*BLOCK_SIZE];
	double tmp = row < m1.rows() && col < m1.cols() ? m1(row, col) : DBL_MAX;
	int idx = threadIdx.x*BLOCK_SIZE + threadIdx.y;
	mem[idx] = tmp;
	__syncthreads();
#pragma unroll 9
	for (int stride = BLOCK_SIZE*BLOCK_SIZE/2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (idx < stride) {
			if (mem[idx] > mem[idx+stride]) {
				mem[idx] = mem[idx+stride];
			}
		}
	}
	if (idx == 0) {
		r[blockIdx.x*gridDim.y+blockIdx.y] = mem[0];
	}
}

__global__ void kernelRelu(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m1.rows() && col < m1.cols()) {
		if (m1(row, col) < 0) {
			m1(row, col) = 0;
		}
	}
}

__global__ void kernelReluBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < m1.rows() && col < m1.cols()) {
		if (m1(row, col) != 0) {
			r(row, col) += m2(row, col);
		}
	}
}
