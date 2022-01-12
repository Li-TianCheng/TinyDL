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
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int deltaX = blockDim.x * gridDim.x;
	const unsigned int deltaY = blockDim.y * gridDim.y;
	const unsigned int blockSize = BLOCK_SIZE/CALCULATE_NUM;
	__shared__ double sharedM1[CALCULATE_NUM][blockSize][blockSize];
	__shared__ double sharedM2[CALCULATE_NUM][blockSize][blockSize];
	double re[CALCULATE_NUM][CALCULATE_NUM] = {0};
	unsigned int num = (m1.cols()-1) / blockSize + 1;
	for (int i = 0; i < num; ++i) {
		unsigned int y1 = threadIdx.y + i * blockSize;
		unsigned int x2 = threadIdx.x + i * blockSize;
		for (int j = 0; j < CALCULATE_NUM; ++j) {
			unsigned int x1 = row + j * deltaX;
			unsigned int y2 = col + j * deltaY;
			sharedM1[j][threadIdx.x][threadIdx.y] = x1 < m1.rows() && y1 < m1.cols() ? m1(x1, y1) : 0;
			sharedM2[j][threadIdx.x][threadIdx.y] = x2 < m2.rows() && y2 < m2.cols() ? m2(x2, y2) : 0;
		}
		__syncthreads();
		for (int j = 0; j < blockSize; ++j) {
			for (int k = 0; k < CALCULATE_NUM; ++k) {
				for (int l = 0; l < CALCULATE_NUM; ++l) {
					re[k][l] += sharedM1[k][threadIdx.x][j] * sharedM2[l][j][threadIdx.y];
				}
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < CALCULATE_NUM; ++i) {
		unsigned int x = row + i * deltaX;
		for (int j = 0; j < CALCULATE_NUM; ++j) {
			unsigned int y = col + j * deltaY;
			if (x < r.rows() && y < r.cols()) {
				r(x, y) = re[i][j];
			}
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
                              int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                              Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.x;
	if (j < ((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)) {
		int kx = j / ((dataCol-kernelCol)/stride+1);
		int ky = j % ((dataCol-kernelCol)/stride+1);
		int x0 = kx*stride;
		int y0 = ky*stride;
		double value = DBL_MIN;
		for (int m = 0; m < kernelRow; ++m) {
			for (int n = 0; n < kernelCol; ++n) {
				int x = x0 + m;
				int y = y0 + n;
				int idx = x*dataCol+y+blockIdx.z*dataRow*dataCol;
				value = max(value, m1(i, idx));
			}
		}
		r(i, j+blockIdx.z*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)) = value;
	}
}

__global__ void kernelMaxPoolBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
                                int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m3,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.x;
	if (j < dataRow*dataCol) {
		int x = j / dataCol;
		int y = j % dataCol;
		double value = r(i, j+blockIdx.z*dataRow*dataCol);
		double max = m1(i, j+blockIdx.z*dataRow*dataCol);
		for (int k = 0; k < kernelRow*kernelCol; ++k) {
			int kx = k / kernelCol;
			int ky = k % kernelCol;
			int kx0 = x - kx;
			int ky0 = y - ky;
			if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
				int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
				if (m2(i, n+blockIdx.z*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)) == max) {
					value += m3(i, n+blockIdx.z*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1));
				}
			}
		}
		r(i, j+blockIdx.z*dataRow*dataCol) = value;
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
                                int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.x;
	if (j < dataRow*dataCol) {
		int x = j / dataCol;
		int y = j % dataCol;
		double value = m1(i, j+blockIdx.z*dataRow*dataCol);
		for (int k = 0; k < kernelRow*kernelCol; ++k) {
			int kx = k / kernelCol;
			int ky = k % kernelCol;
			int kx0 = x - kx;
			int ky0 = y - ky;
			if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
				int n = kx0 / stride * ((dataRow - kernelRow) / stride + 1) + ky0 / stride;
				r(i*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)+n, k+blockIdx.z*kernelRow*kernelCol) = value;
			}
		}
	}
}

__global__ void kernelImgToConvBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                                  int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                                  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	int i = blockIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.x;
	if (j < dataRow*dataCol) {
		int x = j / dataCol;
		int y = j % dataCol;
		double value = r(i, j+blockIdx.z*dataCol*dataRow);
		for (int k = 0; k < kernelRow*kernelCol; ++k) {
			int kx = k / kernelCol;
			int ky = k % kernelCol;
			int kx0 = x - kx;
			int ky0 = y - ky;
			if (kx0 >= 0 && ky0 >= 0 && kx0 <= dataRow-kernelRow && ky0 <= dataCol-kernelCol && kx0 % stride == 0 && ky0 % stride == 0) {
				int n = kx0 / stride * ((dataRow-kernelRow)/stride+1) + ky0 / stride;
				value += m1(blockIdx.x*((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)+n, k+blockIdx.z*kernelRow*kernelCol);
			}
		}
		r(i, j+blockIdx.z*dataCol*dataRow) = value;
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
