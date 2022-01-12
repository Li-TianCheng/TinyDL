//
// Created by ltc on 2021/11/1.
//
#include "api.cuh"

void cuda::add(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelAdd<<<grid, block>>>(m1, m2, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelAdd launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::sub(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelSub<<<grid, block>>>(m1, m2, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelSub launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::mul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE/CALCULATE_NUM, BLOCK_SIZE/CALCULATE_NUM);
	kernelMul<<<grid, block>>>(m1, m2, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelMul launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::numMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
            double num,
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelNumMul<<<grid, block>>>(m1, num, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelNumMul launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::dot(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelDot<<<grid, block>>>(m1, m2, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelDot launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::log(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelLog<<<grid, block>>>(m1, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelLog launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::maxPool(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
             int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid(r.rows(), ((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1)/BLOCK_SIZE/BLOCK_SIZE+1, channel);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	kernelMaxPool<<<grid, block>>>(m1, dataRow, dataCol, kernelRow, kernelCol, stride, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelMaxPool launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::maxPoolBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                     Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
					 int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                     Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m3,
					 Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid(m1.rows(), (dataRow*dataCol-1)/BLOCK_SIZE/BLOCK_SIZE+1, channel);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	kernelMaxPoolBp<<<grid, block>>>(m1, m2, dataRow, dataCol, kernelRow, kernelCol, stride, m3, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelMaxPoolBp launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::exp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelExp<<<grid, block>>>(m1, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelExp launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::pow(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
         double num,
         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelPow<<<grid, block>>>(m1, num, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelPow launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::convToImg(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
               int num,
               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelConvToImg<<<grid, block>>>(m1, num, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelConvToImg launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::convToImgBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                 int num,
                 Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelConvToImgBp<<<grid, block>>>(m1, num, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelConvToImgBp launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::imgToConv(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
               int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid(m1.rows(), (dataRow*dataCol-1)/BLOCK_SIZE/BLOCK_SIZE+1, channel);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	kernelImgToConv<<<grid, block>>>(m1, dataRow, dataCol, kernelRow, kernelCol, stride, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelImgToConv launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::imgToConvBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                 int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
                 Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid(r.rows(), (dataRow*dataCol-1)/BLOCK_SIZE/BLOCK_SIZE+1, channel);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	kernelImgToConvBp<<<grid, block>>>(m1, dataRow, dataCol, kernelRow, kernelCol, stride, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelImgToConvBp launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::setValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
              double num) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelSetValue<<<grid, block>>>(m1, num);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelSetValue launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

double cuda::getValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
                int row, int col) {
	double* r;
	double result;
	cudaMalloc((void**)&r, sizeof(double));
	kernelGetValue<<<1, 1>>>(m1, row, col, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelGetValue launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
	cudaMemcpy(&result, r, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(r);
	return result;
}

void cuda::info(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	kernelInfo<<<1, 1>>>(m1);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelInfo launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::transpose(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((r.rows()-1)/BLOCK_SIZE+1, (r.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelTranspose<<<grid, block>>>(m1, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelTranspose launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

double cuda::max(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	double* r;
	cudaMalloc((void**)&r, grid.x*grid.y*sizeof(double));
	kernelMax<<<grid, block>>>(m1, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelMax launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
	double tmp[grid.x*grid.y];
	cudaMemcpy(tmp, r, grid.x*grid.y*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(r);
	double result = DBL_MIN;
	for (int i = 0; i < grid.x*grid.y; ++i) {
		result = std::max(result, tmp[i]);
	}
	return result;
}

double cuda::min(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	double* r;
	cudaMalloc((void**)&r, grid.x*grid.y*sizeof(double));
	kernelMin<<<grid, block>>>(m1, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelMin launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
	double tmp[grid.x*grid.y];
	cudaMemcpy(tmp, r, grid.x*grid.y*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(r);
	double result = DBL_MAX;
	for (int i = 0; i < grid.x*grid.y; ++i) {
		result = std::min(result, tmp[i]);
	}
	return result;
}

void cuda::relu(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelRelu<<<grid, block>>>(m1);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelRelu launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}

void cuda::reluBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r) {
	dim3 grid((m1.rows()-1)/BLOCK_SIZE+1, (m1.cols()-1)/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	kernelReluBp<<<grid, block>>>(m1, m2, r);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelReluBp launch failed:%s\n", cudaGetErrorString(cudaStatus));
	}
}
