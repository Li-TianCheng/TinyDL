//
// Created by ltc on 2021/11/1.
//

#include "CuMatrix.h"
CuMatrix::CuMatrix(int row, int col, bool cuda) {
	data = shared_ptr<Data>(new Data(cuda));
	if (cuda) {
		cudaMalloc((void**)&(data->ptr), row * col * sizeof(double));
	} else {
		data->ptr = (double*)malloc(row * col * sizeof(double));
	}
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, row, col);
}

CuMatrix::CuMatrix(int row, int col, shared_ptr<Data> data) : data(data) {
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, row, col);
}

CuMatrix::CuMatrix(const CuMatrix& c) {
	data = shared_ptr<Data>(new Data(c.data->cuda));
	if (c.data->cuda) {
		cudaMalloc((void**)&(data->ptr), (*c).rows() * (*c).cols() * sizeof(double));
		cudaMemcpy(data->ptr, c.data->ptr, (*c).rows() * (*c).cols() * sizeof(double), cudaMemcpyDeviceToDevice);
	} else {
		data->ptr = (double*)malloc((*c).rows() * (*c).cols() * sizeof(double));
		memcpy(data->ptr, c.data->ptr, (*c).rows() * (*c).cols() * sizeof(double));
	}
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, (*c).rows(), (*c).cols());
}

CuMatrix::CuMatrix(CuMatrix &&c) noexcept : data(c.data) {
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, (*c).rows(), (*c).cols());
}

CuMatrix &CuMatrix::operator=(const CuMatrix& c) {
	data = shared_ptr<Data>(new Data(c.data->cuda));
	if (c.data->cuda) {
		cudaMalloc((void**)&(data->ptr), (*c).rows() * (*c).cols() * sizeof(double));
		cudaMemcpy(data->ptr, c.data->ptr, (*c).rows() * (*c).cols() * sizeof(double), cudaMemcpyDeviceToDevice);
	} else {
		data->ptr = (double*)malloc((*c).rows() * (*c).cols() * sizeof(double));
		memcpy(data->ptr, c.data->ptr, (*c).rows() * (*c).cols() * sizeof(double));
	}
	delete matrix;
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, (*c).rows(), (*c).cols());
	return *this;
}

CuMatrix &CuMatrix::operator=(CuMatrix&& c) noexcept {
	data = c.data;
	delete matrix;
	matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, (*c).rows(), (*c).cols());
	return *this;
}

CuMatrix::~CuMatrix() {
	delete matrix;
}

bool CuMatrix::isCuda() const {
	return data->cuda;
}

void CuMatrix::cuda() {
	if (!data->cuda) {
		auto tmp = data;
		data = shared_ptr<Data>(new Data(true));
		cudaMalloc((void**)&(data->ptr), matrix->rows() * matrix->cols() * sizeof(double));
		cudaMemcpy(data->ptr, tmp->ptr, matrix->rows() * matrix->cols() * sizeof(double), cudaMemcpyHostToDevice);
		auto prev = matrix;
		matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, prev->rows(), prev->cols());
		delete prev;
	}
}

void CuMatrix::cpu() {
	if (data->cuda) {
		auto tmp = data;
		data = shared_ptr<Data>(new Data(false));
		data->ptr = (double*)malloc(matrix->rows() * matrix->cols() * sizeof(double));
		cudaMemcpy(data->ptr, tmp->ptr, matrix->rows() * matrix->cols() * sizeof(double), cudaMemcpyDeviceToHost);
		auto prev = matrix;
		matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, prev->rows(), prev->cols());
		delete prev;
	}
}

CuMatrix CuMatrix::operator+(const CuMatrix &c) {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::add(*matrix, *c.matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = *matrix+*c.matrix;
		return r;
	}
}

CuMatrix CuMatrix::operator-(const CuMatrix &c) {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::sub(*matrix, *c.matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = *matrix-*c.matrix;
		return r;
	}
}

CuMatrix CuMatrix::operator*(const CuMatrix &c) {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), c.matrix->cols(), true);
		cuda::mul(*matrix, *c.matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), c.matrix->cols(), false);
		*r.matrix = *matrix * *c.matrix;
		return r;
	}
}

CuMatrix CuMatrix::operator*(double n) {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::numMul(*matrix, n, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = *matrix * n;
		return r;
	}
}

CuMatrix CuMatrix::operator/(double n) {
	return *this * (1.0/n);
}

void CuMatrix::setOnes() {
	if (data->cuda) {
		cuda::setValue(*matrix, 1);
	} else {
		matrix->setOnes();
	}
}

void CuMatrix::setZero() {
	if (data->cuda) {
		cuda::setValue(*matrix, 0);
	} else {
		matrix->setZero();
	}
}

void CuMatrix::setRandom() {
	if (data->cuda) {
		cpu();
		matrix->setRandom();
		cuda();
	} else {
		matrix->setRandom();
	}
}

void CuMatrix::setIdentity() {
	if (data->cuda) {
		cuda::setValue(*matrix, -1);
	} else {
		matrix->setIdentity();
	}
}

Map<Matrix<double, Dynamic, Dynamic, RowMajor>> &CuMatrix::operator*() const {
	return *matrix;
}

void CuMatrix::resize(int row, int col) {
	if (row*col <= matrix->rows()*matrix->cols()) {
		delete matrix;
		matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, row, col);
	} else {
		auto prev = matrix;
		auto tmp = data;
		data = shared_ptr<Data>(new Data(tmp->cuda));
		if (tmp->cuda) {
			cudaMalloc((void**)&(data->ptr), row * col * sizeof(double));
			cudaMemcpy(data->ptr, tmp->ptr, prev->rows()*prev->cols() * sizeof(double), cudaMemcpyDeviceToDevice);
		} else {
			data->ptr = (double*)malloc(row * col * sizeof(double));
			memcpy(data->ptr, tmp->ptr, prev->rows()*prev->cols() * sizeof(double));
		}
		matrix = new Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(data->ptr, row, col);
		delete prev;
	}
}

double CuMatrix::operator()(int row, int col) {
	if (data->cuda) {
		return cuda::getValue(*matrix, row, col);
	} else {
		return (*matrix)(row, col);
	}
}

void CuMatrix::info() {
	if (data->cuda) {
		std::cout << "device: cuda" << std::endl;
		cuda::info(*matrix);
	} else {
		std::cout << "device: cpu" << std::endl;
		std::cout << *matrix << std::endl;
	}
}

CuMatrix &CuMatrix::operator+=(const CuMatrix &c) {
	if (data->cuda) {
		cuda::add(*matrix, *c.matrix, *matrix);
	} else {
		*matrix += *c.matrix;
	}
	return *this;
}

CuMatrix &CuMatrix::operator*=(double n) {
	if (data->cuda) {
		cuda::numMul(*matrix, n, *matrix);
	} else {
		*matrix *= n;
	}
	return *this;
}

CuMatrix &CuMatrix::operator/=(double n) {
	*this *= (1.0/n);
	return *this;
}

CuMatrix &CuMatrix::operator-=(const CuMatrix &c) {
	if (data->cuda) {
		cuda::sub(*matrix, *c.matrix, *matrix);
	} else {
		*matrix -= *c.matrix;
	}
	return *this;
}

CuMatrix CuMatrix::log() const {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::log(*matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = ((*matrix).array().log()).matrix();
		return r;
	}
}

CuMatrix CuMatrix::exp() const {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::exp(*matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = ((*matrix).array().exp()).matrix();
		return r;
	}
}

CuMatrix CuMatrix::pow(double num) const {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::pow(*matrix, num, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = ((*matrix).array().pow(num)).matrix();
		return r;
	}
}

CuMatrix CuMatrix::dot(const CuMatrix& c) const {
	if (data->cuda) {
		CuMatrix r(matrix->rows(), matrix->cols(), true);
		cuda::dot(*matrix, *c.matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->rows(), matrix->cols(), false);
		*r.matrix = ((*matrix).array()*(*c.matrix).array()).matrix();
		return r;
	}
}

CuMatrix CuMatrix::transpose() const {
	if (data->cuda) {
		CuMatrix r(matrix->cols(), matrix->rows(), true);
		cuda::transpose(*matrix, *r.matrix);
		return r;
	} else {
		CuMatrix r(matrix->cols(), matrix->rows(), false);
		*r.matrix = (*matrix).transpose().eval();
		return r;
	}
}

double CuMatrix::max() {
	if (data->cuda) {
		return cuda::max(*matrix);
	} else {
		return matrix->maxCoeff();
	}
}

double CuMatrix::min() {
	if (data->cuda) {
		return cuda::min(*matrix);
	} else {
		return matrix->minCoeff();
	}
}






