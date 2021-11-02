//
// Created by ltc on 2021/11/1.
//

#ifndef TINYDL_CUMATRIX_H
#define TINYDL_CUMATRIX_H

#include <iostream>
#include <memory>
#include "../cuda/include/api.cuh"

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

using std::shared_ptr;
using std::static_pointer_cast;

struct Data {
	bool cuda;
	double* ptr;
	explicit Data(bool cuda) : cuda(cuda) {}
	~Data() {
		if (cuda) {
			cudaFree(ptr);
		} else {
			free(ptr);
		}
	}
};

class CuMatrix {
public:
	CuMatrix(int row, int col, bool cuda);
	CuMatrix(int row, int col, shared_ptr<Data> data);
	CuMatrix(const CuMatrix& c);
	CuMatrix(CuMatrix&& c) noexcept;
	CuMatrix& operator=(const CuMatrix& c);
	CuMatrix& operator=(CuMatrix&& c) noexcept;
	double operator()(int row, int col);
	void info();
	bool isCuda() const;
	void cuda();
	void cpu();
	void setValue(int row, int col, double value);
	void resize(int row, int col);
	CuMatrix operator+(const CuMatrix& c);
	CuMatrix& operator+=(const CuMatrix& c);
	CuMatrix operator-(const CuMatrix& c);
	CuMatrix& operator-=(const CuMatrix& c);
	CuMatrix operator*(const CuMatrix& c);
	CuMatrix operator*(double n);
	CuMatrix& operator*=(double n);
	CuMatrix operator/(double n);
	CuMatrix& operator/=(double n);
	Map<Matrix<double, Dynamic, Dynamic, RowMajor>>& operator*() const;
	CuMatrix log() const;
	CuMatrix exp() const;
	CuMatrix pow(double num) const;
	CuMatrix dot(const CuMatrix& c) const;
	CuMatrix transpose() const;
	void setOnes();
	void setZero();
	void setRandom();
	void setIdentity();
	double max();
	double min();
	~CuMatrix();
private:
	friend class ResizeOperator;
	shared_ptr<Data> data;
	Map<Matrix<double, Dynamic, Dynamic, RowMajor>>* matrix;
};

#endif //TINYDL_CUMATRIX_H
