//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_TENSOR_H
#define TINYDL_TENSOR_H

#define EIGEN_USE_MKL_VML
#define EIGEN_VECTORIZE_SSE4_2

#include <eigen3/Eigen/Core>
#include <memory>
#include <iostream>
#include <vector>

using std::shared_ptr;
using std::static_pointer_cast;
using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;

class Operator;

class Tensor {
public:
	Tensor(double value);
	Tensor(int rowNum, int colNum);
	explicit Tensor(const vector<vector<double>>& v);
	template<int rowNum, int colNum>
	explicit Tensor(Matrix<double, rowNum, colNum, RowMajor> value);
	Tensor(shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>>, shared_ptr<Operator> op);
	Tensor(const Tensor& t) = default;
	Tensor(Tensor&& t) = default;
	Tensor& operator=(const Tensor& t) = default;
	Tensor& operator=(Tensor&& t) = default;
	~Tensor() = default;
	double operator()(int row, int col);
	void backward();
	void clearGradient();
	void setZero();
	void setOnes();
	void setIdentity();
	void setRandom();
	bool isConstant() const;
	int row() const;
	int col() const;
	Tensor copy() const;
	Matrix<double, Dynamic, Dynamic, RowMajor>& operator*();
	Matrix<double, Dynamic, Dynamic, RowMajor>& grad();
	Tensor operator+(const Tensor& t) const;
	Tensor& operator+=(const Tensor& t);
	Tensor& operator++();
	Tensor operator++(int);
	Tensor operator-(const Tensor& t) const;
	Tensor& operator-=(const Tensor& t);
	Tensor& operator--();
	Tensor operator--(int);
	Tensor operator*(const Tensor& t) const;
	Tensor& operator*=(const Tensor& t);
	Tensor operator/(const Tensor& t) const;
	Tensor& operator/=(const Tensor& t);
	Tensor log(double t=std::exp(1.0)) const;
	Tensor pow(double t) const;
	Tensor exp() const;
	Tensor resize(int rowNum, int colNum, bool isNew=false) const;
	Tensor transpose(bool isNew=false) const;
	Tensor dot(const Tensor& t) const;
private:
	friend std::ostream& operator<<(std::ostream &out, const Tensor& t);
	void _backward();
private:
	bool constant;
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value;
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> gradient;
	shared_ptr<Operator> op;
};

template<int rowNum, int colNum> inline
Tensor::Tensor(Matrix<double, rowNum, colNum, RowMajor> value) : op(nullptr), constant(false) {
	this->value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(value);
	gradient = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(value);
	gradient->setZero();
}

std::ostream& operator<<(std::ostream &out, const Tensor& t);

#endif //TINYDL_TENSOR_H
