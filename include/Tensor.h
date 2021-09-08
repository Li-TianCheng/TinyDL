//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_TENSOR_H
#define TINYDL_TENSOR_H

#include <eigen3/Eigen/Core>
#include <memory>
#include <iostream>

using std::shared_ptr;
using std::static_pointer_cast;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;

class Operator;

class Tensor {
public:
	Tensor(double value);
	template<int rowNum, int colNum>
	Tensor(Matrix<double, rowNum, colNum, RowMajor> value);
	Tensor(const Tensor& t) = default;
	Tensor(Tensor&& t) = default;
	Tensor& operator=(const Tensor& t) = default;
	Tensor& operator=(Tensor&& t) = default;
	~Tensor() = default;
	double operator()(int row, int col);
	void backward();
	void clearGradient();
	Matrix<double, Dynamic, Dynamic, RowMajor> grad();
	Tensor operator+(const Tensor& t);
	Tensor& operator+=(const Tensor& t);
	Tensor& operator++();
	Tensor operator++(int);
	Tensor operator-(const Tensor& t);
	Tensor& operator-=(const Tensor& t);
	Tensor& operator--();
	Tensor operator--(int);
	Tensor operator*(const Tensor& t);
	Tensor& operator*=(const Tensor& t);
	Tensor operator/(const Tensor& t);
	Tensor& operator/=(const Tensor& t);
	Tensor log(double t=std::exp(1.0));
	Tensor pow(double t);
	Tensor exp();
	Tensor resize(int rowNum, int colNum, bool isNew=false);
	Tensor transpose(bool isNew=false);
	Tensor dot(const Tensor& t);
private:
	friend std::ostream& operator<<(std::ostream &out, Tensor& t);
	friend class Operator;
	friend class AddOperator;
	friend class SubOperator;
	friend class MulOperator;
	friend class DivOperator;
	friend class PowOperator;
	friend class LogOperator;
	friend class ResizeOperator;
	friend class TransposeOperator;
	friend class DotOperator;
private:
	Tensor(shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>>, shared_ptr<Operator> op);
	void _backward();
private:
	bool isConstant;
	double constValue;
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value;
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> gradient;
	shared_ptr<Operator> op;
};

template<int rowNum, int colNum> inline
Tensor::Tensor(Matrix<double, rowNum, colNum, RowMajor> value) : op(nullptr), isConstant(false), constValue(0) {
	this->value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(value);
	gradient = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(value);
	gradient->setZero();
}

std::ostream& operator<<(std::ostream &out, Tensor& t);

#endif //TINYDL_TENSOR_H
