//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_TENSOR_H
#define TINYDL_TENSOR_H

#include <eigen3/Eigen/Core>
#include <memory>

using std::shared_ptr;
using std::static_pointer_cast;
using Eigen::Matrix;
using Eigen::Dynamic;

class Operator;

class Tensor {
public:
	Tensor(double value);
	template<int row, int col>
	Tensor(Matrix<double, row, col> value);
	Tensor(const Tensor& t) = default;
	Tensor(Tensor&& t) = default;
	Tensor& operator=(const Tensor& t) = default;
	Tensor& operator=(Tensor&& t) = default;
	~Tensor() = default;
	void backward();
	Matrix<double, Dynamic, Dynamic> operator*();
	Matrix<double, Dynamic, Dynamic> grad();
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
private:
	friend class Operator;
	friend class AddOperator;
	friend class SubOperator;
	friend class MulOperator;
	friend class DivOperator;
	friend class PowOperator;
	friend class LogOperator;
private:
	Tensor(shared_ptr<Matrix<double, Dynamic, Dynamic>>, shared_ptr<Operator> op);
	void _backward();
private:
	bool isConstant;
	double constValue;
	int rowNum;
	int colNum;
	shared_ptr<Matrix<double, Dynamic, Dynamic>> value;
	shared_ptr<Matrix<double, Dynamic, Dynamic>> _grad;
	shared_ptr<Operator> op;
};

template<int row, int col> inline
Tensor::Tensor(Matrix<double, row, col> value) : op(nullptr), isConstant(false), constValue(0), rowNum(row), colNum(col) {
	this->value = std::make_shared<Matrix<double, Dynamic, Dynamic>>(value);
	_grad = std::make_shared<Matrix<double, Dynamic, Dynamic>>(value);
	_grad->setZero();
}

#endif //TINYDL_TENSOR_H
