//
// Created by ltc on 2021/9/6.
//
#include <memory>
#include "Tensor.h"
#include "operator/Operator.h"
#include "operator/AddOperator.h"
#include "operator/SubOperator.h"
#include "operator/MulOperator.h"
#include "operator/DivOperator.h"
#include "operator/PowOperator.h"
#include "operator/LogOperator.h"

Tensor::Tensor(double value) : op(nullptr), value(nullptr), _grad(nullptr), constValue(value), isConstant(true), rowNum(0), colNum(0) {

}

Tensor::Tensor(shared_ptr<Matrix<double, Dynamic, Dynamic>> value, shared_ptr<Operator> op) : value(value), op(op), isConstant(false), constValue(0), rowNum(value->rows()), colNum(value->cols())  {
	_grad = std::make_shared<Matrix<double, Dynamic, Dynamic>>(*value);
	_grad->setZero();
}

Matrix<double, Dynamic, Dynamic> Tensor::operator*() {
	if (isConstant) {
		auto m = Matrix<double, Dynamic, Dynamic>(1, 1);
		m << constValue;
		return m;
	}
	return *value;
}

Matrix<double, Dynamic, Dynamic> Tensor::grad() {
	if (isConstant) {
		auto m = Matrix<double, Dynamic, Dynamic>(1, 1);
		m << 0;
		return m;
	}
	return *_grad;
}

void Tensor::backward() {
	if (!isConstant && _grad->size() == 1) {
		_grad->setIdentity();
		_backward();
	}
}

void Tensor::_backward() {
	if (op != nullptr) {
		op->grad(*this);
		op->tensor1._backward();
		op->tensor2._backward();
	}
}

Tensor Tensor::operator+(const Tensor &t) {
	return (*shared_ptr<Operator>(new AddOperator(*this, t)))();
}

Tensor &Tensor::operator+=(const Tensor &t) {
	*this = *this + t;
	return *this;
}

Tensor &Tensor::operator++() {
	*this = *this + 1;
	return *this;
}

Tensor Tensor::operator++(int) {
	Tensor tmp = *this;
	++*this;
	return tmp;
}

Tensor Tensor::operator-(const Tensor &t) {
	return (*shared_ptr<Operator>(new SubOperator(*this, t)))();
}

Tensor &Tensor::operator-=(const Tensor &t) {
	*this = *this - t;
	return *this;
}

Tensor &Tensor::operator--() {
	*this = *this - 1;
	return *this;
}

Tensor Tensor::operator--(int) {
	Tensor tmp = *this;
	--*this;
	return tmp;
}

Tensor Tensor::operator*(const Tensor &t) {
	return (*shared_ptr<Operator>(new MulOperator(*this, t)))();
}

Tensor &Tensor::operator*=(const Tensor &t) {
	*this = *this * t;
	return *this;
}

Tensor Tensor::operator/(const Tensor &t) {
	return (*shared_ptr<Operator>(new DivOperator(*this, t)))();
}

Tensor &Tensor::operator/=(const Tensor &t) {
	*this = *this / t;
	return *this;
}

Tensor Tensor::log(double t) {
	return (*shared_ptr<Operator>(new LogOperator(t, *this)))();
}

Tensor Tensor::pow(double t) {
	return (*shared_ptr<Operator>(new PowOperator(*this, t)))();
}

Tensor Tensor::exp() {
	return (*shared_ptr<Operator>(new PowOperator(1.0, *this)))();
}