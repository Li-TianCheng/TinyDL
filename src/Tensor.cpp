//
// Created by ltc on 2021/9/6.
//
#include "Tensor.h"
#include "operator/Operator.h"
#include "operator/AddOperator.h"
#include "operator/SubOperator.h"
#include "operator/MulOperator.h"
#include "operator/DivOperator.h"
#include "operator/PowOperator.h"
#include "operator/LogOperator.h"
#include "operator/ResizeOperator.h"
#include "operator/TransposeOperator.h"
#include "operator/DotOperator.h"

Tensor::Tensor(double value) : op(nullptr), value(nullptr), gradient(nullptr), isConstant(true) {
	this->value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(1, 1);
	*this->value << value;
	gradient = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(1, 1);
}

Tensor::Tensor(int rowNum, int colNum) : op(nullptr), isConstant(false) {
	value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>();
	value->resize(rowNum, colNum);
	gradient = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>();
	gradient->resize(rowNum, colNum);
}

Tensor::Tensor(shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value, shared_ptr<Operator> op) : value(value), op(op), isConstant(false) {
	gradient = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*value);
	gradient->setZero();
}

double Tensor::operator()(int row, int col) {
	return (*value)(row, col);
}

int Tensor::row() {
	return value->rows();
}

int Tensor::col() {
	return value->cols();
}

std::ostream& operator<<(std::ostream &out, const Tensor& t) {
	out << *t.value;
	return out;
}

void Tensor::clearGradient() {
	gradient->setZero();
}

void Tensor::setZero() {
	value->setZero();
}

void Tensor::setOnes() {
	value->setOnes();
}

void Tensor::setIdentity() {
	value->setIdentity();
}

void Tensor::setRandom() {
	value->setRandom();
}

Tensor Tensor::copy() {
	return Tensor(*value);
}

Matrix<double, Dynamic, Dynamic, RowMajor>& Tensor::operator*() {
	return *value;
}

Matrix<double, Dynamic, Dynamic, RowMajor>& Tensor::grad() {
	return *gradient;
}

void Tensor::backward() {
	if (!isConstant) {
		gradient->resize(value->rows(), value->cols());
		gradient->setOnes();
		_backward();
	}
}

void Tensor::_backward() {
	if (op != nullptr) {
		op->backward(*this);
		op->tensor1._backward();
		op->tensor2._backward();
	}
	op = nullptr;
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

Tensor Tensor::resize(int rowNum, int colNum, bool isNew) {
	return (*shared_ptr<Operator>(new ResizeOperator(*this, rowNum, colNum, isNew)))();
}

Tensor Tensor::transpose(bool isNew) {
	return (*shared_ptr<Operator>(new TransposeOperator(*this, isNew)))();
}

Tensor Tensor::dot(const Tensor &t) {
	return (*shared_ptr<Operator>(new DotOperator(*this, t)))();
}