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

Tensor::Tensor(double value, bool cuda) : op(nullptr), value(nullptr), gradient(nullptr), constant(true) {
	this->value = std::make_shared<CuMatrix>(1, 1, cuda);
	this->value->setValue(0, 0, value);
	gradient = std::make_shared<CuMatrix>(1, 1, cuda);
	gradient->setZero();
}

Tensor::Tensor(int rowNum, int colNum, bool cuda) : op(nullptr), constant(false) {
	value = std::make_shared<CuMatrix>(rowNum, colNum, cuda);
	value->setZero();
	gradient = std::make_shared<CuMatrix>(rowNum, colNum, cuda);
	gradient->setZero();
}

Tensor::Tensor(const vector<vector<double>>& v, bool cuda) : op(nullptr), constant(false) {
	value = std::make_shared<CuMatrix>(v.size(), v[0].size(), cuda);
	value->setZero();
#pragma omp parallel for
	for (int i = 0; i < v.size(); ++i) {
		for (int j = 0; j < v[0].size(); ++j) {
			value->setValue(i, j, v[i][j]);
		}
	}
	gradient = std::make_shared<CuMatrix>(v.size(), v[0].size(), cuda);
	gradient->setZero();
}

Tensor::Tensor(const CuMatrix& value) : op(nullptr), constant(false) {
	this->value = std::make_shared<CuMatrix>(value);
	gradient = std::make_shared<CuMatrix>((*value).rows(), (*value).cols(), value.isCuda());
	gradient->setZero();
}

Tensor::Tensor(shared_ptr<CuMatrix> value, shared_ptr<Operator> op) : value(value), op(op), constant(false) {
	gradient = std::make_shared<CuMatrix>((**value).rows(), (**value).cols(), value->isCuda());
	gradient->setZero();
}

double Tensor::operator()(int row, int col) const {
	return (*value)(row, col);
}

int Tensor::row() const {
	return (**value).rows();
}

int Tensor::col() const {
	return (**value).cols();
}

bool Tensor::isConstant() const {
	return constant;
}

void Tensor::clearGradient() {
	gradient->setZero();
}

void Tensor::freeOperator() {
	op = nullptr;
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
	*value /= sqrt((**value).cols());
}

Tensor Tensor::copy() const{
	return Tensor(*value);
}

CuMatrix& Tensor::operator*() const {
	return *value;
}

CuMatrix& Tensor::grad() {
	return *gradient;
}

void Tensor::backward() {
	if (!constant) {
		gradient->resize((**value).rows(), (**value).cols());
		gradient->setOnes();
		_backward();
	}
}

void Tensor::_backward() {
	if (op != nullptr && op.use_count() == 1) {
		op->backward(*this);
		op->tensor1._backward();
		op->tensor2._backward();
	}
	op = nullptr;
}

Tensor Tensor::operator+(const Tensor &t) const {
	return (*shared_ptr<Operator>(new AddOperator(*this, t)))();
}

Tensor &Tensor::operator+=(const Tensor &t) {
	*this = *this + t;
	return *this;
}

Tensor &Tensor::operator++() {
	*this = *this + Tensor(1, isCuda());
	return *this;
}

const Tensor Tensor::operator++(int) {
	Tensor tmp = *this;
	++*this;
	return tmp;
}

Tensor Tensor::operator-(const Tensor &t) const {
	return (*shared_ptr<Operator>(new SubOperator(*this, t)))();
}

Tensor &Tensor::operator-=(const Tensor &t) {
	*this = *this - t;
	return *this;
}

Tensor &Tensor::operator--() {
	*this = *this - Tensor(1, isCuda());
	return *this;
}

const Tensor Tensor::operator--(int) {
	Tensor tmp = *this;
	--*this;
	return tmp;
}

Tensor Tensor::operator*(const Tensor &t) const {
	return (*shared_ptr<Operator>(new MulOperator(*this, t)))();
}

Tensor &Tensor::operator*=(const Tensor &t) {
	*this = *this * t;
	return *this;
}

Tensor Tensor::operator/(const Tensor &t) const {
	return (*shared_ptr<Operator>(new DivOperator(*this, t)))();
}

Tensor &Tensor::operator/=(const Tensor &t) {
	*this = *this / t;
	return *this;
}

Tensor Tensor::log(double t) const {
	return (*shared_ptr<Operator>(new LogOperator(Tensor(t, isCuda()), *this)))();
}

Tensor Tensor::pow(double t) const {
	return (*shared_ptr<Operator>(new PowOperator(*this, Tensor(t, isCuda()))))();
}

Tensor Tensor::exp() const{
	return (*shared_ptr<Operator>(new PowOperator(Tensor(1.0, isCuda()), *this)))();
}

Tensor Tensor::resize(int rowNum, int colNum, bool isNew) const {
	return (*shared_ptr<Operator>(new ResizeOperator(*this, rowNum, colNum, isNew)))();
}

Tensor Tensor::transpose(bool isNew) const {
	return (*shared_ptr<Operator>(new TransposeOperator(*this, isNew)))();
}

Tensor Tensor::dot(const Tensor &t) const {
	return (*shared_ptr<Operator>(new DotOperator(*this, t)))();
}

void Tensor::info() const {
	value->info();
}

void Tensor::cpu() {
	value->cpu();
	gradient->cpu();
}

void Tensor::cuda() {
	value->cuda();
	gradient->cuda();
}

bool Tensor::isCuda() const {
	return value->isCuda();
}
