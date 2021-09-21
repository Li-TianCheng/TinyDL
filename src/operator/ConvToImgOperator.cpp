//
// Created by ltc on 2021/9/13.
//

#include "operator/ConvToImgOperator.h"

ConvToImgOperator::ConvToImgOperator(const Tensor &tensor1, int num) : Operator(tensor1, 0), num(num) {

}

Tensor ConvToImgOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>();
	value->resize(tensor1.row()/num, tensor1.col()*num);
	value->setZero();
#pragma omp parallel for
	for (int i = 0; i < tensor1.row(); ++i) {
		for (int j = 0; j < tensor1.col(); ++j) {
			(*value)(i/num, j*num+i%num) = tensor1(i, j);
		}
	}
	return Tensor(value, shared_from_this());
}

void ConvToImgOperator::backward(Tensor &result) {
#pragma omp parallel for
	for (int i = 0; i < tensor1.row(); ++i) {
		for (int j = 0; j < tensor1.col(); ++j) {
			tensor1.grad()(i, j) += result.grad()(i/num, j*num+i%num);
		}
	}
}
