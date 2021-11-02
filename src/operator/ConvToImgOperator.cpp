//
// Created by ltc on 2021/9/13.
//

#include "operator/ConvToImgOperator.h"

ConvToImgOperator::ConvToImgOperator(const Tensor &tensor1, int num) : Operator(tensor1, Tensor(0, tensor1.isCuda())), num(num) {

}

Tensor ConvToImgOperator::operator()() {
	auto value = std::make_shared<CuMatrix>(tensor1.row()/num, tensor1.col()*num, tensor1.isCuda());
	value->setZero();
	if (value->isCuda()) {
		cuda::convToImg(**tensor1, num, **value);
	} else {
#pragma omp parallel for
		for (int i = 0; i < tensor1.row(); ++i) {
			for (int j = 0; j < tensor1.col(); ++j) {
				(**value)(i/num, j*num+i%num) = tensor1(i, j);
			}
		}
	}
	return Tensor(value, shared_from_this());
}

void ConvToImgOperator::backward(Tensor &result) {
	if (tensor1.grad().isCuda()) {
		cuda::convToImgBp(*result.grad(), num, *tensor1.grad());
	} else {
#pragma omp parallel for
		for (int i = 0; i < tensor1.row(); ++i) {
			for (int j = 0; j < tensor1.col(); ++j) {
				(*tensor1.grad())(i, j) += (*result.grad())(i/num, j*num+i%num);
			}
		}
	}
}
