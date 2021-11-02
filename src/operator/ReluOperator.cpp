//
// Created by ltc on 2021/11/2.
//

#include "operator/ReluOperator.h"

ReluOperator::ReluOperator(const Tensor &tensor1) : Operator(tensor1, Tensor(0, tensor1.isCuda())) {

}

Tensor ReluOperator::operator()() {
	auto value = std::make_shared<CuMatrix>(*tensor1);
	if (value->isCuda()) {
		cuda::relu(**value);
	} else {
#pragma omp parallel
		for (int i = 0; i < tensor1.row(); ++i) {
			for (int j = 0; j < tensor1.col(); ++j) {
				if ((*value)(i, j) < 0) {
					(*value).setValue(i, j, 0);
				}
			}
		}
	}
	return Tensor(value, shared_from_this());
}

void ReluOperator::backward(Tensor &result) {
	if (tensor1.isCuda()) {
		cuda::reluBp(**result, *result.grad(), *tensor1.grad());
	} else {
#pragma omp parallel
		for (int i = 0; i < tensor1.row(); ++i) {
			for (int j = 0; j < tensor1.col(); ++j) {
				if ((*result)(i, j) != 0) {
					(*tensor1.grad())(i, j) += (*result.grad())(i, j);
				}
			}
		}
	}
}
