//
// Created by ltc on 2021/9/10.
//

#include "operator/ReluOperator.h"

ReluOperator::ReluOperator(const Tensor &tensor1) : Operator(tensor1, 0) {

}

Tensor ReluOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1);
	for (int i = 0; i < value->rows(); ++i) {
		for (int j = 0; j < value->cols(); ++j) {
			if ((*value)(i, j) < 0) {
				(*value)(i, j) = 0;
			}
		}
	}
	return Tensor(value, shared_from_this());
}

void ReluOperator::backward(Tensor &result) {
	auto value = *tensor1;
	for (int i = 0; i < value.rows(); ++i) {
		for (int j = 0; j < value.cols(); ++j) {
			if (value(i, j) > 0) {
				tensor1.grad()(i, j) += result.grad()(i, j);
			}
		}
	}
}
