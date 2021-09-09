//
// Created by ltc on 2021/9/2.
//

#include "operator/PowOperator.h"

PowOperator::PowOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor PowOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor2.value).array().exp().matrix());
	}
	if (tensor2.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1.value).array().pow((*tensor2.value)(0, 0)).matrix());
	}
	return Tensor(value, shared_from_this());
}

void PowOperator::backward(Tensor& result) {
	if (tensor1.isConstant) {
		*tensor2.gradient += ((*tensor2.value).array().exp() * (*result.gradient).array()).matrix();
	}
	if (tensor2.isConstant) {
		*tensor1.gradient += ((*tensor1.value).array().pow((*tensor2.value)(0, 0) - 1) * (*result.gradient).array()).matrix() * (*tensor2.value)(0, 0);
	}
}
