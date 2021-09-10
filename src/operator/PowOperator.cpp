//
// Created by ltc on 2021/9/2.
//

#include "operator/PowOperator.h"

PowOperator::PowOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor PowOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant()) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor2).array().exp().matrix());
	}
	if (tensor2.isConstant()) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1).array().pow((*tensor2)(0, 0)).matrix());
	}
	return Tensor(value, shared_from_this());
}

void PowOperator::backward(Tensor& result) {
	if (tensor1.isConstant()) {
		tensor2.grad() += ((*tensor2).array().exp() * result.grad().array()).matrix();
	}
	if (tensor2.isConstant()) {
		tensor1.grad() += ((*tensor1).array().pow((*tensor2)(0, 0) - 1) * result.grad().array()).matrix() * (*tensor2)(0, 0);
	}
}
