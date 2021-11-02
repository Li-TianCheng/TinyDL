//
// Created by ltc on 2021/9/2.
//

#include "operator/PowOperator.h"

PowOperator::PowOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor PowOperator::operator()() {
	shared_ptr<CuMatrix> value = nullptr;
	if (tensor1.isConstant()) {
		value = std::make_shared<CuMatrix>((*tensor2).exp());
	}
	if (tensor2.isConstant()) {
		value = std::make_shared<CuMatrix>((*tensor1).pow((*tensor2)(0, 0)));
	}
	return Tensor(value, shared_from_this());
}

void PowOperator::backward(Tensor& result) {
	if (tensor1.isConstant()) {
		tensor2.grad() += (*tensor2).exp().dot(result.grad());
	}
	if (tensor2.isConstant()) {
		tensor1.grad() += (*tensor1).pow((*tensor2)(0, 0) - 1).dot(result.grad())* (*tensor2)(0, 0);
	}
}
