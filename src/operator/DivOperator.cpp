//
// Created by ltc on 2021/9/2.
//

#include "operator/DivOperator.h"

DivOperator::DivOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DivOperator::operator()() {
	shared_ptr<CuMatrix> value = nullptr;
	if (tensor1.isConstant()) {
		value = std::make_shared<CuMatrix>(((*tensor2)/(*tensor1)(0, 0)).pow(-1));
	}
	if (tensor2.isConstant()) {
		value = std::make_shared<CuMatrix>(*tensor1 / (*tensor2)(0, 0));
	}
	return Tensor(value, shared_from_this());
}

void DivOperator::backward(Tensor& result) {
	if (tensor1.isConstant()) {
		tensor2.grad() += (*tensor2).pow(-2) * result.grad()* (-(*tensor1)(0, 0));
	}
	if (tensor2.isConstant()) {
		tensor1.grad() += result.grad() / (*tensor2)(0, 0);
	}
}
