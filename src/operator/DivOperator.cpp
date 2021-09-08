//
// Created by ltc on 2021/9/2.
//

#include "operator/DivOperator.h"

DivOperator::DivOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DivOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic>> value = nullptr;
	if (tensor1.isConstant) {
		std::make_shared<Matrix<double, Dynamic, Dynamic>>((tensor1.constValue / (*tensor2.value).array()).matrix());
	}
	if (tensor2.isConstant) {
		std::make_shared<Matrix<double, Dynamic, Dynamic>>(*tensor1.value / tensor2.constValue);
	}
	return Tensor(value, shared_from_this());
}

void DivOperator::grad(Tensor& result) {
	if (tensor1.isConstant) {
		*tensor2._grad += ((*tensor2.value).array().pow(-2).matrix() * *result._grad * tensor1.constValue).transpose();
	}
	if (tensor2.isConstant) {
		*tensor1._grad += (*result._grad / tensor2.constValue).transpose();
	}
}
