//
// Created by ltc on 2021/9/2.
//

#include "operator/DivOperator.h"

DivOperator::DivOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DivOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((tensor1.constValue / (*tensor2.value).array()).matrix());
	}
	if (tensor2.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value / tensor2.constValue);
	}
	return Tensor(value, shared_from_this());
}

void DivOperator::backward(Tensor& result) {
	if (tensor1.isConstant) {
		*tensor2.gradient += -((*tensor2.value).array().pow(-2) * (*result.gradient).array()).matrix() * tensor1.constValue;
	}
	if (tensor2.isConstant) {
		*tensor1.gradient += *result.gradient / tensor2.constValue;
	}
}
