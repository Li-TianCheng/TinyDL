//
// Created by ltc on 2021/9/2.
//

#include "operator/MulOperator.h"

MulOperator::MulOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor MulOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic>> value = nullptr;
	if (tensor1.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic>>(tensor1.constValue**tensor2.value);
	} else if (tensor2.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic>>(*tensor1.value*tensor2.constValue);
	} else {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic>>(*tensor1.value**tensor2.value);
	}
	return Tensor(value, shared_from_this());
}

void MulOperator::grad(Tensor& result) {
	if (!tensor1.isConstant) {
		if (tensor2.isConstant) {
			*tensor1._grad += (tensor2.constValue * *result._grad).transpose();
		} else {
			*tensor1._grad += (*tensor2.value * *result._grad).transpose();
		}
	}
	if (!tensor2.isConstant) {
		if (tensor1.isConstant) {
			*tensor2._grad += (*result._grad * tensor1.constValue).transpose();
		} else {
			*tensor2._grad += (*result._grad * *tensor1.value).transpose();
		}
	}
}
