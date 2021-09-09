//
// Created by ltc on 2021/9/2.
//

#include "operator/AddOperator.h"

AddOperator::AddOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor AddOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant) {
		Matrix<double, Dynamic, Dynamic> m(*tensor2.value);
		m.setOnes();
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1.value)(0, 0)*m + *tensor2.value);
	} else if (tensor2.isConstant) {
		Matrix<double, Dynamic, Dynamic> m(*tensor1.value);
		m.setOnes();
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value + (*tensor2.value)(0, 0)*m);
	} else {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value+*tensor2.value);
	}
	return Tensor(value, shared_from_this());
}

void AddOperator::backward(Tensor& result) {
	if (!tensor1.isConstant) {
		*tensor1.gradient += *result.gradient;
	}
	if (!tensor2.isConstant) {
		*tensor2.gradient += *result.gradient;
	}
}
