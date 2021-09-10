//
// Created by ltc on 2021/9/2.
//

#include "operator/AddOperator.h"

AddOperator::AddOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor AddOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant()) {
		Matrix<double, Dynamic, Dynamic> m(*tensor2);
		m.setOnes();
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1)(0, 0)*m + *tensor2);
	} else if (tensor2.isConstant()) {
		Matrix<double, Dynamic, Dynamic> m(*tensor1);
		m.setOnes();
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1 + (*tensor2)(0, 0)*m);
	} else {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1+*tensor2);
	}
	return Tensor(value, shared_from_this());
}

void AddOperator::backward(Tensor& result) {
	if (!tensor1.isConstant()) {
		tensor1.grad() += result.grad();
	}
	if (!tensor2.isConstant()) {
		tensor2.grad() += result.grad();
	}
}
