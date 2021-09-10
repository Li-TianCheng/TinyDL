//
// Created by ltc on 2021/9/2.
//

#include "operator/MulOperator.h"

MulOperator::MulOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor MulOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant()) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1)(0, 0)**tensor2);
	} else if (tensor2.isConstant()) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1*(*tensor2)(0, 0));
	} else {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1**tensor2);
	}
	return Tensor(value, shared_from_this());
}

void MulOperator::backward(Tensor& result) {
	if (!tensor1.isConstant()) {
		if (tensor2.isConstant()) {
			tensor1.grad() += (*tensor2)(0, 0) * result.grad();
		} else {
			tensor1.grad() += result.grad() * (*tensor2).transpose();
		}
	}
	if (!tensor2.isConstant()) {
		if (tensor1.isConstant()) {
			tensor2.grad() += result.grad() * (*tensor1)(0, 0);
		} else {
			tensor2.grad() += (*tensor1).transpose() * result.grad();
		}
	}
}
