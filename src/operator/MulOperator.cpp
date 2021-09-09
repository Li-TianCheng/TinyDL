//
// Created by ltc on 2021/9/2.
//

#include "operator/MulOperator.h"

MulOperator::MulOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor MulOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic, RowMajor>> value = nullptr;
	if (tensor1.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>((*tensor1.value)(0, 0)**tensor2.value);
	} else if (tensor2.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value*(*tensor2.value)(0, 0));
	} else {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value**tensor2.value);
	}
	return Tensor(value, shared_from_this());
}

void MulOperator::backward(Tensor& result) {
	if (!tensor1.isConstant) {
		if (tensor2.isConstant) {
			*tensor1.gradient += (*tensor2.value)(0, 0) * *result.gradient;
		} else {
			*tensor1.gradient += *result.gradient * (*tensor2.value).transpose();
		}
	}
	if (!tensor2.isConstant) {
		if (tensor1.isConstant) {
			*tensor2.gradient += *result.gradient * (*tensor1.value)(0, 0);
		} else {
			*tensor2.gradient += (*tensor1.value).transpose() * *result.gradient;
		}
	}
}
