//
// Created by ltc on 2021/9/2.
//

#include "operator/PowOperator.h"

PowOperator::PowOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor PowOperator::operator()() {
	shared_ptr<Matrix<double, Dynamic, Dynamic>> value = nullptr;
	if (tensor1.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic>>((*tensor2.value).array().exp().matrix());
	}
	if (tensor2.isConstant) {
		value = std::make_shared<Matrix<double, Dynamic, Dynamic>>((*tensor1.value).array().pow(tensor2.constValue).matrix());
	}
	return Tensor(value, shared_from_this());
}

void PowOperator::grad(Tensor& result) {
	if (tensor1.isConstant) {
		*tensor2._grad += ((*tensor2.value).array().exp().matrix() * *result._grad).transpose();
	}
	if (tensor2.isConstant) {
		*tensor1._grad += ((*tensor1.value).array().pow(tensor2.constValue-1).matrix() * *result._grad * tensor2.constValue).transpose();
	}
}
