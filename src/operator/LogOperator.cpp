//
// Created by ltc on 2021/9/2.
//

#include "operator/LogOperator.h"

LogOperator::LogOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor LogOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(((*tensor2.value).array().log() / std::log(tensor1.constValue)).matrix());
	return Tensor(value, shared_from_this());
}

void LogOperator::grad(Tensor& result) {
	*tensor2.gradient += ((1 / (*tensor2.value).array()) * (*result.gradient).array()).matrix() / std::log(tensor1.constValue);
}
