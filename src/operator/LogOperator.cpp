//
// Created by ltc on 2021/9/2.
//

#include "operator/LogOperator.h"

LogOperator::LogOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor LogOperator::operator()() {
	auto value = std::make_shared<CuMatrix>((*tensor2).log() / std::log((*tensor1)(0, 0)));
	return Tensor(value, shared_from_this());
}

void LogOperator::backward(Tensor& result) {
	tensor2.grad() += result.grad().dot((*tensor2).pow(-1)) / std::log((*tensor1)(0, 0));
}
