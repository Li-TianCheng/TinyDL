//
// Created by ltc on 2021/9/2.
//

#include "operator/AddOperator.h"

AddOperator::AddOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor AddOperator::operator()() {
	shared_ptr<CuMatrix> value = nullptr;
	if (tensor1.isConstant()) {
		CuMatrix m(tensor2.row(), tensor2.col(), tensor2.isCuda());
		m.setOnes();
		value = std::make_shared<CuMatrix>(m*(*tensor1)(0, 0)+*tensor2);
	} else if (tensor2.isConstant()) {
		CuMatrix m(tensor1.row(), tensor1.col(), tensor1.isCuda());
		m.setOnes();
		value = std::make_shared<CuMatrix>(*tensor1 + m*(*tensor2)(0, 0));
	} else {
		value = std::make_shared<CuMatrix>(*tensor1+*tensor2);
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
