//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_ADDOPERATOR_H
#define TINYDL_ADDOPERATOR_H

#include "operator/Operator.h"

class AddOperator : public Operator {
public:
	AddOperator(const Tensor& tensor1, const Tensor& tensor2);
	Tensor operator()() override;
	void backward(Tensor& result) override;
};

#endif //TINYDL_ADDOPERATOR_H
