//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_DIVOPERATOR_H
#define TINYDL_DIVOPERATOR_H

#include "operator/Operator.h"

class DivOperator : public Operator {
public:
	DivOperator(const Tensor& tensor1, const Tensor& tensor2);
	Tensor operator()() override;
	void grad(Tensor& result) override;
};


#endif //TINYDL_DIVOPERATOR_H
