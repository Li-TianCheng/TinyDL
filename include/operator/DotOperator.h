//
// Created by ltc on 2021/9/8.
//

#ifndef TINYDL_DOTOPERATOR_H
#define TINYDL_DOTOPERATOR_H

#include "operator/Operator.h"

class DotOperator : public Operator {
public:
	DotOperator(const Tensor& tensor1, const Tensor& tensor2);
	Tensor operator()() override;
	void grad(Tensor& result) override;
};


#endif //TINYDL_DOTOPERATOR_H
