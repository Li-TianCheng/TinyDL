//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_MAXPOOLOPERATOR_H
#define TINYDL_MAXPOOLOPERATOR_H

#include "operator/Operator.h"

class MaxPoolOperator : public Operator {
public:
	MaxPoolOperator(const Tensor& tensor1, int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride);
	Tensor operator()() override;
	void backward(Tensor& result) override;
private:
	int channel;
	int dataRow;
	int dataCol;
	int kernelRow;
	int kernelCol;
	int stride;
};


#endif //TINYDL_MAXPOOLOPERATOR_H
