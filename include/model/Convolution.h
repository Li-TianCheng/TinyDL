//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_CONVOLUTION_H
#define TINYDL_CONVOLUTION_H

#include "Tensor.h"
#include "model/Model.h"
#include "model/Linear.h"
#include "operator/ImgToConvOperator.h"
#include "operator/ConvToImgOperator.h"

class Convolution {
public:
	Convolution(Model& m, int inputChannel, int outputChannel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride=1, bool bias=true);
	Tensor operator()(const Tensor& input);
private:
	int inputChannel;
	int dataRow;
	int dataCol;
	int kernelRow;
	int kernelCol;
	int stride;
	Linear linear;
};


#endif //TINYDL_CONVOLUTION_H
