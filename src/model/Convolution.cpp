//
// Created by ltc on 2021/9/13.
//

#include "model/Convolution.h"

Convolution::Convolution(Model &m, int inputChannel, int outputChannel, int dataRow, int dataCol, int kernelRow, int kernelCol,
                         int stride, bool bias) : linear(m, inputChannel*kernelRow*kernelCol, outputChannel, bias), inputChannel(inputChannel),
                         dataRow(dataRow), dataCol(dataCol), kernelRow(kernelRow), kernelCol(kernelCol), stride(stride){

}

Tensor Convolution::operator()(const Tensor &input) {
	Tensor output = (*shared_ptr<Operator>(new ImgToConvOperator(input, inputChannel, dataRow, dataCol, kernelRow, kernelCol, stride)))();
	output = linear(output);
	output = (*shared_ptr<Operator>(new ConvToImgOperator(output, ((dataRow-kernelRow)/stride+1)*((dataCol-kernelCol)/stride+1))))();
	return output;
}
