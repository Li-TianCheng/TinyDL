//
// Created by ltc on 2021/11/1.
//

#ifndef TINYDL_API_H
#define TINYDL_API_H

#include "kernel.cuh"
namespace cuda {
	void add(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void sub(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void mul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void numMul(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	            double num,
	            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void dot(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void log(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void maxPool(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	             int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
	             Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void maxPoolBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
				   int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m3,
				   Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void exp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void pow(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	         double num,
	         Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void convToImg(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               int num,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void convToImgBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               int num,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void imgToConv(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void imgToConvBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	void setValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	              double num);

	double getValue(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
					int row, int col);

	void info(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

	void transpose(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	               Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);

	double max(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

	double min(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

	void relu(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1);

	void reluBp(Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m1,
	            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m2,
	            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> r);
}

#endif //TINYDL_API_H
