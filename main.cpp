#include <iostream>
#include "Tensor.h"

using namespace std;

int main() {
	Matrix<double, 1, 2, RowMajor> a;
	Matrix<double, -1, -1, RowMajor> b;
	b.resize(2, 1);
	a << 1,2;
	b << 2,1;
	Tensor t1(a);
	Tensor t2(b);
	Tensor t3 = t1.log(2)*t2.log(2);
//	cout << a << endl;
//	cout << b << endl;
	cout << t3 << endl;
	t3.backward();
	cout << t1.grad() << endl;
	cout << t2.grad() << endl;
    return 0;
}
