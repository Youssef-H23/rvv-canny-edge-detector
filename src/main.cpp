#include <riscv_vector.h>
#include <stdio.h>

int main(){

	//puting data in the device RAM
	int32_t src1[4] = {1, 4, 6, 10};
	int32_t src2[4] = {10, 45, 62, 40};
	int32_t dest[4] = {0, 0, 0, 0};

	//Loading data on vector regester from RAM
	size_t vl = __riscv_vsetvl_e32m1(4);
	printf("%d \n", vl);
	vint32m1_t v1 = __riscv_vle32_v_i32m1(src1, vl);
	vint32m1_t v2 = __riscv_vle32_v_i32m1(src2, vl);

	vint32m1_t res = __riscv_vadd_vv_i32m1(v1, v2, vl);

	__riscv_vse32_v_i32m1(dest, res, vl);
	printf("Result: [%d, %d, %d, %d]\n", dest[0], dest[1], dest[2], dest[3]);
	return 0;
}
