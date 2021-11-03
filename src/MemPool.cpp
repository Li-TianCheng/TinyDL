//
// Created by ltc on 2021/11/3.
//

#include "MemPool.h"

memObj* MemPool::allocate(size_t size, bool cuda) {
	if (cuda) {
		if (cudaMem.find(size) == cudaMem.end()) {
			cudaMem[size] = nullptr;
		}
		if (cudaMem[size] == nullptr) {
			memObj* mem = new memObj;
			mem->next = nullptr;
			cudaMalloc((void**)&mem->ptr, size);
			return mem;
		} else {
			memObj* mem = cudaMem[size];
			cudaMem[size] = mem->next;
			mem->next = nullptr;
			return mem;
		}
	} else {
		if (cpuMem.find(size) == cpuMem.end()) {
			cpuMem[size] = nullptr;
		}
		if (cpuMem[size] == nullptr) {
			memObj* mem = new memObj;
			mem->next = nullptr;
			mem->ptr = malloc(size);
			return mem;
		} else {
			memObj* mem = cpuMem[size];
			cpuMem[size] = mem->next;
			mem->next = nullptr;
			return mem;
		}
	}
}

void MemPool::free(memObj *ptr, size_t size, bool cuda) {
	if (cuda) {
		ptr->next = cudaMem[size];
		cudaMem[size] = ptr;
	} else {
		ptr->next = cpuMem[size];
		cpuMem[size] = ptr;
	}
}

MemPool::~MemPool() {
	for (auto& m : cudaMem) {
		memObj* free = m.second;
		while (free != nullptr) {
			memObj* tmp = free->next;
			cudaFree(free->ptr);
			delete free;
			free = tmp;
		}
	}
	for (auto& m : cpuMem) {
		memObj* free = m.second;
		while (free != nullptr) {
			memObj* tmp = free->next;
			::free(free->ptr);
			delete free;
			free = tmp;
		}
	}
}
