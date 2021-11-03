//
// Created by ltc on 2021/11/3.
//

#ifndef TINYDL_MEMPOOL_H
#define TINYDL_MEMPOOL_H

#include <cuda_runtime.h>
#include <unordered_map>

using std::unordered_map;

struct memObj {
	memObj* next;
	void* ptr;
};

class MemPool {
public:
	memObj* allocate(size_t size, bool cuda);
	void free(memObj* ptr, size_t size, bool cuda);
	~MemPool();
private:
	unordered_map<size_t, memObj*> cpuMem;
	unordered_map<size_t, memObj*> cudaMem;
};

static MemPool memPool;

#endif //TINYDL_MEMPOOL_H
