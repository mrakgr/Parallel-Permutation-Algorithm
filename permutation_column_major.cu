#include <thrust\device_vector.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace thrust;
using namespace std;

unsigned long factorial(unsigned long n)
{
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

__device__ unsigned long inline get_element(unsigned long *per_m, int row, int col, int per_num){
	if (row < 0) return 0;
	unsigned long pos = row + col*per_num;
	return per_m[pos];
}

__device__ void inline set_element(unsigned long *per_m, int row, int col, int per_num, unsigned long value){
	unsigned long pos = row + col*per_num;
	per_m[pos] = value;
}

__device__ unsigned long inline get_key_of_zero(unsigned long a, unsigned long n)
{
	a = ~a;              // search for 1-bit instead of 0-bit
	for (; n > 0; n--) {
		a = a & (a - 1);   // clear least significant 1-bit 
	}
	return 1UL << (__ffs(a) - 1);
}

void print_matrix_map(device_vector<unsigned long> matrix_map, unsigned long limit, unsigned long factor){
	// Prints only the first 100 rows, so the screen does not overflow.
	for (unsigned long i = 0; i < std::min(limit*factor, limit*100UL); i += limit){
		thrust::copy(matrix_map.begin() + i, matrix_map.begin() + i + limit, ostream_iterator<unsigned long>(cout, " "));
		cout << endl;
	}
}

__global__ void createPermutationMatrix_kernel(unsigned long *per_m, unsigned long factor, unsigned long per_num, int row, unsigned long factor2){
	factor; // The number of elements in a single row.
	factor2; // The variable that controls the variation in position as the program scans along the rows.
	unsigned long factor3 = per_num-row;
	per_m;  // A map of keys to the indices.

	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (col < factor){
		unsigned long m_previous, pos;
		m_previous = get_element(per_m, row - 1, col, per_num); // Get the element of the previous row.
		pos = col / factor2; // In the first row pos will range from (1...limit) in the next it range from (1...limit-1) and contract after that.
		pos = (pos % factor3) + 1;
		unsigned long idx = get_key_of_zero(m_previous, pos); // idx has only a single bit set to 1 at a location differing from that of m.
		unsigned long new_m = m_previous | idx; // That bit gets added here to the value of min the previous row.
		set_element(per_m, row, col, per_num, new_m); // matrix_map(row,column) is set to new_m.
	}
}

// What is happening here is similar to the optimization technique of dynamic programming where
// first a table is computed and then it is backtracked along the rows to compute the actual values.

// After the loop above, the all the elements of the first row should have only one bit set to 1 and all the elements of the last
// should have all the bits (up to the limit) set to 1. The second should have two bits set to one and the third should have
// three bits set to one and so on.

// To compute the key in the latest row that is being computed, you take the value of m in the previous row, take the bitwise
// (not arithmetic) negative of that and then apply m_previous BITWISE-AND m_current to compute the key. Then to get the index
// value from that you take the log_2 or simply the bitscan of that value to obtain the index.

__global__ void computeIndices_kernel(unsigned long *per_m, unsigned long factor, unsigned long per_num, int row){
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (col < factor){
		unsigned long m_previous, m_current;

		m_previous = get_element(per_m, row - 1, col, per_num);
		m_current = get_element(per_m, row, col, per_num);
		unsigned long new_m = (~m_previous) & m_current; // new_m should have only one bit set to 1.
		unsigned long index_of_new_m = __ffs(new_m) - 1;
		set_element(per_m, row, col, per_num, index_of_new_m);
	}
}

void createPermutationMatrix(unsigned long *d_perm, unsigned long perm_num)
{
	const unsigned long block_size = 512;

	unsigned long factor = factorial(perm_num);

	unsigned long gridx = factor / block_size;
	if (factor % block_size != 0) gridx++;

	unsigned long factor2 = factor / perm_num;
	for (int i = 0; i < perm_num; i++){
		createPermutationMatrix_kernel << <gridx, block_size >> >(d_perm, factor, perm_num, i, factor2);
		if (perm_num - i - 1 != 0) {
			factor2 /= (perm_num - i - 1);
		}
		cudaDeviceSynchronize();
	}

	for (int i = perm_num-1; i >= 0; i--){
		computeIndices_kernel << <gridx, block_size >> >(d_perm, factor, perm_num, i);
		cudaDeviceSynchronize();
	}



}

int main(){
	unsigned long num = 10; // Number of permutations.
	device_vector<unsigned long> perm(factorial(num)*num);

	unsigned long *d_perm = raw_pointer_cast(perm.data());

	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; i++)
		createPermutationMatrix(d_perm, num);
	end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::high_resolution_clock::to_time_t(end);

	std::cout << "finished computation at " << std::ctime(&end_time)
		<< "elapsed time: " << elapsed_seconds.count() << "s\n";

	// On my GTX 970 a thousand iterations of generating 10! matrices gives results at 34.7 seconds. For comparison, the STL iterator would 
	// need 1.7 seconds on my OC'd i-4690k to compute a single matrix. The GPU version is roughly 50 times faster at generating permutation 
	// matrices.
	
	// The column major version of the algorithm as it does not have coalesced memory accesses is much slower than the row major version.

	print_matrix_map(perm, num, factorial(num));

	return 0;
}
