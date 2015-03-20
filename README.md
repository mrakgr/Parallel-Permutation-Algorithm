# Parallel-Permutation-Algorithm C++
C++ parallel matrix permutation algorithm for the Traveling Salesman Problem. Non-parallel CPU version only for now.

It is actually slower that the STL permutation iterator, but this example should scale much better on the GPU than the recursive versions for the CPU as it computes the table and then the actual value on the backwards pass.
