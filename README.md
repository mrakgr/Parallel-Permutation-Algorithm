# Parallel Permutation Matrix Algorithm C++
C++ parallel permutation matrix generation algorithm.

The CPU version is actually slower that the STL permutation iterator, but the CUDA version that is meant for the ArrayFire library is order or two of magnitude faster depending on the version.

Update (year later): Of some interest might be the algorithm for generating [permutations without repetition](https://github.com/mrakgr/Pathfinding-Experiments/blob/master/Pathfinding%20Experiments/permutation_hashing_encoder.fsx) from a key. It generalizes the permutation algorithm here, and can encode key to a string and decode from a key.

Update (2 days later): Here is the optimized version of the [encoder and the decoder algorithms.](https://github.com/mrakgr/Pathfinding-Experiments/blob/master/Pathfinding%20Experiments/permutation_hashing_encoder_decoder_v2.fsx)
