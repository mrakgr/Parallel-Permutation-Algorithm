unsigned long inline get_element(vector<unsigned long> &map, int row, int col, int factor){
	if (row < 0) return 0;
	unsigned long pos = factor*row + col;
	return map[pos];
}

void inline set_element(vector<unsigned long> &map, int row, int col, int factor, unsigned long value){
	unsigned long pos = factor*row + col;
	map[pos] = value;
}

unsigned long inline get_key_of_zero(unsigned long a, unsigned long n)
{
	a = ~a;              // search for 1-bit instead of 0-bit
	for (; n > 0; n--) {
		a = a & (a - 1);   // clear least significant 1-bit 
	}
	_BitScanForward(&a, a);
	return 1UL << a;
}


void print_matrix_map(vector<unsigned long> matrix_map, unsigned long limit, unsigned long factor){
	for (unsigned long i = 0; i <= (limit-1)*factor; i+=factor){
		copy(matrix_map.begin() + i, matrix_map.begin() + factor + i, ostream_iterator<unsigned long>(cout, " "));
		cout << endl;
	}
}

void create_permutation_matrix_parallel_cpu(array &p_matrix, float *p_host, unsigned long const limit){
	unsigned long const factor = factorial(limit); // The number of elements in a single row.
	unsigned long factor2 = factor/limit; // The variable that controls the variation in position as the program scans along the rows.
	unsigned long factor3 = limit;
	vector<unsigned long> matrix_map(factor*limit); // A map of keys to the indices.
	for (int i = 0; i < limit; i++){
		unsigned long m_previous, pos;
		for (unsigned long j = 0; j < factor; j++){
			m_previous = get_element(matrix_map, i - 1, j, factor); // Get the element of the previous row.
			pos = j / factor2; // In the first row pos will range from (1...limit) in the next it range from (1...limit-1) and contract after that.
			pos = (pos % factor3)+1;
			unsigned long idx = get_key_of_zero(m_previous, pos); // idx has only a single bit set to 1 at a location differing from that of m.
			unsigned long new_m = m_previous | idx; // That bit gets added here to the value of min the previous row.
			set_element(matrix_map, i, j, factor, new_m); // matrix_map(row,column) is set to new_m.
		}

		factor3--;
		if (limit - i - 1 != 0) {
			factor2 /= (limit - i - 1);
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

	for (int i = limit - 1; i >= 0; i--){
		unsigned long m_previous, m_current;
		for (unsigned long j = 0; j < factor; j++){
			m_previous = get_element(matrix_map, i - 1, j, factor);
			m_current = get_element(matrix_map, i, j, factor);
			unsigned long new_m = (~m_previous) & m_current; // new_m should have only one bit set to 1.
			unsigned long index_of_new_m;
			_BitScanForward(&index_of_new_m, new_m);
			set_element(matrix_map, i, j, factor, index_of_new_m);
		}
	}

	//p_matrix = array(limit, factor, matrix_map.data());

	//print_matrix_map(matrix_map, limit, factor);

	//cout << matrix_map.size() << endl;
	
}
