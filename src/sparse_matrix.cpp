#include "sparse_matrix.h"

// CONSTRUCTORS

template <typename T> SparseMatrix<T>::SparseMatrix(std::string filename) {
  // Constructor from file
  load_from_file(filename);
}

template <typename T>
void SparseMatrix<T>::load_from_file(std::string filename) {
  start_timer(load_from_file, SparseMatrix);
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  // Open the file descriptor
  if ((f = fopen(filename.c_str(), "r")) == NULL) {
    std::cerr << "Failed to open matrix file " << filename << ENDL;
    exit(-1);
  }
  // Read in the banner
  if (mm_read_banner(f, &matcode) != 0) {
    std::cerr << "Could not read matrix market banner" << ENDL;
    exit(-1);
  }
  std::cerr << "Matcode: " << matcode << ENDL;
  // Check the banner properties
  if (mm_is_matrix(matcode) && mm_is_coordinate(matcode) &&
      (mm_is_real(matcode) || mm_is_integer(matcode) ||
       mm_is_pattern(matcode))) {
    // TODO: Need to use float/integer conversions
    // TODO: Need to check if the matrix is general/symmetric etc
    // do matrix reading
    // Find size of matrix
    if ((ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nonz)) != 0) {
      std::cerr << "Cannot read matrix sizes and number of non-zeros" << ENDL;
      return;
    }
    std::cerr << "Rows " << rows << " cols " << cols << " non-zeros " << nonz
              << ENDL;
    // read the entries from the file
    int I, J;
    double val;
    int pat = mm_is_pattern(matcode);
    // reserve nonz entries for nz_entries to size nonz so that it's faster to
    // call push_back on
    nz_entries.reserve(mm_is_symmetric(matcode) ? nonz : nonz * 2);
    for (int i = 0; i < nonz; i++) {
      if (pat) {
        fscanf(f, "%d %d\n", &I, &J);
        val = 1.0;
      } else {
        fscanf(f, "%d %d %lg\n", &I, &J, &val);
      }
      // adjust from 1 based to 0 based
      I--;
      J--;
      nz_entries.push_back(std::make_tuple(I, J, static_cast<T>(val)));
      if (mm_is_symmetric(matcode) && I != J) {
        nz_entries.push_back(std::make_tuple(J, I, static_cast<T>(val)));
      }
    }
    nz_entries.shrink_to_fit();
  } else {
    std::cerr << "Cannot process this matrix type. Typecode: " << matcode
              << ENDL;
    exit(-1);
  }
}

template <typename T> void SparseMatrix<T>::calculate_ellpack() {
  if (ellpack_calculated) {
    return;
  } else {
    ellpack_calculated = true;
  }
  start_timer(calculate_ellpack, sparse_matrix);
  // start off by doing a histogram sum of the values in the sparse matrix,
  // and (simultaneously) calculate the maximum row length (we might use this
  // when we're padding the width later)
  // std::vector<unsigned int> row_lengths(height(), 0);
  row_lengths.resize(height(), 0);
  // unsigned int max_width = 0;
  for (unsigned int i = 0; i < nz_entries.size(); i++) {
    int y = std::get<1>(nz_entries[i]);
    row_lengths[y]++;
    if (row_lengths[y] > max_width) {
      max_width = row_lengths[y];
    }
  }
  LOG_DEBUG("max width: ", max_width);

  // based on that, create a "standard" AOS ragged std::vector based structure
  // to fill with values from the sparse matrix. Reserve the rows to the lengths
  // we've just calculated with the histogram
  // SparseMatrix::ellpack_matrix<T> ellpackMatrix(height(), ellpack_row<T>(0));
  ellpackMatrix.resize(height(), ellpack_row<T>(0));
  for (int i = 0; i < height(); i++) {
    ellpackMatrix[i].reserve(row_lengths[i]);
  }

  // fill the ellpackMatrix using standard push_back etc from the nz_entries
  for (unsigned int i = 0; i < nz_entries.size(); i++) {
    int x = std::get<0>(nz_entries[i]);
    int y = std::get<1>(nz_entries[i]);
    int val = std::get<2>(nz_entries[i]);
    std::pair<int, T> r_entry(x, val);
    ellpackMatrix[y].push_back(r_entry);
  }

  // sort each row by the x values
  for (auto row : ellpackMatrix) {
    std::sort(row.begin(), row.end(),
              [](std::pair<int, T> a, std::pair<int, T> b) {
                return a.first < b.first;
              });
  }
}

template <typename T> void SparseMatrix<T>::calculate_transposed_sum() {
  start_timer(calculate_transposed_sum, sparse_matrix);
  // make a container for the sums
  column_sums.resize(width(), 0);
  for (unsigned int i = 0; i < nz_entries.size(); i++) {
    int x = std::get<0>(nz_entries[i]);
    column_sums[x] = column_sums[x] + std::get<2>(nz_entries[i]);
  }
}

template <typename T>
CL_matrix SparseMatrix<T>::cl_encode(unsigned int device_max_alloc_bytes,
                                     T zero, bool pad_height, bool pad_width,
                                     bool rsa, int height_pad_modulo,
                                     int width_pad_modulo) {
  start_timer(cl_encode, sparse_matrix);
  // =========================================================================
  // STEP ONE: CREATE AN ELLPACK MATRIX (AS SIMPLE AS POSSIBLE), WHICH
  //           WE CAN ANALYSE TO ACTUALLY BUILD THE ENCODED ARGUMENTS
  // =========================================================================
  calculate_ellpack();

  // typedef some useful sizes
  typedef unsigned long byte_size;
  typedef unsigned int value_size;

  // =========================================================================
  // STEP TWO: CALCULATE THE SIZE OF OUR FINAL ENCODED MATRIX - THIS IS OUR
  //           CHANCE TO BAIL OUT OF WE'RE GOING TO ALLOCATE TOO MUCH MEMORY
  // =========================================================================
  // -------------------------------------------------------------------------
  // Step 2.1: Vertical padding, for "chunking" optimisations
  // -------------------------------------------------------------------------
  // calculate the actual height to start with
  int concrete_height = height();
  // extend the concrete_lengths array if we're padding vertically
  if (pad_height) {
    concrete_height = concrete_height + (height_pad_modulo -
                                         (concrete_height % height_pad_modulo));
  }
  // and if we're RSA, add a row for the offsets
  if (rsa) {
    concrete_height = concrete_height + 1;
  }
  LOG_DEBUG("concrete height: ", concrete_height);
  // now we have the height, create an array for the actual lengths, and copy
  // the row lengths - copy them along one later if we're rsa
  std::vector<value_size> concrete_lengths(concrete_height, 0);
  std::copy(row_lengths.begin(), row_lengths.end(),
            concrete_lengths.begin() + (rsa ? 1 : 0));

  // -------------------------------------------------------------------------
  // Step 2.2: Horizontal padding, for "splitting" optimisations
  // -------------------------------------------------------------------------
  // next, pad the sizes horizontally, with behavior dependent on whether
  // we're in RSA or not
  int regular_width = max_width;
  if (rsa) {
    // we just need to pad each width to the modulo
    // TODO: IMPLEMENT WHEN WE NEED TO!
  } else {
    if (pad_width) {
      regular_width =
          max_width + (width_pad_modulo - (max_width % width_pad_modulo));
      LOG_DEBUG("Padding to regular width: ", regular_width, " mod modulo: ",
                regular_width % width_pad_modulo);
    }
    std::fill(concrete_lengths.begin(), concrete_lengths.end(), regular_width);
  }
  LOG_DEBUG("regular width: ", regular_width);

  // -------------------------------------------------------------------------
  // Step 2.3: Header information, for "RSA" implementations
  // -------------------------------------------------------------------------
  // first, transform the concrete lengths into two lengths arrays that encode
  // the concrete lengths in terms of the number of bytes, if we're generating
  // for a runtime size array, also add on space for the lengths
  std::vector<value_size> byte_lengths_indices(concrete_lengths);
  std::vector<value_size> byte_lengths_values(concrete_lengths);
  std::transform(byte_lengths_indices.begin(), byte_lengths_indices.end(),
                 byte_lengths_indices.begin(),
                 [rsa](value_size l) -> value_size {
                   return (l * sizeof(int)) + (rsa ? 2 * sizeof(int) : 0);
                 });
  std::transform(byte_lengths_values.begin(), byte_lengths_values.end(),
                 byte_lengths_values.begin(),
                 [rsa](value_size l) -> value_size {
                   return (l * sizeof(T)) + (rsa ? 2 * sizeof(int) : 0);
                 });
  // set the offset sizes if we're RSA, and append size for the offsets
  // essentially, correct for whatever we just did :P
  if (rsa) {
    int offset_array_size = (concrete_height - 1) * sizeof(int);
    byte_lengths_indices[0] = offset_array_size;
    byte_lengths_values[0] = offset_array_size;
  }
  // -------------------------------------------------------------------------
  // Step 2.4: Calculate the overall sizes.
  // -------------------------------------------------------------------------
  // the final sizes we'll need.
  // we can also set the matrix sizes here, as we know them!
  int cl_height = -1;
  int cl_width = -1;
  if (rsa) {
    cl_height = concrete_height - 1;
  } else {
    cl_height = concrete_height;
    cl_width = regular_width;
  }

  byte_size ixs_arr_size = std::accumulate(
      byte_lengths_indices.begin(), byte_lengths_indices.end(), (byte_size)0);
  byte_size vals_arr_size = std::accumulate(
      byte_lengths_values.begin(), byte_lengths_values.end(), (byte_size)0);

  LOG_DEBUG("ixs_arr_size: (GB) - ",
            (double)ixs_arr_size / (double)(1024 * 1024 * 1024));
  LOG_DEBUG("ixs_arr_size: (GB) - ",
            (double)vals_arr_size / (double)(1024 * 1024 * 1024));

  if (ixs_arr_size > device_max_alloc_bytes) {
    throw ixs_arr_size;
  }

  if (!rsa && !pad_height &&
      ((regular_width * concrete_height * sizeof(int)) != ixs_arr_size)) {
    LOG_ERROR("Something has gone catastrophically wrong building the regular "
              "size! Expected array size to be ",
              (regular_width * concrete_height * sizeof(int)), " is actually ",
              ixs_arr_size);
    throw ixs_arr_size;
  }
  // =========================================================================
  // STEP THREE: CREATE THE TWO ARRAYS, AND FILL WITH MATRIX INFORMATION
  // =========================================================================
  // Create the matrix structure that we're going to fill with data
  CL_matrix matrix(ixs_arr_size, vals_arr_size, cl_width, cl_height);

  // fill the matrix with garbage
  std::fill(matrix.indices.begin(), matrix.indices.end(), 0);
  std::fill(matrix.values.begin(), matrix.values.end(), 0);

  // -------------------------------------------------------------------------
  // Step 3.1 build offset information and a lambda to quickly access it
  //          for each array
  // -------------------------------------------------------------------------
  // Perform a scan/inclusive scan over each of the data arrays
  // to figure out the offsets. This shouldn't change whether we're in rsa
  // or not - as it's a concrete piece of information.
  std::vector<byte_size> indices_offsets(concrete_height, 0);
  std::vector<byte_size> values_offsets(concrete_height, 0);
  std::partial_sum(byte_lengths_indices.begin(), byte_lengths_indices.end() - 1,
                   indices_offsets.begin() + 1);
  std::partial_sum(byte_lengths_values.begin(), byte_lengths_values.end() - 1,
                   values_offsets.begin() + 1);

  // std::cout << "indices offsets: [";
  // for (auto i : indices_offsets) {
  //   std::cout << i << ",";
  // }
  // std::cout << "]\n";

  // std::cout << "value offsets: [";
  // for (auto i : values_offsets) {
  //   std::cout << i << ",";
  // }
  // std::cout << "]\n";

  if (vals_arr_size % sizeof(T) != 0) {
    LOG_DEBUG("Potential alignment issue writing to vals buffer!");
  } else {
    LOG_DEBUG("Vals arr size ", vals_arr_size, " aligns with sizeof(T), ",
              sizeof(T));
  }

  // -------------------------------------------------------------------------
  // Step 3.1 use the above information to actually input data into the array!
  // -------------------------------------------------------------------------
  // if we're RSA, write the offset information, and the row sizes + capacities
  if (rsa) {
    LOG_DEBUG("Writing RSA offsets and lengths");
    // take a pointer to each array,as an integer
    int *ixptr = reinterpret_cast<int *>(matrix.indices.data());
    int *valptr = reinterpret_cast<int *>(matrix.values.data());
    for (int i = 0; i < concrete_height - 1; i++) {
      // write an offset to ixptr and valptr to the header
      ixptr[i] = static_cast<int>(indices_offsets[i + 1]);
      valptr[i] = static_cast<int>(values_offsets[i + 1]);
      // write size and capacity information to the headers
      {
        // write the index
        byte_size row_offset = indices_offsets[i + 1];
        // std::cout << "@ " << row_offset << "\n";
        int *cixptr =
            reinterpret_cast<int *>(matrix.indices.data() + row_offset);
        int row_length =
            (byte_lengths_indices[i + 1] - (2 * sizeof(int))) / sizeof(int);
        cixptr[0] = row_length;
        cixptr[1] = row_length;
      }
      {
        // write the value
        byte_size row_offset = values_offsets[i + 1];
        // std::cout << "@ " << row_offset << "\n";
        int *cvalptr =
            reinterpret_cast<int *>(matrix.values.data() + row_offset);
        int row_length =
            (byte_lengths_values[i + 1] - (2 * sizeof(int))) / sizeof(int);
        cvalptr[0] = row_length;
        cvalptr[1] = row_length;
      }
      // write_ix(0, i + 1, byte_lengths_indices[i + 1]);
      // write_ix(1, i + 1, byte_lengths_indices[i + 1]);
      // // todo - replace with proper solution!
      // write_val(0, i + 1, byte_lengths_values[i + 1]);
      // write_val(1, i + 1, byte_lengths_values[i + 1]);
    }

  } else {
    LOG_DEBUG("Filling with -1 and zero");
    // if not, fill the ixs array with -1, and the vals array with 0
    int *ixptr = reinterpret_cast<int *>(matrix.indices.data());
    T *valptr = reinterpret_cast<T *>(matrix.values.data());
    for (byte_size i = 0; i < ixs_arr_size / sizeof(int); ++i) {
      *(ixptr + i) = -1;
    }
    for (byte_size i = 0; i < vals_arr_size / sizeof(T); ++i) {
      *(valptr + i) = zero;
    }
  }
  // actually write the matrix data (finally!)
  LOG_DEBUG("Writing array values");
  bool ixs_out_of_bounds = false;
  bool vals_out_of_bounds = false;

  // iterate over rows
  for (value_size y = 0; y < ellpackMatrix.size(); y++) {
    // get the row, and iterate over the values
    std::vector<std::pair<int, T>> &row = (ellpackMatrix[y]);
    for (value_size i = 0; i < row.size(); i++) {
      std::pair<int, T> t = row[i];
      {
        // write the index
        byte_size row_offset = indices_offsets[rsa ? y + 1 : y];
        byte_size column_offset =
            (sizeof(int) * i) + (rsa ? sizeof(int) * 2 : 0);
        byte_size offset = row_offset + column_offset;

        char *cixptr = matrix.indices.data() + offset;
        *reinterpret_cast<int *>(cixptr) = t.first;
      }
      {
        // start off with our offset at zero
        byte_size row_offset = values_offsets[rsa ? y + 1 : y];
        byte_size column_offset = (sizeof(T) * i) + (rsa ? sizeof(int) * 2 : 0);
        byte_size offset = row_offset + column_offset;
        if (offset > ixs_arr_size) {
          vals_out_of_bounds = true;
        }
        // NEVER EVER EVER EVER DO THIS IN REAL LIFE
        char *cvalptr = matrix.values.data() + offset;
        *(reinterpret_cast<T *>(cvalptr)) = t.second;
      }
    }
  }

  if (ixs_out_of_bounds) {
    LOG_WARNING("At least one index was written out of bounds!");
  }
  if (vals_out_of_bounds) {
    LOG_WARNING("At least one value was written out of bounds!");
  }

  // if (rsa) {
  //   print_rsa_matrix<int>(matrix.indices, indices_offsets,
  //                         byte_lengths_indices.back());
  //   print_rsa_matrix<T>(matrix.values, values_offsets,
  //                       byte_lengths_values.back());
  // } else {
  //   printc_vec<int>(matrix.indices, matrix.indices.size());
  //   printc_vec<T>(matrix.values, matrix.values.size());
  // }

  LOG_DEBUG("Done encoding");
  return matrix;
}

template <typename T>
SparseMatrix<T>::ellpack_matrix<T> &SparseMatrix<T>::ellpack_encode() {
  if (!ellpack_calculated) {
    calculate_ellpack();
  }
  return ellpackMatrix;
}

template <typename T> int SparseMatrix<T>::width() { return cols; }

template <typename T> inline int SparseMatrix<T>::height() { return rows; }

template <typename T> int SparseMatrix<T>::nonZeros() { return nonz; }

template class SparseMatrix<float>;
template class SparseMatrix<int>;
template class SparseMatrix<bool>;
template class SparseMatrix<double>;