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
  // start off by doing a histogram sum of the values in the sparse matrix,
  // and (simultaneously) calculate the maximum row length (we might use this
  // when we're padding the width later)
  std::vector<int> row_lengths(height(), 0);
  int max_width = 0;
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
  SparseMatrix::ellpack_matrix<T> ellpackMatrix(height(), ellpack_row<T>(0));
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
  // if we're RSA, add a row for the offsets
  if (rsa) {
    concrete_height = concrete_height + 1;
  }
  LOG_DEBUG("concrete height: ", concrete_height);
  // now we have the height, create an array for the actual lengths, and copy
  // the row lengths - with an offset if we're in RSA!
  std::vector<unsigned int> concrete_lengths(concrete_height, 0);
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
  std::vector<unsigned int> byte_lengths_indices(concrete_lengths);
  std::vector<unsigned int> byte_lengths_values(concrete_lengths);
  std::transform(byte_lengths_indices.begin(), byte_lengths_indices.end(),
                 byte_lengths_indices.begin(),
                 [rsa](unsigned int l) -> unsigned int {
                   return (l * sizeof(int)) + (rsa ? 2 * sizeof(int) : 0);
                 });
  std::transform(byte_lengths_values.begin(), byte_lengths_values.end(),
                 byte_lengths_values.begin(),
                 [rsa](unsigned int l) -> unsigned int {
                   return (l * sizeof(T)) + (rsa ? 2 * sizeof(int) : 0);
                 });
  // set the offset sizes if we're RSA, and append size for the offsets
  if (rsa) {
    int offset_size = (concrete_height - 1) * sizeof(int);
    byte_lengths_indices[0] = offset_size;
    byte_lengths_values[0] = offset_size;
  }
  // -------------------------------------------------------------------------
  // Step 2.4: Calculate the overall sizes.
  // -------------------------------------------------------------------------
  // the final sizes we'll need.
  // we can also set the matrix sizes here, as we know them!
  int cl_height = -1, cl_width = -1;
  if (rsa) {
    cl_height = concrete_height - 1;
  } else {
    cl_height = concrete_height;
    cl_width = regular_width;
  }
  unsigned int ixs_arr_size = std::accumulate(byte_lengths_indices.begin(),
                                              byte_lengths_indices.end(), 0);
  unsigned int vals_arr_size = std::accumulate(byte_lengths_values.begin(),
                                               byte_lengths_values.end(), 0);

  if (ixs_arr_size > device_max_alloc_bytes) {
    throw ixs_arr_size;
  }
  LOG_DEBUG("ixs_arr_size: (GB) - ",
            (double)ixs_arr_size / (double)(1024 * 1024 * 1024));
  LOG_DEBUG("ixs_arr_size: (GB) - ",
            (double)vals_arr_size / (double)(1024 * 1024 * 1024));
  // =========================================================================
  // STEP THREE: CREATE THE TWO ARRAYS, AND FILL WITH MATRIX INFORMATION
  // =========================================================================
  CL_matrix matrix(ixs_arr_size, vals_arr_size, cl_width, cl_height);

  return matrix;
}

template <typename T> int SparseMatrix<T>::width() { return cols; }

template <typename T> inline int SparseMatrix<T>::height() { return rows; }

template <typename T> int SparseMatrix<T>::nonZeros() { return nonz; }

template class SparseMatrix<float>;
template class SparseMatrix<int>;
template class SparseMatrix<bool>;
template class SparseMatrix<double>;