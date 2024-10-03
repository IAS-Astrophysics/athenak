#ifndef UTILS_WRITE_VECTOR_HPP_
#define UTILS_WRITE_VECTOR_HPP_

#include <iostream>
#include <vector>
#include <cstdio> // For FILE, fopen, fwrite, fclose


void write3DVectorToFile(const std::vector<std::vector<std::vector<Real>>>& array3D, const char* filename) {
    // Open the file in binary write mode
    FILE* file = fopen(filename, "wb");
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    // Get the dimensions of the 3D array
    size_t dim1 = array3D.size();
    size_t dim2 = dim1 > 0 ? array3D[0].size() : 0;
    size_t dim3 = (dim1 > 0 && dim2 > 0) ? array3D[0][0].size() : 0;

    // Write the dimensions to the file
    fwrite(&dim1, sizeof(size_t), 1, file);
    fwrite(&dim2, sizeof(size_t), 1, file);
    fwrite(&dim3, sizeof(size_t), 1, file);

    // Write the data
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            // Write the innermost vector data
            const Real* dataPtr = array3D[i][j].data();
            size_t elementsWritten = fwrite(dataPtr, sizeof(Real), dim3, file);
            if (elementsWritten != dim3) {
                std::cerr << "Error: Failed to write data to file." << std::endl;
                fclose(file);
                return;
            }
        }
    }

    // Close the file
    fclose(file);
}

#endif // UTILS_WRITE_VECTOR_HPP_
