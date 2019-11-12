#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <vector>
#include <cassert>



int main() {
    // --- create library handles:
    cusolverSpHandle_t cusolver_handle;
    cusolverStatus_t cusolver_status;
    cusolver_status = cusolverSpCreate(&cusolver_handle);
    std::cout << "status create cusolver handle: " << cusolver_status << std::endl;

    cusparseHandle_t cusparse_handle;
    cusparseStatus_t cusparse_status;
    cusparse_status = cusparseCreate(&cusparse_handle);
    std::cout << "status create cusparse handle: " << cusparse_status << std::endl;

    // --- prepare matrix:
    int Nrows = 4;
    int Ncols = 4;
    std::vector<float> csrVal;
    std::vector<int> cooRow;
    std::vector<int> csrColInd;
    std::vector<float> b;

    assemble_poisson_matrix_coo(csrVal, cooRow, csrColInd, b, Nrows, Ncols);

    int nnz = csrVal.size();
    int m = Nrows * Ncols;
    std::vector<int> csrRowPtr(m+1);

    // --- prepare solving and copy to GPU:
    std::vector<float> x(m);
    float tol = 1e-5;
    int reorder = 0;
    int singularity = 0;

    float *db, *dcsrVal, *dx;
    int *dcsrColInd, *dcsrRowPtr, *dcooRow;
    cudaMalloc((void**)&db, m*sizeof(float));
    cudaMalloc((void**)&dx, m*sizeof(float));
    cudaMalloc((void**)&dcsrVal, nnz*sizeof(float));
    cudaMalloc((void**)&dcsrColInd, nnz*sizeof(int));
    cudaMalloc((void**)&dcsrRowPtr, (m+1)*sizeof(int));
    cudaMalloc((void**)&dcooRow, nnz*sizeof(int));

    cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dcsrVal, csrVal.data(), csrVal.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dcsrColInd, csrColInd.data(), csrColInd.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dcooRow, cooRow.data(), cooRow.size()*sizeof(int), cudaMemcpyHostToDevice);

    cusparse_status = cusparseXcoo2csr(cusparse_handle, dcooRow, nnz, m,
                                       dcsrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    std::cout << "status cusparse coo2csr conversion: " << cusparse_status << std::endl;

    cudaDeviceSynchronize(); // matrix format conversion has to be finished!

    // --- everything ready for computation:

    cusparseMatDescr_t descrA;

    cusparse_status = cusparseCreateMatDescr(&descrA);
    std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;

    // optional: print dense matrix that has been allocated on GPU

    std::vector<float> A(m*m, 0);
    float *dA;
    cudaMalloc((void**)&dA, A.size()*sizeof(float));

    cusparseScsr2dense(cusparse_handle, m, m, descrA, dcsrVal,
                       dcsrRowPtr, dcsrColInd, dA, m);

    cudaMemcpy(A.data(), dA, A.size()*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "A: \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << A[i*m + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(dA);

    std::cout << "b: \n";
    cudaMemcpy(b.data(), db, (m)*sizeof(int), cudaMemcpyDeviceToHost);
    for (auto a : b) {
        std::cout << a << ",";
    }
    std::cout << std::endl;


    // --- solving!!!!

//    cusolver_status = cusolverSpScsrlsvchol(cusolver_handle, m, nnz, descrA, dcsrVal,
//                       dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
//                       &singularity);

     cusolver_status = cusolverSpScsrlsvqr(cusolver_handle, m, nnz, descrA, dcsrVal,
                        dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
                        &singularity);

    cudaDeviceSynchronize();

    std::cout << "singularity (should be -1): " << singularity << std::endl;

    std::cout << "status cusolver solving (!): " << cusolver_status << std::endl;

    cudaMemcpy(x.data(), dx, m*sizeof(float), cudaMemcpyDeviceToHost);

    // relocated these 2 lines from above to solve (2):
    cusparse_status = cusparseDestroy(cusparse_handle);
    std::cout << "status destroy cusparse handle: " << cusparse_status << std::endl;

    cusolver_status = cusolverSpDestroy(cusolver_handle);
    std::cout << "status destroy cusolver handle: " << cusolver_status << std::endl;

    for (auto a : x) {
        std::cout << a << " ";
    }
    std::cout << std::endl;



    cudaFree(db);
    cudaFree(dx);
    cudaFree(dcsrVal);
    cudaFree(dcsrColInd);
    cudaFree(dcsrRowPtr);
    cudaFree(dcooRow);

    return 0;
}

