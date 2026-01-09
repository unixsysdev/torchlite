#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

int rocblas_dgemm_host(
    int transA,
    int transB,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double* C,
    int ldc) {
  rocblas_handle handle = NULL;
  rocblas_status rstat = rocblas_create_handle(&handle);
  if (rstat != rocblas_status_success) {
    return 1;
  }
  rstat = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
  if (rstat != rocblas_status_success) {
    rocblas_destroy_handle(handle);
    return 2;
  }

  const int a_cols = (transA == 0) ? k : m;
  const int b_cols = (transB == 0) ? n : k;
  const size_t a_bytes = (size_t)lda * (size_t)a_cols * sizeof(double);
  const size_t b_bytes = (size_t)ldb * (size_t)b_cols * sizeof(double);
  const size_t c_bytes = (size_t)ldc * (size_t)n * sizeof(double);

  double* dA = NULL;
  double* dB = NULL;
  double* dC = NULL;
  if (hipMalloc((void**)&dA, a_bytes) != hipSuccess) {
    rocblas_destroy_handle(handle);
    return 3;
  }
  if (hipMalloc((void**)&dB, b_bytes) != hipSuccess) {
    hipFree(dA);
    rocblas_destroy_handle(handle);
    return 4;
  }
  if (hipMalloc((void**)&dC, c_bytes) != hipSuccess) {
    hipFree(dA);
    hipFree(dB);
    rocblas_destroy_handle(handle);
    return 5;
  }

  if (hipMemcpy(dA, A, a_bytes, hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    rocblas_destroy_handle(handle);
    return 6;
  }
  if (hipMemcpy(dB, B, b_bytes, hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    rocblas_destroy_handle(handle);
    return 7;
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  const rocblas_operation opA = (transA == 0) ? rocblas_operation_none : rocblas_operation_transpose;
  const rocblas_operation opB = (transB == 0) ? rocblas_operation_none : rocblas_operation_transpose;

  rstat = rocblas_dgemm(handle,
                        opA,
                        opB,
                        m,
                        n,
                        k,
                        &alpha,
                        dA,
                        lda,
                        dB,
                        ldb,
                        &beta,
                        dC,
                        ldc);
  if (rstat != rocblas_status_success) {
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    rocblas_destroy_handle(handle);
    return 8;
  }

  if (hipMemcpy(C, dC, c_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    rocblas_destroy_handle(handle);
    return 9;
  }

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  rocblas_destroy_handle(handle);
  return 0;
}
