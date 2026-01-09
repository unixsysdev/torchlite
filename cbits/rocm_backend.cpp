#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <math.h>

static rocblas_handle g_handle = NULL;

extern "C" int rocm_init() {
  if (g_handle != NULL) {
    return 0;
  }
  return rocblas_create_handle(&g_handle) == rocblas_status_success ? 0 : 1;
}

extern "C" int rocm_shutdown() {
  if (g_handle == NULL) {
    return 0;
  }
  int rc = rocblas_destroy_handle(g_handle) == rocblas_status_success ? 0 : 1;
  g_handle = NULL;
  return rc;
}

extern "C" int rocm_alloc(void** out, size_t bytes) {
  return hipMalloc(out, bytes) == hipSuccess ? 0 : 1;
}

extern "C" int rocm_free(void* ptr) {
  return hipFree(ptr) == hipSuccess ? 0 : 1;
}

extern "C" int rocm_copy_h2d(void* dst, const double* src, size_t bytes) {
  return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess ? 0 : 1;
}

extern "C" int rocm_copy_d2h(double* dst, const void* src, size_t bytes) {
  return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost) == hipSuccess ? 0 : 1;
}

extern "C" int rocm_memset(void* dst, int value, size_t bytes) {
  return hipMemset(dst, value, bytes) == hipSuccess ? 0 : 1;
}

__global__ void fill_kernel(double* dst, double value, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = value;
  }
}

extern "C" int rocm_fill(double* dst, double value, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(fill_kernel, dim3(grid), dim3(block), 0, 0, dst, value, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void add_kernel(const double* a, const double* b, double* out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" int rocm_add(const double* a, const double* b, double* out, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(add_kernel, dim3(grid), dim3(block), 0, 0, a, b, out, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void add_inplace_kernel(double* dst, const double* src, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

extern "C" int rocm_add_inplace(double* dst, const double* src, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(add_inplace_kernel, dim3(grid), dim3(block), 0, 0, dst, src, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void mul_kernel(const double* a, const double* b, double* out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] * b[idx];
  }
}

extern "C" int rocm_mul(const double* a, const double* b, double* out, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(mul_kernel, dim3(grid), dim3(block), 0, 0, a, b, out, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void scale_kernel(const double* a, double* out, double alpha, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = alpha * a[idx];
  }
}

extern "C" int rocm_scale(const double* a, double* out, double alpha, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(scale_kernel, dim3(grid), dim3(block), 0, 0, a, out, alpha, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void relu_kernel(const double* a, double* out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double v = a[idx];
    out[idx] = v > 0.0 ? v : 0.0;
  }
}

extern "C" int rocm_relu(const double* a, double* out, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(relu_kernel, dim3(grid), dim3(block), 0, 0, a, out, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void relu_backward_kernel(const double* a, const double* grad, double* out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] > 0.0 ? grad[idx] : 0.0;
  }
}

extern "C" int rocm_relu_backward(const double* a, const double* grad, double* out, size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(relu_backward_kernel, dim3(grid), dim3(block), 0, 0, a, grad, out, n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void add_bias_kernel(const double* a, const double* bias, double* out, int rows, int cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * (size_t)cols;
  if (idx < n) {
    int r = (int)(idx % rows);
    out[idx] = a[idx] + bias[r];
  }
}

extern "C" int rocm_add_bias(const double* a, const double* bias, double* out, int rows, int cols) {
  size_t n = (size_t)rows * (size_t)cols;
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(add_bias_kernel, dim3(grid), dim3(block), 0, 0, a, bias, out, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void sum_columns_kernel(const double* a, double* out, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
      acc += a[r + c * rows];
    }
    out[r] = acc;
  }
}

extern "C" int rocm_sum_columns(const double* a, double* out, int rows, int cols) {
  const size_t block = 256;
  const size_t grid = (rows + (int)block - 1) / (int)block;
  hipLaunchKernelGGL(sum_columns_kernel, dim3(grid), dim3(block), 0, 0, a, out, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void softmax_cols_kernel(const double* a, double* out, int rows, int cols) {
  int col = blockIdx.x;
  if (col >= cols) {
    return;
  }
  const double* col_ptr = a + col * rows;
  double max_v = col_ptr[0];
  for (int r = 1; r < rows; ++r) {
    double v = col_ptr[r];
    if (v > max_v) {
      max_v = v;
    }
  }
  double sum = 0.0;
  for (int r = 0; r < rows; ++r) {
    double e = exp(col_ptr[r] - max_v);
    out[r + col * rows] = e;
    sum += e;
  }
  double inv = 1.0 / sum;
  for (int r = 0; r < rows; ++r) {
    out[r + col * rows] *= inv;
  }
}

extern "C" int rocm_softmax_cols(const double* a, double* out, int rows, int cols) {
  hipLaunchKernelGGL(softmax_cols_kernel, dim3(cols), dim3(1), 0, 0, a, out, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void softmax_backward_kernel(const double* soft, const double* grad, double* out, int rows, int cols) {
  int col = blockIdx.x;
  if (col >= cols) {
    return;
  }
  const double* s_ptr = soft + col * rows;
  const double* g_ptr = grad + col * rows;
  double dot = 0.0;
  for (int r = 0; r < rows; ++r) {
    dot += s_ptr[r] * g_ptr[r];
  }
  for (int r = 0; r < rows; ++r) {
    out[r + col * rows] = s_ptr[r] * (g_ptr[r] - dot);
  }
}

extern "C" int rocm_softmax_backward(const double* soft, const double* grad, double* out, int rows, int cols) {
  hipLaunchKernelGGL(softmax_backward_kernel, dim3(cols), dim3(1), 0, 0, soft, grad, out, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void write_row_kernel(const double* row, double* out, int row_idx, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < cols) {
    out[row_idx + col * rows] = row[col];
  }
}

extern "C" int rocm_write_row(const double* row, double* out, int row_idx, int rows, int cols) {
  const size_t block = 256;
  const size_t grid = (cols + (int)block - 1) / (int)block;
  hipLaunchKernelGGL(write_row_kernel, dim3(grid), dim3(block), 0, 0, row, out, row_idx, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void add_row_kernel(const double* src, double* dst, int row_idx, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < cols) {
    dst[col] += src[row_idx + col * rows];
  }
}

extern "C" int rocm_add_row(const double* src, double* dst, int row_idx, int rows, int cols) {
  const size_t block = 256;
  const size_t grid = (cols + (int)block - 1) / (int)block;
  hipLaunchKernelGGL(add_row_kernel, dim3(grid), dim3(block), 0, 0, src, dst, row_idx, rows, cols);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

extern "C" int rocm_matmul(const double* a,
                           const double* b,
                           double* out,
                           int m,
                           int n,
                           int k,
                           int lda,
                           int ldb,
                           int ldc,
                           int transA,
                           int transB) {
  if (g_handle == NULL) {
    if (rocm_init() != 0) {
      return 1;
    }
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  rocblas_operation opA = transA ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation opB = transB ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_status stat = rocblas_dgemm(g_handle,
                                      opA,
                                      opB,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      a,
                                      lda,
                                      b,
                                      ldb,
                                      &beta,
                                      out,
                                      ldc);
  return stat == rocblas_status_success ? 0 : 1;
}

__global__ void adamw_kernel(double* param,
                             const double* grad,
                             double* m,
                             double* v,
                             double lr,
                             double beta1,
                             double beta2,
                             double eps,
                             double weight_decay,
                             double bias_corr1,
                             double bias_corr2,
                             size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double g = grad[idx];
    double m_new = beta1 * m[idx] + (1.0 - beta1) * g;
    double v_new = beta2 * v[idx] + (1.0 - beta2) * (g * g);
    m[idx] = m_new;
    v[idx] = v_new;
    double m_hat = m_new / bias_corr1;
    double v_hat = v_new / bias_corr2;
    double update = m_hat / (sqrt(v_hat) + eps);
    update += weight_decay * param[idx];
    param[idx] -= lr * update;
  }
}

extern "C" int rocm_adamw_step(double* param,
                               const double* grad,
                               double* m,
                               double* v,
                               double lr,
                               double beta1,
                               double beta2,
                               double eps,
                               double weight_decay,
                               double bias_corr1,
                               double bias_corr2,
                               size_t n) {
  const size_t block = 256;
  const size_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(adamw_kernel,
                     dim3(grid),
                     dim3(block),
                     0,
                     0,
                     param,
                     grad,
                     m,
                     v,
                     lr,
                     beta1,
                     beta2,
                     eps,
                     weight_decay,
                     bias_corr1,
                     bias_corr2,
                     n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}
