// CSR sparse matrix products (spmv / spmm) via cuSPARSE.
//
// The `lower-enzymexla-sparse` pass emits stablehlo.custom_calls with
// api_version = 4 (TYPED_FFI) targeting either
//   - "enzymejax_sparse_csr_matmul_ffi":     out = alpha * A * dense, with
//     operands (rowptr, colind, nzval, dense), or
//   - "enzymejax_sparse_csr_matmul_acc_ffi": out = alpha * A * dense + beta *
//     acc, with operands (rowptr, colind, nzval, dense, acc) and acc aliased to
//     out.
// Column-major layouts are pinned; the result is dense. The backend_config
// dict carries i64 attributes "m", "n", "transpose" (must be 0 for now) and
// "index_base" (0 or 1; sparse_tensor-dialect buffers are 0-based), plus f64
// "alpha" (and "beta" for the accumulating variant).

#include <cstddef>
#include <cstdint>

#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if defined(ENZYMEJAX_CUDA)
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusparse.h"

namespace enzymexla {
namespace ffi_internal {

namespace ffi = xla::ffi;

#define ENZYMEJAX_CUSPARSE_RET(expr)                                           \
  do {                                                                         \
    cusparseStatus_t status__ = (expr);                                        \
    if (status__ != CUSPARSE_STATUS_SUCCESS) {                                 \
      return ffi::Error(                                                       \
          ffi::ErrorCode::kInternal,                                           \
          absl::StrFormat("enzymejax_sparse_csr_matmul: %s failed: %s", #expr, \
                          cusparseGetErrorString(status__)));                  \
    }                                                                          \
  } while (0)

// Computes out = alpha_v * A * dense (+ beta_v * *acc when acc != nullptr;
// the accumulated-into operand is aliased to out by the lowering, so it is
// copied into out first if XLA did not reuse the buffer).
static ffi::Error csrMatmulCudaImpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer *acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha_v, double beta_v) {
  if (transpose != 0) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "enzymejax_sparse_csr_matmul: transposed products are not supported");
  }
  if (colind.element_type() != rowptr.element_type()) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "enzymejax_sparse_csr_matmul: rowptr and colind dtypes must match");
  }

  cusparseIndexType_t index_type;
  switch (rowptr.element_type()) {
  case ffi::DataType::S32:
    index_type = CUSPARSE_INDEX_32I;
    break;
  case ffi::DataType::S64:
    index_type = CUSPARSE_INDEX_64I;
    break;
  default:
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "enzymejax_sparse_csr_matmul: index buffers must be i32 or i64");
  }

  const float alpha_f = static_cast<float>(alpha_v),
              beta_f = static_cast<float>(beta_v);
  const double alpha_d = alpha_v, beta_d = beta_v;
  cudaDataType value_type;
  size_t value_bytes;
  const void *alpha, *beta;
  switch (nzval.element_type()) {
  case ffi::DataType::F32:
    value_type = CUDA_R_32F;
    value_bytes = sizeof(float);
    alpha = &alpha_f;
    beta = &beta_f;
    break;
  case ffi::DataType::F64:
    value_type = CUDA_R_64F;
    value_bytes = sizeof(double);
    alpha = &alpha_d;
    beta = &beta_d;
    break;
  default:
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "enzymejax_sparse_csr_matmul: only f32 and f64 values are supported");
  }
  if (dense.element_type() != nzval.element_type() ||
      out->element_type() != nzval.element_type() ||
      (acc != nullptr && acc->element_type() != nzval.element_type())) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "enzymejax_sparse_csr_matmul: value, operand and result "
                      "dtypes must match");
  }
  if (acc != nullptr && acc->untyped_data() != out->untyped_data()) {
    cudaError_t copy_status =
        cudaMemcpyAsync(out->untyped_data(), acc->untyped_data(),
                        static_cast<size_t>(out->element_count()) * value_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    if (copy_status != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        absl::StrFormat("enzymejax_sparse_csr_matmul: copying "
                                        "the accumulated-into operand "
                                        "failed: %s",
                                        cudaGetErrorString(copy_status)));
    }
  }

  static thread_local cusparseHandle_t handle = nullptr;
  if (handle == nullptr) {
    ENZYMEJAX_CUSPARSE_RET(cusparseCreate(&handle));
  }
  ENZYMEJAX_CUSPARSE_RET(cusparseSetStream(handle, stream));

  int64_t nnz = colind.element_count();
  cusparseSpMatDescr_t mat_a;
  ENZYMEJAX_CUSPARSE_RET(cusparseCreateCsr(
      &mat_a, m, n, nnz, rowptr.untyped_data(), colind.untyped_data(),
      nzval.untyped_data(), index_type, index_type,
      index_base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
      value_type));

  auto with_workspace = [&](size_t buffer_size, auto &&compute) -> ffi::Error {
    void *workspace = nullptr;
    if (buffer_size > 0) {
      auto maybe_workspace = scratch.Allocate(buffer_size);
      if (!maybe_workspace.has_value()) {
        return ffi::Error(
            ffi::ErrorCode::kResourceExhausted,
            "enzymejax_sparse_csr_matmul: failed to allocate workspace");
      }
      workspace = *maybe_workspace;
    }
    return compute(workspace);
  };

  ffi::Error err = ffi::Error::Success();
  int64_t rank = dense.dimensions().size();
  if (rank == 1) {
    cusparseDnVecDescr_t vec_x, vec_y;
    ENZYMEJAX_CUSPARSE_RET(
        cusparseCreateDnVec(&vec_x, n, dense.untyped_data(), value_type));
    ENZYMEJAX_CUSPARSE_RET(
        cusparseCreateDnVec(&vec_y, m, out->untyped_data(), value_type));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      ENZYMEJAX_CUSPARSE_RET(cusparseSpMV_bufferSize(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x, beta,
          vec_y, value_type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        ENZYMEJAX_CUSPARSE_RET(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x, beta,
            vec_y, value_type, CUSPARSE_SPMV_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
  } else if (rank == 2) {
    // Layouts are pinned column-major by lower-enzymexla-sparse.
    int64_t k = dense.dimensions()[0];
    int64_t c = dense.dimensions()[1];
    cusparseDnMatDescr_t mat_b, mat_c;
    ENZYMEJAX_CUSPARSE_RET(cusparseCreateDnMat(&mat_b, k, c, /*ld=*/k,
                                               dense.untyped_data(), value_type,
                                               CUSPARSE_ORDER_COL));
    ENZYMEJAX_CUSPARSE_RET(cusparseCreateDnMat(&mat_c, m, c, /*ld=*/m,
                                               out->untyped_data(), value_type,
                                               CUSPARSE_ORDER_COL));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      ENZYMEJAX_CUSPARSE_RET(cusparseSpMM_bufferSize(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
          value_type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        ENZYMEJAX_CUSPARSE_RET(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
            value_type, CUSPARSE_SPMM_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    cusparseDestroyDnMat(mat_b);
    cusparseDestroyDnMat(mat_c);
  } else {
    err = ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "enzymejax_sparse_csr_matmul: dense operand must have rank 1 or 2");
  }
  cusparseDestroySpMat(mat_a);
  return err;
}

static ffi::Error csrMatmulCuda(cudaStream_t stream,
                                ffi::ScratchAllocator scratch,
                                ffi::AnyBuffer rowptr, ffi::AnyBuffer colind,
                                ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
                                ffi::Result<ffi::AnyBuffer> out, int64_t m,
                                int64_t n, int64_t transpose,
                                int64_t index_base, double alpha) {
  return csrMatmulCudaImpl(stream, scratch, rowptr, colind, nzval, dense,
                           /*acc=*/nullptr, out, m, n, transpose, index_base,
                           alpha, /*beta_v=*/0.0);
}

static ffi::Error csrMatmulAccCuda(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha, double beta) {
  return csrMatmulCudaImpl(stream, scratch, rowptr, colind, nzval, dense, &acc,
                           out, m, n, transpose, index_base, alpha, beta);
}

XLA_FFI_DEFINE_HANDLER(csrMatmulHandlerCUDA, csrMatmulCuda,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER(csrMatmulAccHandlerCUDA, csrMatmulAccCuda,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Arg<ffi::AnyBuffer>() // acc (aliased to out)
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha")
                           .Attr<double>("beta"));

void registerEnzymeJaXXLACudaSparseFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymejax_sparse_csr_matmul_ffi", "CUDA",
                           csrMatmulHandlerCUDA);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymejax_sparse_csr_matmul_acc_ffi", "CUDA",
                           csrMatmulAccHandlerCUDA);
}

} // namespace ffi_internal
} // namespace enzymexla

#else

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLACudaSparseFFI() {}

} // namespace ffi_internal
} // namespace enzymexla

#endif
