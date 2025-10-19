#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(MLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define MLIR_CAPI_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if MLIR_CAPI_BUILDING_LIBRARY
#define MLIR_CAPI_EXPORTED __declspec(dllexport)
#else
#define MLIR_CAPI_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define MLIR_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

template <bool withError> struct CallInfo;

template <> struct CallInfo<false> {
  void (*run)(const void **, void *, void *);
  void *(*init)();
};

template <> struct CallInfo<true> {
  char *(*run)(const void **, void *, void *);
  void *(*init)();
};

XLA_FFI_Error *instantiate(XLA_FFI_CallFrame *call_frame) { return nullptr; }

XLA_FFI_Error *prepare(XLA_FFI_CallFrame *call_frame) { return nullptr; }

struct CuFuncWrapper {
  void *func;
};

void noop(void *){};

template <bool withError>
XLA_FFI_Error *initialize(XLA_FFI_CallFrame *call_frame) {
  assert(call_frame->attrs.size == 1);
  assert(call_frame->attrs.types[0] == XLA_FFI_AttrType_STRING);

  auto *bspan =
      reinterpret_cast<XLA_FFI_ByteSpan *>(call_frame->attrs.attrs[0]);

  CallInfo<withError> *cinfo;
  if constexpr (withError) {
    cinfo = (CallInfo<true> *)bspan->ptr;
  } else {
    cinfo = (CallInfo<false> *)bspan->ptr;
  }

  auto internal_api = call_frame->api->internal_api;
  auto ctx = call_frame->ctx;
  void *stream =
      ((stream_executor::Stream *)internal_api->XLA_FFI_INTERNAL_Stream_Get(
           ctx))
          ->platform_specific_handle()
          .stream;

  (void)stream;
  /*
  CUcontext pctx;
  auto err = cuStreamGetCtx ((CUstream)stream, &pctx);

  CUcontext cctx;
  err = cuCtxGetCurrent ( &cctx );

  err = cuCtxPushCurrent (pctx);
  */

  void *cufunc = cinfo->init();

  /*
  CUcontext tctx;
  err = cuCtxPopCurrent(&tctx);
  */

  auto *execution_state = reinterpret_cast<xla::ffi::ExecutionState *>(
      internal_api->XLA_FFI_INTERNAL_ExecutionState_Get(ctx));
  (void)execution_state->Set(
      xla::ffi::TypeRegistry::GetTypeId<CuFuncWrapper>(), cufunc, noop);

  return nullptr;
}

template <bool withError>
XLA_FFI_Error *execute(XLA_FFI_CallFrame *call_frame) {
  // If passed a call frame with the metadata extension, just return the
  // metadata.
  if (call_frame->extension_start != nullptr &&
      call_frame->extension_start->type == XLA_FFI_Extension_Metadata) {
    auto extension = reinterpret_cast<XLA_FFI_Metadata_Extension *>(
        call_frame->extension_start);
    extension->metadata->api_version = XLA_FFI_Api_Version{
        XLA_FFI_Api_Version_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        XLA_FFI_API_MAJOR,
        XLA_FFI_API_MINOR,
    };
    return nullptr;
  }

  auto *bspan =
      reinterpret_cast<XLA_FFI_ByteSpan *>(call_frame->attrs.attrs[0]);

  CallInfo<withError> *cinfo;
  if constexpr (withError) {
    cinfo = (CallInfo<true> *)bspan->ptr;
  } else {
    cinfo = (CallInfo<false> *)bspan->ptr;
  }

  auto internal_api = call_frame->api->internal_api;
  auto ctx = call_frame->ctx;
  void *stream =
      ((stream_executor::Stream *)internal_api->XLA_FFI_INTERNAL_Stream_Get(
           ctx))
          ->platform_specific_handle()
          .stream;

  size_t numargs = call_frame->args.size;
  void *ptrs[numargs];
  for (size_t i = 0; i < numargs; i++) {
    ptrs[i] =
        &reinterpret_cast<XLA_FFI_Buffer *>(call_frame->args.args[i])->data;
  }

  auto *execution_state = reinterpret_cast<xla::ffi::ExecutionState *>(
      internal_api->XLA_FFI_INTERNAL_ExecutionState_Get(ctx));
  auto cufunc = (void *)execution_state->Get<CuFuncWrapper>().value();

  const void **const_ptrs = const_cast<const void **>(ptrs);

  if constexpr (withError) {
    char *err = cinfo->run(const_ptrs, stream, cufunc);
    if (err) {
      XLA_FFI_Error_Create_Args error_args = {
          XLA_FFI_Error_Create_Args_STRUCT_SIZE,
          /*extension_start=*/nullptr,
          /*message=*/err,
          /*errc=*/XLA_FFI_Error_Code_INTERNAL};
      auto ffi_api = call_frame->api;
      return ffi_api->XLA_FFI_Error_Create(&error_args);
    }
  } else {
    cinfo->run(const_ptrs, stream, cufunc);
  }

  return nullptr;
}

extern "C" MLIR_CAPI_EXPORTED void RegisterEnzymeXLAGPUHandler() {
  XLA_FFI_Handler_Bundle bundle = {instantiate, prepare, initialize<false>,
                                   execute<false>};

  xla::ffi::Ffi::RegisterStaticHandler(xla::ffi::GetXlaFfiApi(),
                                       "enzymexla_compile_gpu", "CUDA", bundle,
                                       /*XLA_FFI_Handler_Traits traits = */ 0);

  XLA_FFI_Handler_Bundle bundle_with_error = {instantiate, prepare,
                                              initialize<true>, execute<true>};

  xla::ffi::Ffi::RegisterStaticHandler(
      xla::ffi::GetXlaFfiApi(), "enzymexla_compile_gpu_with_error", "CUDA",
      bundle_with_error, /*XLA_FFI_Handler_Traits traits = */ 0);
}
