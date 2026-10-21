#include "xla/ffi/api/c_api_internal.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/stream_executor/stream.h"

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

typedef void *CUStream;

template <bool withError> struct CallInfo;

template <> struct CallInfo<false> {
  void (*run)(const void **, void *, void *);
  void *(*init)(void **);
};

template <> struct CallInfo<true> {
  char *(*run)(const void **, void *, void *);
  void *(*init)(void **);
};

struct CuKernelWrapper {
  static xla::ffi::TypeId id;

  void *library = nullptr;
  void *cuKernel = nullptr;
};
xla::ffi::TypeId CuKernelWrapper::id = {};

XLA_FFI_REGISTER_TYPE(xla::ffi::GetXlaFfiApi(), "enzymexla.CuKernelWrapper",
                      &CuKernelWrapper::id,
                      xla::ffi::MakeTypeInfo<CuKernelWrapper>());

xla::ffi::ErrorOr<std::unique_ptr<CuKernelWrapper>>
instantiate(std::string_view bspan) {
  CallInfo<false> *cinfo = (CallInfo<false> *)bspan.data();

  auto kwrapper = std::make_unique<CuKernelWrapper>();

  void *library;
  kwrapper->cuKernel = cinfo->init(&kwrapper->library);

  if (!kwrapper->cuKernel ||
    !kwrapper->library) {
      return xla::ffi::Error::Internal("enzymexla_compile_gpu: could not load module");
    }

  return kwrapper;
}

template <bool withError>
xla::ffi::Error execute(std::string_view bspan, CUStream stream,
                        CuKernelWrapper *kwrapper, xla::ffi::RemainingArgs args,
                        xla::ffi::RemainingRets rets) {
  CallInfo<withError> *cinfo;
  if constexpr (withError) {
    cinfo = (CallInfo<true> *)bspan.data();
  } else {
    cinfo = (CallInfo<false> *)bspan.data();
  }

  size_t numargs = args.size();
  void *devptrs[numargs];
  void *ptrs[numargs];
  for (size_t i = 0; i < numargs; i++) {
    auto buf = args.get<xla::ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return xla::ffi::Error::Internal(absl::StrFormat(
          "enzymexla_compile_gpu: could not get %d argument buffer", i));
    devptrs[i] = buf->untyped_data();
    ptrs[i]  = &devptrs[i];
  }

  const void **const_ptrs = const_cast<const void **>(ptrs);
  auto cufunc = kwrapper->cuKernel;
  assert(cufunc);

  if constexpr (withError) {
    char *err = cinfo->run(const_ptrs, stream, cufunc);
    if (err) {
      auto errmsg = absl::StrFormat("error running gpu thunk: %s", err);
      return xla::ffi::Error::Internal(errmsg);
    }
  } else {
    cinfo->run(const_ptrs, stream, cufunc);
  }

  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kInstantiate, instantiate,
    xla::ffi::Ffi::BindInstantiate().Attr<std::string_view>("attr"));

XLA_FFI_DEFINE_HANDLER(kExecute, execute<false>,
                       xla::ffi::Ffi::BindExecute()
                           .Attr<std::string_view>("attr")
                           .Ctx<xla::ffi::PlatformStream<CUStream>>()
                           .Ctx<xla::ffi::State<CuKernelWrapper>>()
                           .RemainingArgs()
                           .RemainingRets());
XLA_FFI_DEFINE_HANDLER(kExecuteWithError, execute<true>,
                       xla::ffi::Ffi::BindExecute()
                           .Attr<std::string_view>("attr")
                           .Ctx<xla::ffi::PlatformStream<CUStream>>()
                           .Ctx<xla::ffi::State<CuKernelWrapper>>()
                           .RemainingArgs()
                           .RemainingRets());

extern "C" MLIR_CAPI_EXPORTED void RegisterEnzymeXLAGPUHandler() {
  XLA_FFI_Handler_Bundle bundle = {kInstantiate, /*prepare*/ nullptr,
                                   /*initialize*/ nullptr, kExecute};

  xla::ffi::Ffi::RegisterStaticHandler(xla::ffi::GetXlaFfiApi(),
                                       "enzymexla_compile_gpu", "CUDA", bundle,
                                       /*XLA_FFI_Handler_Traits traits = */ 0);

  XLA_FFI_Handler_Bundle bundle_with_error = {kInstantiate, /*prepare*/ nullptr,
                                              /*initialize*/ nullptr,
                                              kExecuteWithError};

  xla::ffi::Ffi::RegisterStaticHandler(
      xla::ffi::GetXlaFfiApi(), "enzymexla_compile_gpu_with_error", "CUDA",
      bundle_with_error, /*XLA_FFI_Handler_Traits traits = */ 0);
}
