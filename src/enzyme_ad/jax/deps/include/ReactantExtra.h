#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"

// #include "Enzyme/MLIR/Dialect/Dialect.h"
// #include "Enzyme/MLIR/Dialect/Ops.h"
// #include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
// #include "Enzyme/MLIR/Passes/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "xla/pjrt/pjrt_client.h"
// #include "xla/pjrt/pjrt_executable.h"

// int google::protobuf::io::CodedInputStream::default_recursion_limit_ = 100;
// int xla::_LayoutProto_default_instance_;

extern "C" void InitializeLogs();

// extern "C" MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx, int32_t val);

extern "C" xla::PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id, int num_nodes);

// xla/python/xla.cc 390
extern "C" xla::PjRtClient *MakeGPUClient(int node_id, int num_nodes,
                                          int *allowed_devices,
                                          int num_allowed_devices,
                                          const char *platform_name,
                                          const char **error);

extern "C" int ClientNumDevices(xla::PjRtClient *client);

extern "C" int ClientNumAddressableDevices(xla::PjRtClient *client);

extern "C" int ClientProcessIndex(xla::PjRtClient *client);

extern "C" xla::PjRtDevice *ClientGetDevice(xla::PjRtClient *client, int device_id);

extern "C" xla::PjRtDevice *ClientGetAddressableDevice(xla::PjRtClient *client, int device_id);

extern "C" void ExecutableFree(xla::PjRtLoadedExecutable *exec);

extern "C" xla::PjRtDevice *BufferToDevice(xla::PjRtBuffer *Buffer);

extern "C" xla::PjRtClient *BufferToClient(xla::PjRtBuffer *Buffer);

extern "C" xla::PjRtClient *DeviceToClient(xla::PjRtDevice *Device);

extern "C" void PjRtBufferFree(xla::PjRtBuffer *Buffer);

extern "C" void *UnsafeBufferPointer(xla::PjRtBuffer *buffer);

extern "C" xla::PjRtBuffer *ArrayFromHostBuffer(xla::PjRtClient *client, void *data,
                                                MlirType mtype, size_t dim,
                                                const int64_t *cshape,
                                                xla::PjRtDevice *device);

extern "C" uint8_t BufferOnCPU(xla::PjRtBuffer *buffer);

extern "C" xla::PjRtBuffer *CopyBufferToDevice(xla::PjRtBuffer *buffer,
                                               xla::PjRtDevice *dst_device);

extern "C" void BufferToHost(xla::PjRtBuffer *buffer, void *data);

extern "C" void FreeClient(xla::PjRtClient *client);

/* Note that this */
extern "C" xla::PjRtLoadedExecutable *ClientCompile(xla::PjRtClient *client,
                                                    MlirModule cmod);

typedef xla::PjRtFuture<> FutureType;

extern "C" void FreeFuture(FutureType *Future);

extern "C" uint8_t FutureIsReady(FutureType *Future);

extern "C" void FutureAwait(FutureType *Future);

extern "C" void RunPassPipeline(const char *pass_pipeline, MlirModule cmod);

extern "C" void XLAExecute(xla::PjRtLoadedExecutable *exec, int num_args,
                           xla::PjRtBuffer **op_args, uint8_t *is_arg_donatable,
                           int num_results, xla::PjRtBuffer **op_results,
                           uint8_t *futures, FutureType **future_results);

extern "C" void RegisterDialects(MlirContext cctx);

extern "C" void InitializeRegistryAndPasses(MlirDialectRegistry creg);