#include <type_traits>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "mpi.h"
#include "mpi_ffi.h"

#include "src/enzyme_ad/jax/Passes/Comm/Dialect.h"

#include "../export_macro.h"

int mpi_unimplemented_stub(...) {
  abort();
  return -1;
}

#define EXLA_FFI_PREFIX enzymexla_ffi

// generates a function pointer for `FNAME` that points to
// `mpi_unimplemented_stub` by default and a exported C function for setting the
// value dynamically
// TODO replace with call to libdl like libblastrampoline does
#define EXLA_FFI_MPI_FUNCTION_BINDING(FNAME, CNAME)                            \
  decltype(FNAME) *EXLA_##FNAME =                                              \
      reinterpret_cast<decltype(FNAME) *>(&mpi_unimplemented_stub);            \
  extern "C" MLIR_CAPI_EXPORTED void EXLA_FFI_PREFIX##_set_##CNAME(            \
      void *ptr) {                                                             \
    EXLA_##FNAME = reinterpret_cast<decltype(FNAME) *>(ptr);                   \
  }

// generates a global variable for `FNAME` that defaults to the value by
// MPItrampoline by default and a exported C function for setting the value
// dynamically NOTE this should not be required once MPI v5 ABI is used as
// minimum version
#define EXLA_FFI_MPI_CONSTANT_BINDING(FNAME, CNAME)                            \
  int EXLA_##FNAME = FNAME;                                                    \
  extern "C" MLIR_CAPI_EXPORTED void EXLA_FFI_PREFIX##_set_##CNAME(int val) {  \
    EXLA_##FNAME = val;                                                        \
  }                                                                            \
  extern "C" MLIR_CAPI_EXPORTED int EXLA_FFI_PREFIX##_get_##CNAME() {          \
    return EXLA_##FNAME;                                                       \
  }

EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Comm_rank, mpi_comm_rank)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Comm_size, mpi_comm_size)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Comm_split, mpi_comm_split)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Barrier, mpi_barrier)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Send, mpi_send)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Isend, mpi_isend)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Recv, mpi_recv)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Irecv, mpi_irecv)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Wait, mpi_wait)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Waitall, mpi_waitall)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Allreduce, mpi_allreduce)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Bcast, mpi_bcast)
EXLA_FFI_MPI_FUNCTION_BINDING(MPI_Error_string, mpi_error_string)

EXLA_FFI_MPI_CONSTANT_BINDING(MPI_STATUS_SIZE, mpi_status_size)
EXLA_FFI_MPI_CONSTANT_BINDING(MPI_SUCCESS, mpi_success)
EXLA_FFI_MPI_CONSTANT_BINDING(MPI_MAX_ERROR_STRING, mpi_max_error_string)

#define GENERATE_MPI_OP_LIST(X)                                                \
  X(MPI_OP_NULL, mpi_op_null)                                                  \
  X(MPI_SUM, mpi_sum)                                                          \
  X(MPI_MIN, mpi_min)                                                          \
  X(MPI_MAX, mpi_max)                                                          \
  X(MPI_PROD, mpi_prod)                                                        \
  X(MPI_BAND, mpi_band)                                                        \
  X(MPI_BOR, mpi_bor)                                                          \
  X(MPI_BXOR, mpi_bxor)                                                        \
  X(MPI_LAND, mpi_land)                                                        \
  X(MPI_LOR, mpi_lor)                                                          \
  X(MPI_LXOR, mpi_lxor)                                                        \
  X(MPI_MINLOC, mpi_minloc)                                                    \
  X(MPI_MAXLOC, mpi_maxloc)                                                    \
  X(MPI_REPLACE, mpi_replace)                                                  \
  X(MPI_NO_OP, mpi_no_op)

GENERATE_MPI_OP_LIST(EXLA_FFI_MPI_ENUM_BINDING_CASE)

std::optional<MPI_Op> symbolizeMpiOp(llvm::StringRef op) {
  switch (op) {
    // clang-format off
    #define X(NAME, _) case #NAME: return EXLA_##NAME;
    GENERATE_MPI_OP_LIST(X)
    #undef X
    default: return std::nullopt;
    // clang-format on
  }
}

std::optional<MPI_Datatype>
convertPrimitiveTypeToMpiDatatype(ffi::DataType type) {
  switch (type) {
    // clang-format off
    case ffi::DataType::INVALID: return std::nullopt;
    case ffi::DataType::PRED: return MPI_C_BOOL;
    case ffi::DataType::S1: return std::nullopt;
    case ffi::DataType::S2: return std::nullopt;
    case ffi::DataType::S4: return std::nullopt;
    case ffi::DataType::S8: return MPI_INT8_T;
    case ffi::DataType::S16: return MPI_INT16_T;
    case ffi::DataType::S32: return MPI_INT32_T;
    case ffi::DataType::S64: return MPI_INT64_T;
    case ffi::DataType::U1: return std::nullopt;
    case ffi::DataType::U2: return std::nullopt;
    case ffi::DataType::U4: return std::nullopt;
    case ffi::DataType::U8: return MPI_UINT8_T;
    case ffi::DataType::U16: return MPI_UINT16_T;
    case ffi::DataType::U32: return MPI_UINT32_T;
    case ffi::DataType::U64: return MPI_UINT64_T;
    case ffi::DataType::F16: return std::nullopt;
    case ffi::DataType::F32: return MPI_FLOAT;
    case ffi::DataType::F64: return MPI_DOUBLE;
    case ffi::DataType::BF16: return std::nullopt;
    case ffi::DataType::C64: return MPI_C_FLOAT_COMPLEX;
    case ffi::DataType::C128: return MPI_C_DOUBLE_COMPLEX;
    case ffi::DataType::TOKEN: return std::nullopt;
    case ffi::DataType::F8E5M2: return std::nullopt;
    case ffi::DataType::F8E4M3: return std::nullopt;
    case ffi::DataType::F8E4M3FN: return std::nullopt;
    case ffi::DataType::F8E4M3B11FNUZ: return std::nullopt;
    case ffi::DataType::F8E5M2FNUZ: return std::nullopt;
    case ffi::DataType::F8E4M3FNUZ: return std::nullopt;
    case ffi::DataType::F8E3M4: return std::nullopt;
    case ffi::DataType::F4E2M1FN: return std::nullopt;
    case ffi::DataType::F8E8M0FNU: return std::nullopt;
    default: return std::nullopt;
    // clang-format on
  }
}

namespace enzymexla::ffi_internal {
namespace ffi = xla::ffi;

using ffi::Buffer, ffi::AnyBuffer;
using ffi::Result, ffi::RemainingArgs, ffi::RemainingRets;

using IntBuffer = Buffer<ffi::S32, 0>;
using PtrBuffer = Buffer<ffi::U64, 0>; // pointers, so use U64

using MpiCommBuffer = PtrBuffer;
using MpiRequestBuffer = PtrBuffer;

// MPI_Status is a non-ABI-stable struct, so use U8 x N buffer to hold it
// its size is platform-dependent and given by MPI_STATUS_SIZE
// NOTE its size stabilizes in MPI v5 ABI, but meanwhile we need to support
// variable size
using MpiStatusBuffer = Buffer<ffi::U8, 1>;

ffi::Error checkMpiStatusSize(const MpiStatusBuffer &buf) {
  if (buf.element_count() != EXLA_MPI_STATUS_SIZE) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Status buffer must have %d elements, got %d",
                        EXLA_MPI_STATUS_SIZE, buf.element_count()));
  }
  return ffi::Error::Success();
}

ffi::Error checkMpiError(const char *fname, const int err) {
  if (err == EXLA_MPI_SUCCESS)
    return ffi::Error::Success();
  std::vector<char> cstr(EXLA_MPI_MAX_ERROR_STRING);
  int len;
  EXLA_MPI_Error_string(err, cstr.data(), &len);
  std::string str(cstr.data(), len);
  return ffi::Error::InvalidArgument(
      absl::StrFormat("%s failed with error code %d: %s", fname, err, str));
}

ffi::Error MpiCommRankImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> rank_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Comm_rank(comm, rank_ptr->typed_data());
  return checkMpiError("MPI_Comm_rank", err);
}

XLA_FFI_DEFINE_HANDLER(MpiCommRankFfi, MpiCommRankImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Ret<IntBuffer>()     // rank
);

ffi::Error MpiCommSizeImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> size_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Comm_size(comm, size_ptr->typed_data());
  return checkMpiError("MPI_Comm_size", err);
}

XLA_FFI_DEFINE_HANDLER(MpiCommSizeFfi, MpiCommSizeImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Ret<IntBuffer>()     // size
);

ffi::Error MpiCommSplitImpl(MpiCommBuffer comm_ptr, IntBuffer color_ptr,
                            IntBuffer key_ptr,
                            Result<MpiCommBuffer> newcomm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int color = *color_ptr.typed_data();
  int key = *key_ptr.typed_data();
  MPI_Comm *newcomm = reinterpret_cast<MPI_Comm *>(newcomm_ptr->typed_data());
  int err = EXLA_MPI_Comm_split(comm, color, key, newcomm);
  return checkMpiError("MPI_Comm_split", err);
}

XLA_FFI_DEFINE_HANDLER(MpiCommSplitFfi, MpiCommSplitImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Arg<IntBuffer>()     // color
                           .Arg<IntBuffer>()     // key
                           .Ret<MpiCommBuffer>() // newcomm
);

ffi::Error MpiBarrierImpl(MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Barrier(comm);
  return checkMpiError("MPI_Barrier", err);
}

XLA_FFI_DEFINE_HANDLER(MpiBarrierFfi, MpiBarrierImpl,
                       xla::ffi::Ffi::Bind().Arg<MpiCommBuffer>());

ffi::Error MpiSendImpl(ffi::AnyBuffer buf, IntBuffer dest_ptr,
                       IntBuffer tag_ptr, MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Send: unsupported datatype %s", buf.element_type().getAsString()));
  }
  int err = EXLA_MPI_Send(buf.untyped_data(), count, datatype.value(), dest,
                          tag, comm);
  return checkMpiError("MPI_Send", err);
}

XLA_FFI_DEFINE_HANDLER(MpiSendFfi, MpiSendImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>() // buf
                           .Arg<IntBuffer>()      // dest
                           .Arg<IntBuffer>()      // tag
                           .Arg<MpiCommBuffer>()  // comm
);

ffi::Error MpiIsendImpl(ffi::AnyBuffer buf, IntBuffer dest_ptr,
                        IntBuffer tag_ptr, MpiCommBuffer comm_ptr,
                        Result<MpiRequestBuffer> request_ptr) {
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Isend: unsupported datatype %s",
                        buf.element_type().getAsString()));
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = EXLA_MPI_Isend(buf.untyped_data(), count, datatype.value(), dest,
                           tag, comm, request);
  return checkMpiError("MPI_Isend", err);
}

XLA_FFI_DEFINE_HANDLER(MpiIsendFfi, MpiIsendImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()   // buf
                           .Arg<IntBuffer>()        // dest
                           .Arg<IntBuffer>()        // tag
                           .Arg<MpiCommBuffer>()    // comm
                           .Ret<MpiRequestBuffer>() // request
);

ffi::Error MpiRecvImpl(int32_t datatype, IntBuffer source_ptr,
                       IntBuffer tag_ptr, MpiCommBuffer comm_ptr,
                       Result<ffi::AnyBuffer> buf,
                       Result<MpiStatusBuffer> status_ptr) {
  if (auto error = checkMpiStatusSize(*status_ptr); error.failure()) {
    return error;
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf->element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Recv: unsupported datatype %s",
                        buf->element_type().getAsString()));
  }
  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = EXLA_MPI_Recv(buf->untyped_data(), count, datatype.value(), source,
                          tag, comm, status);
  return checkMpiError("MPI_Recv", err);
}

XLA_FFI_DEFINE_HANDLER(MpiRecvFfi, MpiRecvImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<IntBuffer>()       // source
                           .Arg<IntBuffer>()       // tag
                           .Arg<MpiCommBuffer>()   // comm
                           .Ret<ffi::AnyBuffer>()  // buf
                           .Ret<MpiStatusBuffer>() // status
);

ffi::Error MpiIrecvImpl(int32_t datatype, IntBuffer source_ptr,
                        IntBuffer tag_ptr, MpiCommBuffer comm_ptr,
                        Result<ffi::AnyBuffer> buf,
                        Result<MpiRequestBuffer> request_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf->element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Irecv: unsupported datatype %s",
                        buf->element_type().getAsString()));
  }
  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = EXLA_MPI_Irecv(buf->untyped_data(), count, datatype.value(), source,
                           tag, comm, request);
  return checkMpiError("MPI_Irecv", err);
}

XLA_FFI_DEFINE_HANDLER(MpiIrecvFfi, MpiIrecvImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<IntBuffer>()        // source
                           .Arg<IntBuffer>()        // tag
                           .Arg<MpiCommBuffer>()    // comm
                           .Ret<ffi::AnyBuffer>()   // buf
                           .Ret<MpiRequestBuffer>() // request
);

ffi::Error MpiWaitImpl(MpiRequestBuffer request_ptr,
                       Result<MpiStatusBuffer> status_ptr) {
  if (auto error = checkMpiStatusSize(*status_ptr); error.failure()) {
    return error;
  }
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr.typed_data());
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = EXLA_MPI_Wait(request, status);
  return checkMpiError("MPI_Wait", err);
}

XLA_FFI_DEFINE_HANDLER(MpiWaitFfi, MpiWaitImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiRequestBuffer>() // request
                           .Ret<MpiStatusBuffer>()  // status
);

ffi::Error MpiWaitallImpl(ffi::RemainingArgs requests,
                          ffi::RemainingRets statuses) {
  if (requests.size() != statuses.size()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Waitall: requests and statuses must have the same "
                        "size, but got %d and %d",
                        requests.size(), statuses.size()));
  }
  int count = requests.size();

  // stack requests in an array
  std::vector<MPI_Request> request_vector(count);
  for (int i = 0; i < count; ++i) {
    auto buffer_or_error = requests.get<MpiRequestBuffer>(i);
    if (buffer_or_error.has_error())
      return buffer_or_error.error();

    auto buffer = buffer_or_error.value();
    auto value = *reinterpret_cast<MPI_Request *>(buffer.typed_data());
    request_vector[i] = value;
  }

  std::vector<MPI_Status> status_vector(count);
  int err =
      EXLA_MPI_Waitall(count, request_vector.data(), status_vector.data());
  return checkMpiError("MPI_Waitall", err);

  // copy statuses back to the output buffers
  for (int i = 0; i < count; ++i) {
    auto buffer_or_error = statuses.get<MpiStatusBuffer>(i);
    if (buffer_or_error.has_error())
      return buffer_or_error.error();

    auto buffer = buffer_or_error.value();
    if (auto error = checkMpiStatusSize(*buffer); error.failure()) {
      return error;
    }

    auto ptr = reinterpret_cast<MPI_Status *>(buffer->typed_data());
    *ptr = status_vector[i];
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiWaitallFfi, MpiWaitallImpl,
                       xla::ffi::Ffi::Bind()
                           .RemainingArgs() // requests
                           .RemainingRets() // statuses
);
ffi::Error MpiAllreduceImpl(ffi::AnyBuffer sendbuf, llvm::StringRef op_str,
                            MpiCommBuffer comm_ptr,
                            Result<ffi::AnyBuffer> recvbuf) {
  if (sendbuf.element_count() <= recvbuf->element_count()) {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Allreduce: recvbuf size (%d) must be at least sendbuf size (%d)",
        recvbuf->element_count(), sendbuf.element_count()));
  }
  if (sendbuf.element_type() != recvbuf->element_type()) {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Allreduce: sendbuf type (%s) must match recvbuf type (%s)",
        sendbuf.element_type().getAsString(),
        recvbuf->element_type().getAsString()));
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(sendbuf.element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Allreduce: unsupported datatype %s",
                        sendbuf.element_type().getAsString()));
  }
  auto op = symbolizeMpiOp(op_str);
  if (!op.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Allreduce: invalid operation %s", op_str.str()));
  }
  int count = sendbuf.element_count();
  int err = EXLA_MPI_Allreduce(sendbuf.untyped_data(), recvbuf->untyped_data(),
                               count, datatype.value(), op.value(), comm);
  return checkMpiError("MPI_Allreduce", err);
}

XLA_FFI_DEFINE_HANDLER(MpiAllreduceFfi, MpiAllreduceImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()   // sendbuf
                           .Attr<llvm::StringRef>() // op
                           .Arg<MpiCommBuffer>()    // comm
                           .Ret<ffi::AnyBuffer>()   // recvbuf
);

ffi::Error MpiBcastImpl(ffi::AnyBuffer buf, IntBuffer root_ptr,
                        MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (!datatype.has_value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Bcast: unsupported datatype %s",
                        buf.element_type().getAsString()));
  }
  int root = *root_ptr.typed_data();
  int count = buf.element_count();
  int err =
      EXLA_MPI_Bcast(buf.untyped_data(), count, datatype.value(), root, comm);
  return checkMpiError("MPI_Bcast", err);
}

XLA_FFI_DEFINE_HANDLER(MpiBcastFfi, MpiBcastImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>() // buf
                           .Arg<IntBuffer>()      // root
                           .Arg<MpiCommBuffer>()  // comm
);

void registerEnzymeJaXXLAHostMPIFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiCommRank", "Host",
                           MpiCommRankFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiCommSize", "Host",
                           MpiCommSizeFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiCommSplit", "Host",
                           MpiCommSplitFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiBarrier", "Host",
                           MpiBarrierFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiSend", "Host",
                           MpiSendFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiIsend", "Host",
                           MpiIsendFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiRecv", "Host",
                           MpiRecvFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiIrecv", "Host",
                           MpiIrecvFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiWait", "Host",
                           MpiWaitFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiWaitall", "Host",
                           MpiWaitallFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiAllreduce", "Host",
                           MpiAllreduceFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "MpiBcast", "Host",
                           MpiBcastFfi);
}

} // namespace enzymexla::ffi_internal
