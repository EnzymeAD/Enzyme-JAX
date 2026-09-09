#include <cstdint>
#include <string_view>
#include <type_traits>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "src/enzyme_ad/jax/Runtime/jit/jit.h"
#include "src/enzyme_ad/jax/xla_ffi/export_macro.h"

#if defined(_WIN32)
void registerEnzymeJaXXLAHostMPIFFI() {}
#else

#include "mpi.h"

namespace enzymexla::ffi_internal {
namespace ffi = xla::ffi;
using namespace ::enzymexla;

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

template <typename T> ffi::ErrorOr<T> LookupSymbol(const char *name) {
  auto addr = enzymexla::LookupSymbol(name);
  if (!addr)
    return ffi::Error::Internal(
        absl::StrFormat("Failed to lookup symbol `%s`", name));
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<T>(addr.get());
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(reinterpret_cast<std::intptr_t>(addr.get()));
  } else {
    return ffi::Error::Internal(absl::StrFormat(
        "Unsupported type for symbol lookup: %s", typeid(T).name()));
  }
}

ffi::Error checkMpiStatusSize(const MpiStatusBuffer &buf) {
  auto mpi_status_size = LookupSymbol<int>("MPI_STATUS_SIZE");
  if (!mpi_status_size)
    return mpi_status_size.error();

  if (buf.element_count() != mpi_status_size.value()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Status buffer must have %d elements, got %d",
                        mpi_status_size.value(), buf.element_count()));
  }
  return ffi::Error::Success();
}

ffi::Error checkMpiError(const char *fname, const int err) {
  auto mpi_success = LookupSymbol<int>("MPI_SUCCESS");
  if (!mpi_success)
    return mpi_success.error();

  if (err == mpi_success.value())
    return ffi::Error::Success();

  auto mpi_max_error_string = LookupSymbol<int>("MPI_MAX_ERROR_STRING");
  if (!mpi_max_error_string)
    return mpi_max_error_string.error();

  auto fptr = LookupSymbol<decltype(MPI_Error_string) *>("MPI_Error_string");

  if (!fptr)
    return fptr.error();

  std::vector<char> cstr(mpi_max_error_string.value());
  int len;

  fptr.value()(err, cstr.data(), &len);

  std::string str(cstr.data(), len);
  return ffi::Error::InvalidArgument(
      absl::StrFormat("%s failed with error code %d: %s", fname, err, str));
}

// clang-format off
const char *
convertPrimitiveTypeToMpiDatatypeName(ffi::DataType type, bool allow_cast = false) {
  switch (type) {
    // case ffi::DataType::INVALID: return nullptr;
    case ffi::DataType::PRED: return "MPI_C_BOOL";
    // case ffi::DataType::S1: return nullptr;
    // case ffi::DataType::S2: return nullptr;
    // case ffi::DataType::S4: return nullptr;
    case ffi::DataType::S8: return "MPI_INT8_T";
    case ffi::DataType::S16: return "MPI_INT16_T";
    case ffi::DataType::S32: return "MPI_INT32_T";
    case ffi::DataType::S64: return "MPI_INT64_T";
    // case ffi::DataType::U1: return nullptr;
    // case ffi::DataType::U2: return nullptr;
    // case ffi::DataType::U4: return nullptr;
    case ffi::DataType::U8: return "MPI_UINT8_T";
    case ffi::DataType::U16: return "MPI_UINT16_T";
    case ffi::DataType::U32: return "MPI_UINT32_T";
    case ffi::DataType::U64: return "MPI_UINT64_T";
    case ffi::DataType::F16: return (allow_cast ? "MPI_UINT16_T" : nullptr);
    case ffi::DataType::F32: return "MPI_FLOAT";
    case ffi::DataType::F64: return "MPI_DOUBLE";
    case ffi::DataType::BF16: return (allow_cast ? "MPI_UINT16_T" : nullptr);
    case ffi::DataType::C64: return "MPI_C_FLOAT_COMPLEX";
    case ffi::DataType::C128: return "MPI_C_DOUBLE_COMPLEX";
    // case ffi::DataType::TOKEN: return nullptr;
    case ffi::DataType::F8E5M2: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E4M3: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E4M3FN: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E4M3B11FNUZ: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E5M2FNUZ: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E4M3FNUZ: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    case ffi::DataType::F8E3M4: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    // case ffi::DataType::F4E2M1FN: return nullptr;
    case ffi::DataType::F8E8M0FNU: return (allow_cast ? "MPI_UINT8_T" : nullptr);
    default: return nullptr;
  }
}
// clang-format on

ffi::ErrorOr<MPI_Datatype>
convertPrimitiveTypeToMpiDatatype(ffi::DataType type, bool allow_cast = false) {
  const char *name = convertPrimitiveTypeToMpiDatatypeName(type, allow_cast);
  if (name == nullptr) {
    std::ostringstream oss;
    oss << type;
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI: unsupported datatype `%s`", oss.str()));
  }

  auto dt = LookupSymbol<MPI_Datatype>(name);
  if (!dt)
    return dt.error();
  return dt.value();
}

ffi::Error MpiCommRankImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> rank_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Comm_rank) *>("MPI_Comm_rank");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = fptr.value()(comm, rank_ptr->typed_data());
  return checkMpiError("MPI_Comm_rank", err);
}

XLA_FFI_DEFINE_HANDLER(MpiCommRankFfi, MpiCommRankImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Ret<IntBuffer>()     // rank
);

ffi::Error MpiCommSizeImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> size_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Comm_size) *>("MPI_Comm_size");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = fptr.value()(comm, size_ptr->typed_data());
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
  auto fptr = LookupSymbol<decltype(MPI_Comm_split) *>("MPI_Comm_split");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int color = *color_ptr.typed_data();
  int key = *key_ptr.typed_data();
  MPI_Comm *newcomm = reinterpret_cast<MPI_Comm *>(newcomm_ptr->typed_data());
  int err = fptr.value()(comm, color, key, newcomm);
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
  auto fptr = LookupSymbol<decltype(MPI_Barrier) *>("MPI_Barrier");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = fptr.value()(comm);
  return checkMpiError("MPI_Barrier", err);
}

XLA_FFI_DEFINE_HANDLER(MpiBarrierFfi, MpiBarrierImpl,
                       xla::ffi::Ffi::Bind().Arg<MpiCommBuffer>());

ffi::Error MpiSendImpl(ffi::AnyBuffer buf, IntBuffer dest_ptr,
                       IntBuffer tag_ptr, MpiCommBuffer comm_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Send) *>("MPI_Send");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (datatype.has_error())
    return datatype.error();

  int err = fptr.value()(buf.untyped_data(), count, datatype.value(), dest, tag,
                         comm);
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
  auto fptr = LookupSymbol<decltype(MPI_Isend) *>("MPI_Isend");
  if (!fptr)
    return fptr.error();
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (datatype.has_error())
    return datatype.error();

  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = fptr.value()(buf.untyped_data(), count, datatype.value(), dest, tag,
                         comm, request);
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

ffi::Error MpiRecvImpl(IntBuffer source_ptr, IntBuffer tag_ptr,
                       MpiCommBuffer comm_ptr, Result<ffi::AnyBuffer> buf,
                       Result<MpiStatusBuffer> status_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Recv) *>("MPI_Recv");
  if (!fptr)
    return fptr.error();
  if (auto error = checkMpiStatusSize(*status_ptr); error.failure()) {
    return error;
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf->element_type());
  if (datatype.has_error())
    return datatype.error();

  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = fptr.value()(buf->untyped_data(), count, datatype.value(), source,
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

ffi::Error MpiIrecvImpl(IntBuffer source_ptr, IntBuffer tag_ptr,
                        MpiCommBuffer comm_ptr, Result<ffi::AnyBuffer> buf,
                        Result<MpiRequestBuffer> request_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Irecv) *>("MPI_Irecv");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf->element_type());
  if (datatype.has_error())
    return datatype.error();

  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = fptr.value()(buf->untyped_data(), count, datatype.value(), source,
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
  auto fptr = LookupSymbol<decltype(MPI_Wait) *>("MPI_Wait");
  if (!fptr)
    return fptr.error();

  if (auto error = checkMpiStatusSize(*status_ptr); error.failure())
    return error;

  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr.typed_data());
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = fptr.value()(request, status);
  return checkMpiError("MPI_Wait", err);
}

XLA_FFI_DEFINE_HANDLER(MpiWaitFfi, MpiWaitImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiRequestBuffer>() // request
                           .Ret<MpiStatusBuffer>()  // status
);

ffi::Error MpiWaitallImpl(ffi::RemainingArgs requests,
                          ffi::RemainingRets statuses) {
  auto fptr = LookupSymbol<decltype(MPI_Waitall) *>("MPI_Waitall");
  if (!fptr)
    return fptr.error();

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
  int err = fptr.value()(count, request_vector.data(), status_vector.data());
  auto error = checkMpiError("MPI_Waitall", err);
  if (error.failure())
    return error;

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
ffi::Error MpiAllreduceImpl(ffi::AnyBuffer sendbuf, std::string_view op_str,
                            MpiCommBuffer comm_ptr,
                            Result<ffi::AnyBuffer> recvbuf) {
  if (sendbuf.element_count() <= recvbuf->element_count()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Allreduce: recvbuf size (%d) must be at least "
                        "sendbuf size (%d)",
                        recvbuf->element_count(), sendbuf.element_count()));
  }
  if (sendbuf.element_type() != recvbuf->element_type()) {
    std::ostringstream oss_send, oss_recv;
    oss_send << sendbuf.element_type();
    oss_recv << recvbuf->element_type();
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Allreduce: sendbuf type (%s) must match recvbuf type (%s)",
        oss_send.str(), oss_recv.str()));
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(sendbuf.element_type());
  if (datatype.has_error())
    return datatype.error();

  auto op = LookupSymbol<MPI_Op>(op_str.data());
  if (!op)
    return op.error();

  auto fptr = LookupSymbol<decltype(MPI_Allreduce) *>("MPI_Allreduce");
  if (!fptr)
    return fptr.error();

  int count = sendbuf.element_count();
  int err = fptr.value()(sendbuf.untyped_data(), recvbuf->untyped_data(), count,
                         datatype.value(), op.value(), comm);
  return checkMpiError("MPI_Allreduce", err);
}

XLA_FFI_DEFINE_HANDLER(MpiAllreduceFfi, MpiAllreduceImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()        // sendbuf
                           .Attr<std::string_view>("op") // op
                           .Arg<MpiCommBuffer>()         // comm
                           .Ret<ffi::AnyBuffer>()        // recvbuf
);

ffi::Error MpiBcastImpl(ffi::AnyBuffer buf, IntBuffer root_ptr,
                        MpiCommBuffer comm_ptr) {
  auto fptr = LookupSymbol<decltype(MPI_Bcast) *>("MPI_Bcast");
  if (!fptr)
    return fptr.error();
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  auto datatype = convertPrimitiveTypeToMpiDatatype(buf.element_type());
  if (datatype.has_error())
    return datatype.error();
  int root = *root_ptr.typed_data();
  int count = buf.element_count();
  int err =
      fptr.value()(buf.untyped_data(), count, datatype.value(), root, comm);
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

#endif // _WIN32
