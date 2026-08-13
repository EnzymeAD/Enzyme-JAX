#include <type_traits>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "mpi.h"

#include "../export_macro.h"

// MPI function pointers are initialized to the MPItrampoline implementations by
// default, but can be overridden by the user
decltype(MPI_Comm_rank) *EXLA_MPI_Comm_rank = &MPI_Comm_rank;
decltype(MPI_Comm_size) *EXLA_MPI_Comm_size = &MPI_Comm_size;
decltype(MPI_Comm_split) *EXLA_MPI_Comm_split = &MPI_Comm_split;
decltype(MPI_Barrier) *EXLA_MPI_Barrier = &MPI_Barrier;
decltype(MPI_Send) *EXLA_MPI_Send = &MPI_Send;
decltype(MPI_Isend) *EXLA_MPI_Isend = &MPI_Isend;
decltype(MPI_Recv) *EXLA_MPI_Recv = &MPI_Recv;
decltype(MPI_Irecv) *EXLA_MPI_Irecv = &MPI_Irecv;
decltype(MPI_Wait) *EXLA_MPI_Wait = &MPI_Wait;
decltype(MPI_Waitall) *EXLA_MPI_Waitall = &MPI_Waitall;
decltype(MPI_Allreduce) *EXLA_MPI_Allreduce = &MPI_Allreduce;
decltype(MPI_Bcast) *EXLA_MPI_Bcast = &MPI_Bcast;

size_t EXLA_MPI_STATUS_SIZE = MPI_STATUS_SIZE;

extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_comm_rank(void *ptr) {
  EXLA_MPI_Comm_rank = reinterpret_cast<decltype(MPI_Comm_rank) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_comm_size(void *ptr) {
  EXLA_MPI_Comm_size = reinterpret_cast<decltype(MPI_Comm_size) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_comm_split(void *ptr) {
  EXLA_MPI_Comm_split = reinterpret_cast<decltype(MPI_Comm_split) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_barrier(void *ptr) {
  EXLA_MPI_Barrier = reinterpret_cast<decltype(MPI_Barrier) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_send(void *ptr) {
  EXLA_MPI_Send = reinterpret_cast<decltype(MPI_Send) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_isend(void *ptr) {
  EXLA_MPI_Isend = reinterpret_cast<decltype(MPI_Isend) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_recv(void *ptr) {
  EXLA_MPI_Recv = reinterpret_cast<decltype(MPI_Recv) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_irecv(void *ptr) {
  EXLA_MPI_Irecv = reinterpret_cast<decltype(MPI_Irecv) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_wait(void *ptr) {
  EXLA_MPI_Wait = reinterpret_cast<decltype(MPI_Wait) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_waitall(void *ptr) {
  EXLA_MPI_Waitall = reinterpret_cast<decltype(MPI_Waitall) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_allreduce(void *ptr) {
  EXLA_MPI_Allreduce = reinterpret_cast<decltype(MPI_Allreduce) *>(ptr);
}
extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_bcast(void *ptr) {
  EXLA_MPI_Bcast = reinterpret_cast<decltype(MPI_Bcast) *>(ptr);
}

extern "C" MLIR_CAPI_EXPORTED void enzymexla_set_mpi_status_size(size_t size) {
  EXLA_MPI_STATUS_SIZE = size;
}

namespace enzymexla::ffi_internal {
namespace ffi = xla::ffi;

using ffi::Buffer, ffi::AnyBuffer;
using ffi::Result, ffi::RemainingArgs, ffi::RemainingRets;

using IntBuffer = Buffer<ffi::S32, 0>;

// pointers, so use U64
using MpiCommBuffer = Buffer<ffi::U64, 0>;
using MpiDatatypeBuffer = Buffer<ffi::U64, 0>;
using MpiOpBuffer = Buffer<ffi::U64, 0>;
using MpiRequestBuffer = Buffer<ffi::U64, 0>;

// MPI_Status is a non-ABI-stable struct, so use U8 x N buffer to hold it
// its size is platform-dependent and given by MPI_STATUS_SIZE
using MpiStatusBuffer = Buffer<ffi::U8, 1>;

ffi::Error checkMpiStatusSize(const MpiStatusBuffer &buf) {
  if (buf.element_count() != EXLA_MPI_STATUS_SIZE) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Recv: status buffer must have %d elements, got %d",
                        EXLA_MPI_STATUS_SIZE, buf.element_count()));
  }
  return ffi::Error::Success();
}

ffi::Error MpiCommRankImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> rank_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Comm_rank(comm, rank_ptr->typed_data());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_rank failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiCommRankFfi, MpiCommRankImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Ret<IntBuffer>()     // rank
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "enzymexla_ffi_mpi_comm_rank", "Host", MpiCommRankFfi);

ffi::Error MpiCommSizeImpl(MpiCommBuffer comm_ptr, Result<IntBuffer> size_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Comm_size(comm, size_ptr->typed_data());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_size failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiCommSizeFfi, MpiCommSizeImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Ret<IntBuffer>()     // size
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "enzymexla_ffi_mpi_comm_size", "Host", MpiCommSizeFfi);

ffi::Error MpiCommSplitImpl(MpiCommBuffer comm_ptr, IntBuffer color_ptr,
                            IntBuffer key_ptr,
                            Result<MpiCommBuffer> newcomm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int color = *color_ptr.typed_data();
  int key = *key_ptr.typed_data();
  MPI_Comm *newcomm = reinterpret_cast<MPI_Comm *>(newcomm_ptr->typed_data());
  int err = EXLA_MPI_Comm_split(comm, color, key, newcomm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_split failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiCommSplitFfi, MpiCommSplitImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiCommBuffer>() // comm
                           .Arg<IntBuffer>()     // color
                           .Arg<IntBuffer>()     // key
                           .Ret<MpiCommBuffer>() // newcomm
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "enzymexla_ffi_mpi_comm_split", "Host",
                         MpiCommSplitFfi);

ffi::Error MpiBarrierImpl(MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int err = EXLA_MPI_Barrier(comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Barrier failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiBarrierFfi, MpiBarrierImpl,
                       xla::ffi::Ffi::Bind().Arg<MpiCommBuffer>());
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_barrier",
                         "Host", MpiBarrierFfi);

ffi::Error MpiSendImpl(ffi::AnyBuffer buf, MpiDatatypeBuffer datatype_ptr,
                       IntBuffer dest_ptr, IntBuffer tag_ptr,
                       MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  int err = EXLA_MPI_Send(buf.untyped_data(), count, datatype, dest, tag, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Send failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiSendFfi, MpiSendImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()    // buf
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<IntBuffer>()         // dest
                           .Arg<IntBuffer>()         // tag
                           .Arg<MpiCommBuffer>()     // comm
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_send",
                         "Host", MpiSendFfi);

ffi::Error MpiIsendImpl(ffi::AnyBuffer buf, MpiDatatypeBuffer datatype_ptr,
                        IntBuffer dest_ptr, IntBuffer tag_ptr,
                        MpiCommBuffer comm_ptr,
                        Result<MpiRequestBuffer> request_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  int dest = *dest_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf.element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = EXLA_MPI_Isend(buf.untyped_data(), count, datatype, dest, tag, comm,
                           request);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Isend failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiIsendFfi, MpiIsendImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()    // buf
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<IntBuffer>()         // dest
                           .Arg<IntBuffer>()         // tag
                           .Arg<MpiCommBuffer>()     // comm
                           .Ret<MpiRequestBuffer>()  // request
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_isend",
                         "Host", MpiIsendFfi);

ffi::Error MpiRecvImpl(MpiDatatypeBuffer datatype_ptr, IntBuffer source_ptr,
                       IntBuffer tag_ptr, MpiCommBuffer comm_ptr,
                       Result<ffi::AnyBuffer> buf,
                       Result<MpiStatusBuffer> status_ptr) {
  if (auto error = checkMpiStatusSize(*status_ptr); error.failure()) {
    return error;
  }
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = EXLA_MPI_Recv(buf->untyped_data(), count, datatype, source, tag,
                          comm, status);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Recv failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiRecvFfi, MpiRecvImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<IntBuffer>()         // source
                           .Arg<IntBuffer>()         // tag
                           .Arg<MpiCommBuffer>()     // comm
                           .Ret<ffi::AnyBuffer>()    // buf
                           .Ret<MpiStatusBuffer>()   // status
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_recv",
                         "Host", MpiRecvFfi);

ffi::Error MpiIrecvImpl(MpiDatatypeBuffer datatype_ptr, IntBuffer source_ptr,
                        IntBuffer tag_ptr, MpiCommBuffer comm_ptr,
                        Result<ffi::AnyBuffer> buf,
                        Result<MpiRequestBuffer> request_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  int source = *source_ptr.typed_data();
  int tag = *tag_ptr.typed_data();
  int count = buf->element_count();
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr->typed_data());
  int err = EXLA_MPI_Irecv(buf->untyped_data(), count, datatype, source, tag,
                           comm, request);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Irecv failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiIrecvFfi, MpiIrecvImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<IntBuffer>()         // source
                           .Arg<IntBuffer>()         // tag
                           .Arg<MpiCommBuffer>()     // comm
                           .Ret<ffi::AnyBuffer>()    // buf
                           .Ret<MpiRequestBuffer>()  // request
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_irecv",
                         "Host", MpiIrecvFfi);

ffi::Error MpiWaitImpl(MpiRequestBuffer request_ptr,
                       Result<MpiStatusBuffer> status_ptr) {
  if (auto error = checkMpiStatusSize(*status_ptr); error.failure()) {
    return error;
  }
  MPI_Request *request =
      reinterpret_cast<MPI_Request *>(request_ptr.typed_data());
  MPI_Status *status = reinterpret_cast<MPI_Status *>(status_ptr->typed_data());
  int err = EXLA_MPI_Wait(request, status);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Wait failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiWaitFfi, MpiWaitImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MpiRequestBuffer>() // request
                           .Ret<MpiStatusBuffer>()  // status
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_wait",
                         "Host", MpiWaitFfi);

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
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Waitall failed with error code %d", err));
  }

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
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_waitall",
                         "Host", MpiWaitallFfi);

ffi::Error MpiAllreduceImpl(ffi::AnyBuffer sendbuf,
                            MpiDatatypeBuffer datatype_ptr, MpiOpBuffer op_ptr,
                            MpiCommBuffer comm_ptr,
                            Result<ffi::AnyBuffer> recvbuf) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  MPI_Op op = *reinterpret_cast<MPI_Op *>(op_ptr.typed_data());
  if (sendbuf.element_count() <= recvbuf->element_count()) {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Allreduce: recvbuf size (%d) must be at least sendbuf size (%d)",
        recvbuf->element_count(), sendbuf.element_count()));
  }
  int count = sendbuf.element_count();
  int err = EXLA_MPI_Allreduce(sendbuf.untyped_data(), recvbuf->untyped_data(),
                               count, datatype, op, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Allreduce failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiAllreduceFfi, MpiAllreduceImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()    // sendbuf
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<MpiOpBuffer>()       // op
                           .Arg<MpiCommBuffer>()     // comm
                           .Ret<ffi::AnyBuffer>()    // recvbuf
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "enzymexla_ffi_mpi_allreduce", "Host",
                         MpiAllreduceFfi);

ffi::Error MpiBcastImpl(ffi::AnyBuffer buf, MpiDatatypeBuffer datatype_ptr,
                        IntBuffer root_ptr, MpiCommBuffer comm_ptr) {
  MPI_Comm comm = *reinterpret_cast<MPI_Comm *>(comm_ptr.typed_data());
  MPI_Datatype datatype =
      *reinterpret_cast<MPI_Datatype *>(datatype_ptr.typed_data());
  int root = *root_ptr.typed_data();
  int count = buf.element_count();
  int err = EXLA_MPI_Bcast(buf.untyped_data(), count, datatype, root, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Bcast failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MpiBcastFfi, MpiBcastImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()    // buf
                           .Arg<MpiDatatypeBuffer>() // datatype
                           .Arg<IntBuffer>()         // root
                           .Arg<MpiCommBuffer>()     // comm
);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_bcast",
                         "Host", MpiBcastFfi);

} // namespace enzymexla::ffi_internal
