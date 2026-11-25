#ifndef ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Utils.td"
include "Types.td"
include "Attrs.td"

def MpiConstantOp : Comm_Op<"mpi.constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "Creates a value that represents a MPI constant";
  let arguments = (ins TypedAttrInterface : $value);
  let results = (outs AnyTypeOf<[MpiComm, MpiOp]> : $result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def MpiCommRankOp : Comm_Op<"mpi.comm_rank", [Pure]> {
  let summary = "Equivalent to `MPI_Comm_rank`";
  let arguments = (ins MpiComm : $comm);
  let results = (outs TensorI32 : $rank);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def MpiCommSizeOp : Comm_Op<"mpi.comm_size", [Pure]> {
  let summary = "Equivalent to `MPI_Comm_size`";
  let arguments = (ins MpiComm : $comm);
  let results = (outs TensorI32 : $size);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def MpiCommSplitOp : Comm_Op<"mpi.comm_split", []> {
  let summary = "Equivalent to `MPI_Comm_split`";

  let arguments = (ins
    MpiComm : $comm,
    TensorI32 : $color,
    TensorI32 : $key
  );

  let results = (outs MpiComm : $newcomm);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MpiBarrierOp : Comm_Op<"mpi.barrier", []> {
  let summary = "Equivalent to `MPI_Barrier`";
  let arguments = (ins MpiComm : $comm);
  let assemblyFormat = "$comm attr-dict `:` type(operands)";
}

def MpiSendOp : Comm_Op<"mpi.send", []> {
  let summary = "Equivalent to `MPI_Send`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buffer,
    TensorI32 : $dest,
    TensorI32 : $tag,
    MpiComm : $comm
  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def MpiIsendOp : Comm_Op<"mpi.isend", []> {
  let summary = "Equivalent to `MPI_Isend`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buffer,
    TensorI32 : $dest,
    TensorI32 : $tag,
    MpiComm : $comm
  );

  let results = (outs MpiRequest : $request);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MpiRecvOp : Comm_Op<"mpi.recv", []> {
  let summary = "Equivalent to `MPI_Recv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    TensorI32 : $source,
    TensorI32 : $tag,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $buffer);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MpiIrecvOp : Comm_Op<"mpi.irecv", []> {
  let summary = "Equivalent to `MPI_Irecv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    TensorI32 : $source,
    TensorI32 : $tag,
    MpiComm : $comm
  );
  let results = (outs
    AnyTensor : $buffer,
    MpiRequest : $request
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MpiWaitOp : Comm_Op<"mpi.wait", []> {
  let summary = "Equivalent to `MPI_Wait`";
  let arguments = (ins MpiRequest : $request);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def MpiWaitallOp : Comm_Op<"mpi.waitall", []> {
  let summary = "Equivalent to `MPI_Waitall`";
  let arguments = (ins Variadic<MpiRequest> : $requests);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def MpiAllreduceOp : Comm_Op<"mpi.allreduce", [AllTypesMatch<["sendbuf", "recvbuf"]>]> {
  let summary = "Equivalent to `MPI_Allreduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    MpiOpAttr : $reduceOp,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOp `,` $comm attr-dict `:` functional-type(operands, results)";
}

def MpiBcastOp : Comm_Op<"mpi.bcast", [AllTypesMatch<["inBuffer", "outBuffer"]>]> {
  let summary = "Equivalent to `MPI_Bcast`";

  let arguments = (ins
    AnyTensor : $inBuffer,
    TensorI32 : $root,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $outBuffer);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MpiSyncOp : Comm_Op<"mpi.sync", [AllTypesMatch<["buffer", "aliasedBuffer"]>]> {
  let summary = "Equivalent to `MPI_Wait` but aliases the input buffer to the output buffer";
  let description = [{
    This operation is a variant of `mpi.wait` that takes an input buffer and a request, and produces an output buffer that aliases the input buffer.
    This is required for guarding the use of the buffer after the wait, since the buffer is not guaranteed to be valid before the wait operation.
  }];
  let arguments = (ins
    MpiRequest : $request,
    AnyTensor : $buffer
  );
  let results = (outs AnyTensor : $aliasedBuffer);
  let assemblyFormat = "$request `,` $buffer attr-dict `:` type($buffer)";
}

def MpiSyncallOp : Comm_Op<"mpi.syncall", [SameVariadicOperandSize, AllTypesMatch<["buffers", "aliasedBuffers"]>]> {
  let summary = "Equivalent to `MPI_Waitall` but aliases the input buffers to the output buffers";
  let description = [{
    This operation is a variant of `mpi.wait` that takes an input buffer and a request, and produces an output buffer that aliases the input buffer.
    This is required for guarding the use of the buffer after the wait, since the buffer is not guaranteed to be valid before the wait operation.
  }];
  let arguments = (ins
    Variadic<MpiRequest> : $requests,
    Variadic<AnyTensor> : $buffers
  );
  let results = (outs Variadic<AnyTensor> : $aliasedBuffers);
  let assemblyFormat = "`requests` `=` `(` $requests `)` `,` `buffers` `=` `(` $buffers `)` attr-dict `:` type($buffers)";
  let hasVerifier = 1;
}

#endif // ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
