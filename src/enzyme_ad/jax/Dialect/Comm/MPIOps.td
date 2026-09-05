#ifndef ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Types.td"
include "Attrs.td"

def TensorI32
    : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && "
                 "::llvm::cast<::mlir::TensorType>($_self).getShape().size() "
                 "== 0 && "
                 "::llvm::cast<::mlir::TensorType>($_self).getElementType()."
                 "isSignlessInteger(32)">,
           "tensor<i32>", "::mlir::TensorType">,
      BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(32))">;

def MpiConstantOp : Comm_Op<"mpi.constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "Creates a value that represents a MPI constant";
  let arguments = (ins TypedAttrInterface : $value);
  let results = (outs AnyTypeOf<[MpiComm, MpiOp]> : $result);
  let assemblyFormat = "attr-dict $value";
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

#endif // ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
