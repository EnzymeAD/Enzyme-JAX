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

// def MpiPredefinedCommOp : Comm_Op<"mpi.predefined_comm", [Pure]> {
//   let summary = "Get the named predefined MPI communicator";
//   let arguments = (ins MpiCommsAttr : $value);
//   let results = (outs MpiComm : $comm);
//   let assemblyFormat = "attr-dict `:` type(results)";
// }

// def MpiPredefinedOpOp : Comm_Op<"mpi.predefined_op", [Pure]> {
//   let summary = "Get the named predefined MPI reduction operator";
//   let arguments = (ins MpiOpsAttr : $value);
//   let results = (outs MpiOps : $op);
//   let assemblyFormat = "attr-dict `:` type(results)";
// }

// NOTE not sure if necessary since we take the datatype from the MLIR tensor eltype
// def MpiPredefinedDatatypeOp : Comm_Op<"mpi.predefined_datatype", [Pure]> {
//   let summary = "Get the named predefined MPI datatype";
//   let arguments = (ins MpiDatatypesAttr : $value);
//   let results = (outs MpiDatatype : $datatype);
//   let assemblyFormat = "attr-dict `:` type(results)";
// }

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
  let assemblyFormat = "operands attr-dict `:` functional-type($comm, results)";
}

def MpiBarrierOp : Comm_Op<"mpi.barrier", []> {
  let summary = "Equivalent to `MPI_Barrier`";
  let arguments = (ins MpiComm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def MpiSendOp : Comm_Op<"mpi.send", []> {
  let summary = "Equivalent to `MPI_Send`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buf,
    TensorI32 : $tag,
    TensorI32 : $dest,
    MpiComm : $comm
  );

  let assemblyFormat = "operands attr-dict `:` type($buf)";
}

def MpiIsendOp : Comm_Op<"mpi.isend", []> {
  let summary = "Equivalent to `MPI_Isend`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buf,
    TensorI32 : $tag,
    TensorI32 : $dest,
    MpiComm : $comm
  );

  let results = (outs MpiRequest : $request);
  let assemblyFormat = "operands attr-dict `:` functional-type($buf, results)";
}

def MpiRecvOp : Comm_Op<"mpi.recv", []> {
  let summary = "Equivalent to `MPI_Recv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    TensorI32 : $tag,
    TensorI32 : $source,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $buf);
  let assemblyFormat = "operands attr-dict `:` type(results)";
}

def MpiIrecvOp : Comm_Op<"mpi.irecv", []> {
  let summary = "Equivalent to `MPI_Irecv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    TensorI32 : $tag,
    TensorI32 : $source,
    MpiComm : $comm
  );
  let results = (outs
    AnyTensor : $buf,
    MpiRequest : $request
  );
  let assemblyFormat = "operands attr-dict `:` type(results)";
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

def MpiAllreduceOp : Comm_Op<"mpi.allreduce", []> {
  let summary = "Equivalent to `MPI_Allreduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    MpiOpsAttr : $reduceOpAttr,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOpAttr `,` $comm attr-dict `:` functional-type($sendbuf, results)";
}

def MpiBcastOp : Comm_Op<"mpi.bcast", []> {
  let summary = "Equivalent to `MPI_Bcast`";

  let arguments = (ins
    AnyTensor : $inbuf,
    TensorI32 : $root,
    MpiComm : $comm
  );

  let results = (outs AnyTensor : $outbuf);
  let assemblyFormat = "operands attr-dict `:` functional-type($inbuf, results)";
}

#endif // ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
