#include "Ops.h"
#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::comm;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/MPIOps.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/NCCLOps.cpp.inc"

OpFoldResult MpiConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

LogicalResult MpiSyncallOp::verify() {
  // `SameVariadicOperandSize` splits the operands evenly between requests and
  // buffers without checking that it can, so a trailing operand would
  // otherwise be silently ignored.
  auto numRequests = getRequests().size();
  auto numBuffers = getBuffers().size();
  auto numAliases = getAliasedBuffers().size();
  if (numRequests != numBuffers || getNumOperands() % 2 != 0)
    return emitOpError("expected one buffer per request, but got ")
           << numRequests << " requests and " << numBuffers << " buffers";
  else if (numAliases != numBuffers)
    return emitOpError(
               "expected one aliased output buffer per buffer, but got ")
           << numBuffers << " input buffers and " << numAliases
           << " aliased output buffers";
  return success();
}

LogicalResult NcclGroupOp::verify() {
  // `getBody()` is provided by SingleBlockImplicitTerminator.
  auto yield = cast<NcclYieldOp>(getBody()->getTerminator());
  if (!llvm::equal(yield.getValues().getTypes(), getResultTypes()))
    return emitOpError("expected yielded types ")
           << TypeRange(yield.getValues().getTypes())
           << " to match result types " << TypeRange(getResultTypes());
  // Which ops may appear in the body is left to the lowerings: during
  // legalize-mpi-to-nccl the body transiently holds comm.mpi ops.
  return success();
}
