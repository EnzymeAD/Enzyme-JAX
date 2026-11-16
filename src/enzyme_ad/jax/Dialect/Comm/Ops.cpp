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
