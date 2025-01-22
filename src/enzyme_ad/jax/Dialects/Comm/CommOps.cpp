#include "mlir/Support/LogicalResult.h"
#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"

#include "llvm/ADT/DenseSet.h"


using namespace mlir;
using namespace mlir::comm;


LogicalResult CommSplit::verify() {
  for(Operation &op : getDeclarations().getOps()){
    // Check that all ops are allowable as members
    if (!op.hasTrait<SplitMemberOp>()){
      return op.emitOpError("not allowed as immediate split op member");
    }

    // check that all branches have disjoint device sets
    DenseSet<unsigned> used_devices;
    for(CommBranch branch : getBranches()){
      for(unsigned device : branch.getDeviceIds()){
        if (used_devices.contains(device)){
          return branch.emitError("uses device already accounted for in same split");
        }
        used_devices.insert(device);
      }
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"