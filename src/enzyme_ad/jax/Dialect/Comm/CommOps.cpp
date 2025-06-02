#include "mlir/Support/LogicalResult.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::comm;

LogicalResult CommSplit::verify() {

  // Checks that we contain only CommBranch and token declarations
  for (Operation &op : getDeclarations().getOps()) {
    // Check that all ops are allowable as members
    if (!op.hasTrait<SplitMemberOp>()) {
      return op.emitOpError("not allowed as immediate split op member");
    }
  }

  // check that all branches have disjoint device sets
  // check that loop branches only present if split is loopy
  DenseSet<unsigned> used_devices;
  for (CommBranch branch : getBranches()) {
    for (unsigned device : branch.getDeviceIds()) {
      if (used_devices.contains(device)) {
        return branch.emitError(
            "uses device already accounted for in same split");
      }
      used_devices.insert(device);
    }
  }
  
  return success();
}


LogicalResult CommBranch::verify() {
  return success();
}

LogicalResult CommToken::verify() {
  // Check that all users are TokenConsumer ops
  for (Operation *user : getToken().getUsers()) {
    if (!llvm::isa<TokenConsumer>(user)) {
      return emitOpError("token can only be used by TokenConsumer ops");
    }
  }

  return success();
}

// CommSend
LogicalResult CommSend::verify(){

  // Check that the token data type matches the value type
  auto val_type = getData().getType();

  // TODO replace with a "getTokenOp" method in the TokenConsumer interface
  auto token_op = llvm::cast<CommToken>(getToken().getDefiningOp());
  auto token_type = token_op.getDataType();
  if (val_type != token_type) {
    return emitOpError("Data type of send does not match token data type");
  }
  return success();
}

// CommControl
LogicalResult CommControl::verify() {
  // Assert only one op in the body
  if (getBody().empty()) {
    return emitOpError("expects a non-empty body region");
  }
  if (!llvm::hasSingleElement(getBody())) {
    return emitOpError("expects a single operation in the body region");
  }

  // TODO decide what class of control ops are acceptable

  return success();
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.cpp.inc"