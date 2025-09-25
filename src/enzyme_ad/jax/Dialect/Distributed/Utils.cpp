#include "Utils.h"
namespace mlir::enzyme::distributed {
Region *getEnclosingDeviceParallelBranch(DeviceParallelOp parent,
                                         Operation *op) {
  auto region = op->getParentRegion();
  while (region->getParentOp() != parent) {
    auto region_parent =
        region->getParentOp();             // All regoins have parent ops...
    if (!region_parent->getParentRegion()) // But not all ops have parent
                                           // regions (e.g. top level ops)
      return nullptr;
    region = region_parent->getParentRegion();
  }
  return region;
}

int getDeviceParallelBranchIndex(DeviceParallelOp parent, Region *branch) {
  assert(branch->getParentOp() == parent && "branch is not a region of parent");
  for (int i = 0; i < parent.getNumRegions(); i++) {
    if (&parent.getRegion(i) == branch)
      return i;
  }
  llvm_unreachable("branch not found in parent regions");
  return -1;
}

mlir::Operation *getExecutingDevice(mlir::Operation *op) {
  // Find current branch
  auto parent = op->getParentOfType<DeviceParallelOp>();
  auto branch = getEnclosingDeviceParallelBranch(parent, op);
  if (!branch)
    return nullptr;
  // Find index of branch and cross-reference to parent device symbol
  int branch_idx = getDeviceParallelBranchIndex(parent, branch);
  auto device_sym = llvm::cast<mlir::SymbolRefAttr>(
      parent.getBranchAssignments()[branch_idx]);

  return SymbolTable::lookupNearestSymbolFrom(parent, device_sym);
}

llvm::SmallVector<mlir::BlockArgument>
getCorrespondingTokens(mlir::BlockArgument token) {
  unsigned idx = token.getArgNumber();
  auto op = token.getOwner()->getParentOp();
  DeviceParallelOp parent = llvm::cast<DeviceParallelOp>(op);
  llvm::SmallVector<mlir::BlockArgument> results;
  results.reserve(parent.getNumRegions());
  for (auto region : parent.getRegions()) {
    results.push_back(region->getArgument(idx));
  }
  return results;
}

llvm::SmallVector<mlir::Operation *> getTokenUsers(mlir::BlockArgument token) {
  llvm::SmallVector<mlir::Operation *, 4> results;
  for (auto user : token.getUsers()) {
    results.push_back(user);
  }
  return results;
}

} // namespace mlir::enzyme::distributed