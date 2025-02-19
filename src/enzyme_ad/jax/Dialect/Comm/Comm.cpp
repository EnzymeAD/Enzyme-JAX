#include "src/enzyme_ad/jax/Dialect/Comm/Comm.h"

using namespace mlir::comm;

llvm::ArrayRef<int32_t> mlir::comm::getOpDevices(mlir::Operation &op) {
    auto parent_branch = op.getParentOfType<CommBranch>();
    return parent_branch.getDeviceIds();
}  