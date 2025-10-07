#include "Utils.h"
#include "Dialect.h"
namespace mlir::enzyme::distributed {

Region *getEnclosingDeviceParallelBranch(DeviceParallelOp parent,
                                         Operation *op) {
  auto region = op->getParentRegion();
  while (region->getParentOp() != parent) {
    auto region_parent =
        region->getParentOp();             // All regions have parent ops...
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

llvm::SmallVector<Token> getCorrespondingTokens(Token token) {
  unsigned idx = token.asBlockArg().getArgNumber();
  auto op = token.asBlockArg().getOwner()->getParentOp();
  DeviceParallelOp parent = llvm::cast<DeviceParallelOp>(op);
  llvm::SmallVector<Token> results;
  results.reserve(parent.getNumRegions());
  for (auto region : parent.getRegions()) {
    results.push_back(Token(region->getArgument(idx)));
  }
  return results;
}

llvm::SmallVector<mlir::Operation *> getTokenUsers(Token token) {
  auto all_tokens = getCorrespondingTokens(token);
  llvm::SmallVector<mlir::Operation *> results;
  // Concatenate all users of all corresponding tokens.
  // Due to scoping rules and since each token is a block arg to a
  // different region, there should be no duplicates here.
  for (auto t : all_tokens) {
    for (auto user : t.asBlockArg().getUsers()) {
      results.push_back(user);
    }
  }
  return results;
}

bool isSoleSender(TokenWriterOpInterface writer) {
  auto tokens = writer.getWriteTokens();
  // Check for conflicts on all tokens
  for (auto token : tokens) {
    auto users = getTokenUsers(token);
    if (!isSoleSender(writer, token, users)) {
      return false;
    }
  }
  return true;
}

bool isSoleSender(TokenWriterOpInterface writer, Token token,
                  llvm::ArrayRef<Operation *> others) {
  for (auto user : others) {
    TypedValue<TokenType> as_val = token.asTypedValue();
    TokenWriterOpInterface other = dyn_cast<TokenWriterOpInterface>(user);
    if (other && other != writer) {
      // Found another writer using the same token. Check if it uses
      // the token to write, or only for something else:
      auto other_write_tokens = other.getWriteTokens();
      for (auto t : other_write_tokens) {
        if (t == as_val) {
          return false; // Found another op writing to the same token
        }
      }
    }
  }
  return true;
}
} // namespace mlir::enzyme::distributed