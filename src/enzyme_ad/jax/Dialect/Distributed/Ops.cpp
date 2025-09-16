#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"

using mlir::OpTrait::enzyme::distributed::ChannelDefTrait;
using mlir::OpTrait::enzyme::distributed::DeviceDefTrait;
namespace mlir::enzyme::distributed {

/**
 * Returns success if the symbol reference attribute refers to an operation of
 * type T in the given symbol table.
 */
template <typename T>
llvm::LogicalResult checkSymbolIsA(::mlir::SymbolTableCollection &symbol_table,
                                   mlir::Operation *owning_op,
                                   mlir::SymbolRefAttr attr) {
  auto loc = owning_op->getLoc();
  Operation *symOp = symbol_table.lookupNearestSymbolFrom(owning_op, attr);
  if (!symOp || !isa<T>(symOp)) {
    mlir::emitError(loc) << "invalid symbol reference or symbol type: " << attr;
    return mlir::failure();
  }
  return mlir::success();
}

/**
 * Returns success if the attribute is a symbol reference to an operation with
 * the given trait in the provided symbol table.
 */
template <template <class T> class ValidTrait>
llvm::LogicalResult
checkSymbolHasTrait(::mlir::SymbolTableCollection &symbol_table,
                    mlir::Operation *owning_op, mlir::Attribute attr) {
  auto loc = owning_op->getLoc();
  // Check it's a symbol ref
  auto symRef = dyn_cast<mlir::SymbolRefAttr>(attr);
  if (!symRef) {
    mlir::emitError(loc) << "expected symbol reference";
    return mlir::failure();
  }
  Operation *symOp = symbol_table.lookupNearestSymbolFrom(owning_op, symRef);
  if (!symOp || !symOp->hasTrait<ValidTrait>()) {
    mlir::emitError(loc) << "invalid symbol reference or symbol trait: "
                         << symRef;
    return mlir::failure();
  }
  return mlir::success();
}

/**
 * Returns success if all elements of the listAttr are valid symbol references
 * in the given symbol table to ops that conform to SymbolOpType.
 */
template <template <class T> class ValidTrait>
llvm::LogicalResult
isValidSymbolList(::mlir::SymbolTableCollection &symbol_table,
                  mlir::Operation *owning_op, mlir::ArrayAttr listAttr) {
  for (mlir::Attribute attr : listAttr) {
    auto res = checkSymbolHasTrait<ValidTrait>(symbol_table, owning_op, attr);
    if (mlir::failed(res))
      return res;
  }
  return mlir::success();
}

LogicalResult mlir::enzyme::distributed::ChannelOp::verifySymbolUses(
    ::mlir::SymbolTableCollection &symbolTable) {
  // Check that sending, receiving devices are arrays of references to device
  // definitions
  auto res = isValidSymbolList<DeviceDefTrait>(symbolTable, *this,
                                               getSendingDevices());
  if (mlir::failed(res))
    return res;
  res = isValidSymbolList<DeviceDefTrait>(symbolTable, *this,
                                          getReceivingDevices());
  if (mlir::failed(res))
    return res;

  return mlir::success();
}

LogicalResult
DeviceGroupOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Check producing ops for device and channel definitions
  auto res =
      isValidSymbolList<DeviceDefTrait>(symbol_table, *this, getDevices());
  if (mlir::failed(res))
    return res;
  res = isValidSymbolList<ChannelDefTrait>(symbol_table, *this, getChannels());
  return res;
}

LogicalResult
DeviceMeshOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Mesh device template should be a device
  return checkSymbolHasTrait<DeviceDefTrait>(symbol_table, *this,
                                             getDeviceType());
}

LogicalResult
MeshForOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Mesh for ops apply only to meshes
  return checkSymbolIsA<DeviceMeshOp>(symbol_table, *this, getMeshAttr());
}

LogicalResult
GroupSplitOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Group splits apply only to device groups
  return checkSymbolIsA<DeviceGroupOp>(symbol_table, *this,
                                       getDeviceGroupAttr());
}

LogicalResult
SplitBranchOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Split branches have programs for individual devices or channels
  Operation *dev_or_chan =
      symbol_table.lookupNearestSymbolFrom(*this, getDeviceOrChannelAttr());
  if (!dev_or_chan || !(dev_or_chan->hasTrait<DeviceDefTrait>() ||
                        dev_or_chan->hasTrait<ChannelDefTrait>())) {
    mlir::emitError(getLoc())
        << "branches must reference a valid device or channel";
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult
DefineTokenOp::verifySymbolUses(::mlir::SymbolTableCollection &symbol_table) {
  // Tokens need to indicate which channel they communicate over
  return checkSymbolHasTrait<ChannelDefTrait>(symbol_table, *this,
                                              getChannelAttr());
}

} // namespace mlir::enzyme::distributed
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"