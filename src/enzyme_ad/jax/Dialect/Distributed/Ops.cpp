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
  if (dyn_cast_or_null<T>(symOp) == nullptr) {
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

// Printer/parser for GroupsplitOp branches
mlir::ParseResult parseSplitBranches(
    OpAsmParser &parser, mlir::ArrayAttr &branchAssignments,
    llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> &branchesRegions) {
  // Expect 0 or more `branch` $symbol_name $symbol_region
  // While next token is `branch`:
  llvm::SmallVector<mlir::Attribute, 2> assignment_symbols;
  while (parser.parseOptionalKeyword("branch").succeeded()) {
    // Parse symbol name
    mlir::SymbolRefAttr sym;
    auto sym_parse_failed = parser.parseAttribute<mlir::SymbolRefAttr>(sym);
    if (sym_parse_failed)
      return mlir::failure();
    assignment_symbols.push_back(sym);

    // Put placeholder region in list and parse into it
    branchesRegions.push_back(std::make_unique<mlir::Region>());
    auto parse_region_failed = parser.parseRegion(*branchesRegions.back());
    if (parse_region_failed)
      return mlir::failure();
  }

  branchAssignments = mlir::ArrayAttr::get(parser.getBuilder().getContext(),
                                           assignment_symbols);
  return mlir::success();
}

void printSplitBranches(OpAsmPrinter &printer, const GroupSplitOp &op,
                        const mlir::ArrayAttr branchAssignments,
                        const llvm::MutableArrayRef<mlir::Region> branches) {
  // Print each branch as `branch` $symbol_name $symbol_region
  for (size_t i = 0; i < branches.size(); i++) {
    printer << " branch ";
    printer.printAttribute(branchAssignments[i]);
    printer.printRegion(branches[i]);
  }
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