#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme::perfify;

namespace mlir::enzyme::perfify {} // namespace mlir::enzyme::perfify

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"


// TODO: custom parser for internal region
/*
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
 */