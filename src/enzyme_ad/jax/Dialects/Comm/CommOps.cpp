#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"

using namespace mlir;
using namespace mlir::comm;

// Parsing and printing for the split op branches. Modeled after the SCF
// switchcase parsing code ()
static ParseResult
parseSplitBranch(OpAsmParser &p, mlir::ArrayAttr &branch_descriptors,
                 SmallVectorImpl<std::unique_ptr<Region>> &branches) {
  SmallVector<Attribute> branch_desc_values;
  while (succeeded(p.parseOptionalKeyword("branch"))) {
    SplitBranchDescriptorAttr branch_descriptor;
    Region &region = *branches.emplace_back(std::make_unique<Region>());
    if (p.parseAttribute(branch_descriptor) ||
        p.parseRegion(region, /*arguments=*/{}))
      return failure();
    branch_desc_values.push_back(branch_descriptor);
  }
  branch_descriptors = p.getBuilder().getArrayAttr(branch_desc_values);
  return success();
}

/// Print the case regions and values.
static void printSplitBranch(OpAsmPrinter &p, Operation *op,
                             mlir::ArrayAttr cases, RegionRange caseRegions) {
  // for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
  //   p.printNewline();
  //   p << "case " << value << ' ';
  //   p.printRegion(*region, /*printEntryBlockArgs=*/false);
  // }
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"