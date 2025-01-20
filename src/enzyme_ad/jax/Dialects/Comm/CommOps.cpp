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
    Region &region = *branches.emplace_back(std::make_unique<Region>());

    SplitBranchDescriptorAttr branch_descriptor;
    auto descriptor_parse_flag =
        p.parseCustomAttributeWithFallback(branch_descriptor);

    if (!descriptor_parse_flag &&
        branch_descriptor.isa<SplitBranchDescriptorAttr>()) {
      branch_desc_values.push_back(
          branch_descriptor.cast<SplitBranchDescriptorAttr>());
    } else {
      return failure();
    }

    auto parse_region = p.parseRegion(region, /*arguments=*/{});
    if (parse_region) {
      return failure();
    }
  }
  branch_descriptors = p.getBuilder().getArrayAttr(branch_desc_values);
  return success();
}

/// Print the case regions and values.
static void printSplitBranch(OpAsmPrinter &p, Operation *op,
                             mlir::ArrayAttr cases, RegionRange caseRegions) {
  p.increaseIndent();
  for (auto [descriptor, region] : llvm::zip(cases, caseRegions)) {
    p.printNewline();
    p << "branch ";
    descriptor.cast<SplitBranchDescriptorAttr>().print(p);
    p << " ";
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
  p.decreaseIndent();
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"