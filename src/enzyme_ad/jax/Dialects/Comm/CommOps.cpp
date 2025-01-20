#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "llvm/Support/Debug.h" // for dbgs

using namespace mlir;
using namespace mlir::comm;

// Parsing and printing for the split op branches. Modeled after the SCF
// switchcase parsing code ()
static ParseResult
parseSplitBranch(OpAsmParser &p, mlir::ArrayAttr &branch_descriptors,
                 SmallVectorImpl<std::unique_ptr<Region>> &branches) {
  SmallVector<Attribute> branch_desc_values;
  llvm::dbgs() << "Looking for split branch\n";
  while (succeeded(p.parseOptionalKeyword("branch"))) {
    llvm::dbgs() << "Found branch kw\n";
    Region &region = *branches.emplace_back(std::make_unique<Region>());

    llvm::dbgs() << "Parsing branch description...\n";

    SplitBranchDescriptorAttr branch_descriptor;
    auto descriptor_parse_flag =
        p.parseCustomAttributeWithFallback(branch_descriptor);

    llvm::dbgs() << "... done\n";
    if (!descriptor_parse_flag &&
        branch_descriptor.isa<SplitBranchDescriptorAttr>()) {
      branch_desc_values.push_back(
          branch_descriptor.cast<SplitBranchDescriptorAttr>());
    } else {
      llvm::dbgs() << "Failed to parse branch descriptor\n";
      return failure();
    }

    llvm::dbgs() << "Parsing branch region...\n";
    auto parse_region = p.parseRegion(region, /*arguments=*/{});
    if (parse_region) {
      llvm::dbgs() << "Failed to parse branch region\n";
      return failure();
    }
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