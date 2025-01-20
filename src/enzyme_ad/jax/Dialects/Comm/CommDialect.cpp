#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.cpp.inc"

// Attr imports
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h" // for dbgs

using namespace mlir;
using namespace mlir::comm;

void CommunicationDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialects/Comm/CommOps.cpp.inc"
      >();

  // TODO types when we need them
}

/**
 * Attribute implemenation included is needed for the addAttributes<> template
 * to succeed, so for now it is easier to put attribute implementation stuff
 * here.
 */

/**
 * In its current form this almost certainly did not need to be a custom parser-
 * wrote this thinking I would add more to it.
 */
static ParseResult parseSplitBranchDescriptor(AsmParser &p,
                                              ArrayRef<unsigned> devices) {

  llvm::dbgs() << "Starting branch descriptor custom parse\n";

  llvm::SmallVector<unsigned> dev_set;
  do {
    // Do while since list shouldn't be empty
    unsigned &id = dev_set.emplace_back();
    auto parse_id = p.parseInteger(id);
    // Check for parse error
    if (parse_id) {
      return failure();
    }
    llvm::dbgs() << "Parsed " << id << "\n";
  } while (succeeded(p.parseOptionalComma()));
  llvm::dbgs() << "Ending branch descriptor custom parse\n";

  devices = dev_set;
  llvm::dbgs() << "Returning success\n";
  return success();
}

/// Print the case regions and values.
static void printSplitBranchDescriptor(AsmPrinter &p,
                                       llvm::ArrayRef<unsigned> devices) {
  // for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
  //   p.printNewline();
  //   p << "case " << value << ' ';
  //   p.printRegion(*region, /*printEntryBlockArgs=*/false);
  // }
}

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialects/Comm/CommAttrs.cpp.inc"