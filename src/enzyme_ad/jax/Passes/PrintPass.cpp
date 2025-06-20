//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PRINTPASS
#define GEN_PASS_DEF_PRINTLOCATIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct PrintPass : public enzyme::impl::PrintPassBase<PrintPass> {
  using PrintPassBase::PrintPassBase;

  void runOnOperation() override {

    OpPrintingFlags flags;
    if (debug)
      flags.enableDebugInfo(true, /*pretty*/ false);
    if (generic)
      flags.printGenericOpForm(true);
    if (!filename.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream f(filename, EC);
      if (EC) {
        llvm::errs() << "Opening output file failed\n";
        return signalPassFailure();
      }
      getOperation()->print(f, flags);
      f.flush();
    } else if (use_stdout) {
      getOperation()->print(llvm::outs(), flags);
      llvm::outs() << "\n";
    } else {
      getOperation()->print(llvm::errs(), flags);
      llvm::errs() << "\n";
    }
  }
};

static llvm::raw_ostream &printMetadata(llvm::raw_ostream &os, Attribute attr) {
  if (auto diSubprogram = dyn_cast<mlir::LLVM::DISubprogramAttr>(attr)) {
    os << "@" << diSubprogram.getName().getValue() << " in ";
    os << diSubprogram.getFile().getDirectory().getValue() << "/"
       << diSubprogram.getFile().getName().getValue();
    os << ":" << diSubprogram.getLine();
  } else {
    attr.print(os);
  }
  return os;
}

static llvm::raw_ostream &printPartialLocation(llvm::raw_ostream &os,
                                               Location loc) {
  if (isa<UnknownLoc>(loc)) {
    os << "<unknown>";
  } else if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
    os << flc.getFilename() << ":" << flc.getLine() << ":" << flc.getColumn();
  } else if (auto callsite = dyn_cast<CallSiteLoc>(loc)) {
    printPartialLocation(os, callsite.getCallee());
    printPartialLocation(os << "\ncalled from: ", callsite.getCaller());
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    if (fused.getLocations().size() > 1)
      os << "fused<";
    llvm::interleaveComma(fused.getLocations(), os, [&](Location nested) {
      printPartialLocation(os, nested);
    });
    if (fused.getLocations().size() > 1)
      os << ">";
    os << " (";
    if (auto md = fused.getMetadata())
      printMetadata(os, md);
    else
      os << "<null metadata>";
    os << ")";
  } else {
    loc.print(os);
  }

  return os;
}

static void attachAndPrintLocation(Operation *op, bool attach = true,
                                   bool print = true) {
  std::string output;
  llvm::raw_string_ostream os(output);
  printPartialLocation(os, op->getLoc());
  if (attach)
    op->setAttr("enzyme.location", StringAttr::get(op->getContext(), os.str()));
  if (print)
    op->emitRemark() << os.str();
}

class PrintLocationPass
    : public enzyme::impl::PrintLocationPassBase<PrintLocationPass> {
public:
  using PrintLocationPassBase::PrintLocationPassBase;

  void runOnOperation() override {
    SmallVector<Operation *> targets;
    getOperation()->walk([&](Operation *op) {
      if (op->hasAttr("enzyme.print_location")) {
        targets.push_back(op);
      }
    });

    for (Operation *op : targets) {
      attachAndPrintLocation(op, shouldAttach, shouldPrint);
    }
  }
};

} // end anonymous namespace
