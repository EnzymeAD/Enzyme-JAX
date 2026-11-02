#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/GrammarRegistry.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTGRAMMAROPERATORSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

void printAxesAndSignature(llvm::raw_ostream &os,
                           const GrammarDescription &description) {
  os << "axes " << description.axes << " signature [";
  llvm::interleaveComma(description.signature.axes, os);
  os << "]";
}

struct PrintGrammarOperatorsPass
    : public impl::PrintGrammarOperatorsPassBase<PrintGrammarOperatorsPass> {
  using PrintGrammarOperatorsPassBase::PrintGrammarOperatorsPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      FailureOr<GrammarDescription> description = recognize(op);
      if (failed(description))
        return;

      std::string text;
      llvm::raw_string_ostream os(text);
      os << "grammar " << description->kind.getValue() << " ";
      printAxesAndSignature(os, *description);
      if (description->params)
        os << " params " << description->params;
      if (canonical) {
        os << "\ncanonical: ";
        printAxesAndSignature(os, canonicalizeAxes(*description).description);
      }

      op->emitRemark() << text;
    });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
