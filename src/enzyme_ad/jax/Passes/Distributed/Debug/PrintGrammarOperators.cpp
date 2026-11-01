#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/Verifier.h"
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
    // Collected first: materializing inserts and erases ops beside them.
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) { ops.push_back(op); });

    const GrammarRegistry &registry = getGrammarRegistry(&getContext());
    for (Operation *op : ops) {
      FailureOr<GrammarDescription> description = recognize(op);
      if (failed(description))
        continue;

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

      if (materialize && !isa<GrammarTermOp>(op) &&
          registry.lookup(description->kind.getValue())) {
        OpBuilder builder(op);
        GrammarTermOp term =
            materializeGrammarTerm(builder, op->getLoc(), *description,
                                   op->getOperands(), op->getResultTypes());
        // verify() reports its own errors on the term; the remark still names
        // the op the term came from.
        bool verified = succeeded(verify(term));
        os << "\nterm: ";
        term->print(os);
        term.erase();
        if (!verified)
          signalPassFailure();
      }
      op->emitRemark() << text;
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
