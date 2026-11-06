#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/LeanCheck.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "src/enzyme_ad/jax/Passes/Perfify/PerfifyLeanEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_LEANTOPERFIFYPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

/// The tactic proof of each theorem in `text` that isn't just `sorry`.
static llvm::StringMap<std::string> extractProofs(StringRef text) {
  auto isTopLevel = [](StringRef line) {
    return !line.empty() && line.front() != ' ' && line.front() != '\t';
  };

  llvm::StringMap<std::string> proofs;
  SmallVector<StringRef> lines;
  text.split(lines, '\n');
  for (size_t i = 0, e = lines.size(); i != e; ++i) {
    StringRef line = lines[i];
    if (!line.consume_front("theorem "))
      continue;
    StringRef name = line.take_until(
        [](char c) { return c == ' ' || c == ':' || c == '('; });
    if (name.empty())
      continue;

    size_t byLine = i;
    while (!lines[byLine].contains(":= by"))
      if (++byLine == e || isTopLevel(lines[byLine]))
        break;
    if (byLine == e || !lines[byLine].contains(":= by"))
      continue;

    std::string body;
    StringRef sameLine = lines[byLine].split(":= by").second.trim();
    if (!sameLine.empty()) {
      body += "  ";
      body += sameLine;
      body += '\n';
    }
    size_t end = byLine + 1;
    for (; end != e && !isTopLevel(lines[end]); ++end) {
      body += lines[end];
      body += '\n';
    }
    i = end - 1;

    StringRef trimmed = StringRef(body).rtrim();
    if (trimmed.trim().empty() || trimmed.trim() == "sorry")
      continue;
    body.resize(trimmed.size());
    body += '\n';
    proofs[name] = std::move(body);
  }
  return proofs;
}

/// The perfify.assume that `hyp` is the condition of.
static AssumeOp getAssume(HypothesisOp hyp) {
  for (Operation *user : hyp.getResult().getUsers())
    if (auto assume = dyn_cast<AssumeOp>(user))
      return assume;
  return {};
}

struct LeanToPerfifyPass
    : public enzyme::perfify::impl::LeanToPerfifyPassBase<LeanToPerfifyPass> {
  using LeanToPerfifyPassBase::LeanToPerfifyPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(proof);
    if (!buffer) {
      module.emitError("cannot read Lean proof file '")
          << proof << "': " << buffer.getError().message();
      return signalPassFailure();
    }
    llvm::StringMap<std::string> proofs = extractProofs((*buffer)->getBuffer());
    LeanFile file = emitLean(module, imports, proofs);

    SmallVector<LeanTheorem *> checked;
    SmallVector<std::string> names;
    for (LeanTheorem &thm : file.theorems) {
      if (!proofs.contains(thm.name))
        continue;
      checked.push_back(&thm);
      names.push_back(thm.name);
    }
    if (checked.empty())
      return;

    llvm::Expected<SmallVector<std::string>> verdicts =
        checkLeanTheorems(file.text, names, lake, project);
    if (!verdicts) {
      module.emitError("cannot check Lean proofs: ")
          << llvm::toString(verdicts.takeError());
      return signalPassFailure();
    }

    llvm::SetVector<ConditionsOp> touched;
    for (auto [thm, verdict] : llvm::zip_equal(checked, *verdicts)) {
      if (!verdict.empty()) {
        thm->hyp.emitWarning("Lean proof of ")
            << thm->name << " does not check: " << verdict;
        continue;
      }
      if (AssumeOp assume = getAssume(thm->hyp)) {
        assume.setSatresAttr(BoolAttr::get(&getContext(), true));
        touched.insert(thm->cond);
      }
    }
    for (ConditionsOp cond : touched) {
      bool allProven = !cond.walk([](AssumeOp assume) {
                              std::optional<bool> satres = assume.getSatres();
                              return satres && !*satres
                                         ? WalkResult::interrupt()
                                         : WalkResult::advance();
                            })
                            .wasInterrupted();
      if (allProven) {
        cond.setLeanProven(true);
        cond.removeInvalidRegionAttr();
      }
    }
  }
};

} // namespace
