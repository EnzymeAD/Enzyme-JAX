#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/LeanCheck.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "src/enzyme_ad/jax/Passes/Perfify/PerfifyLeanEmitter.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

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

/// The statement and tactic proof of each theorem in `text` whose proof isn't
/// just `sorry`.
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
    size_t nameEnd = line.starts_with("«") ? line.find("»") : 0;
    StringRef name = line.take_front(line.find_first_of(" :(", nameEnd));
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
    StringRef decl(lines[i].data(), lines[end - 1].end() - lines[i].data());
    i = end - 1;

    StringRef trimmed = StringRef(body).rtrim();
    if (trimmed.trim().empty() || trimmed.trim() == "sorry")
      continue;
    proofs[name] = decl.rtrim().str() + "\n";
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

static std::string normalize(StringRef text) {
  SmallVector<StringRef> words;
  SplitString(text, words, " \t\n");
  return llvm::join(words, " ");
}

static std::string checkStatement(const LeanTheorem &thm, StringRef user,
                                  std::string &added) {
  std::string gen = normalize(thm.statement);
  std::string conclusion = " : " + normalize(thm.conclusion);
  StringRef head = StringRef(gen).drop_back(conclusion.size());
  std::string text = normalize(user);
  if (text.size() < gen.size() || !StringRef(text).starts_with(head) ||
      !StringRef(text).ends_with(conclusion))
    return "it does not start and end as `" + gen + "`";
  StringRef middle =
      StringRef(text).slice(head.size(), text.size() - conclusion.size());
  int depth = 0;
  for (char c : middle) {
    if (StringRef("({[").contains(c))
      ++depth;
    else if (StringRef(")}]").contains(c))
      --depth;
    if (depth < 0 || (depth == 0 && !StringRef(" )}]").contains(c)))
      return "it adds more than binders";
  }
  if (depth != 0)
    return "it adds more than binders";
  if (llvm::Regex("(^|[^[:alnum:]_'])call[0-9]+").match(middle))
    return "an added binder mentions a call's cost";
  added = middle.trim().str();
  return "";
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
    SmallVector<std::string> addedBinders;
    for (LeanTheorem &thm : file.theorems) {
      auto proofIt = proofs.find(thm.name);
      if (proofIt == proofs.end())
        continue;
      std::string added;
      std::string error = checkStatement(
          thm, StringRef(proofIt->second).split(":= by").first, added);
      if (!error.empty()) {
        thm.hyp.emitWarning("statement of ")
            << thm.name << " does not match the generated one: " << error;
        continue;
      }
      checked.push_back(&thm);
      names.push_back(thm.name);
      addedBinders.push_back(std::move(added));
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

    llvm::MapVector<ConditionsOp, llvm::SetVector<StringRef>> touched;
    for (auto [thm, verdict, added] :
         llvm::zip_equal(checked, *verdicts, addedBinders)) {
      if (!verdict.empty()) {
        thm->hyp.emitWarning("Lean proof of ")
            << thm->name << " does not check: " << verdict;
        continue;
      }
      if (AssumeOp assume = getAssume(thm->hyp)) {
        assume.setSatresAttr(BoolAttr::get(&getContext(), true));
        llvm::SetVector<StringRef> &assumptions = touched[thm->cond];
        if (!added.empty())
          assumptions.insert(added);
      }
    }
    for (auto &[cond, assumptions] : touched) {
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
        if (!assumptions.empty())
          cond.setLeanAssumptionsAttr(
              Builder(&getContext())
                  .getStrArrayAttr(assumptions.getArrayRef()));
      }
    }
  }
};

} // namespace
