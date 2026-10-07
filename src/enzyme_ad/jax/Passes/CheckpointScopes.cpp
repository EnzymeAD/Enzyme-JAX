//===- CheckpointScopes.cpp - Loops checkpointed by a JAX name scope ------===//
//
// enzyme_ad.jax.checkpoint(schedule, budget) is a JAX name scope,
// `enzyme_checkpoint[schedule,budget]`, which costs the primal nothing: JAX
// records it in the location of each operation lowered inside it, and of the
// stablehlo.while a lax.fori_loop, while_loop or scan becomes there, whose
// name then ends in `enzyme_checkpoint[schedule,budget]/while`. Loops nested
// in that one carry the scope further up their names, and are left alone. A
// JAX transformation wraps the scope's name in its own (vmap(...), jvp(...),
// transpose(jvp(...))); the loop it makes is still the one marked.
// This pass asks Enzyme-MLIR to checkpoint the loops so marked, as Enzyme's
// other frontends ask with the same schedules (enzyme/checkpoint_schedule.h).
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/CheckpointSchedule.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "enzyme-checkpoint-scopes"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CHECKPOINTSCOPESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kScope = "enzyme_checkpoint[";

/// The name JAX gave the operation at `loc`: its name stack, the scopes it was
/// lowered in and the primitive, separated by '/'.
static std::optional<StringRef> getJaxName(Location loc) {
  std::optional<StringRef> name;
  loc->walk([&](Location l) {
    if (auto nl = dyn_cast<NameLoc>(l)) {
      name = nl.getName().getValue();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return name;
}

struct CheckpointScopesPass
    : public enzyme::impl::CheckpointScopesPassBase<CheckpointScopesPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](stablehlo::WhileOp loop) {
      // An attribute already there was put there on purpose (Reactant's
      // @trace checkpointing=...).
      if (loop->hasAttr("enzyme.enable_checkpointing"))
        return;
      std::optional<StringRef> name = getJaxName(loop.getLoc());
      if (!name || !name->consume_back("/while"))
        return;
      // The innermost scope only: a loop nested in a marked one has the
      // marked loop's body between the two.
      StringRef scope = name->rsplit('/').second;
      if (scope.empty())
        scope = *name;
      // A JAX transformation of the loop names the scope inside its own:
      // vmap(enzyme_checkpoint[...]), transpose(jvp(enzyme_checkpoint[...])).
      // The loop it makes is the same loop, batched or differentiated.
      while (!scope.starts_with(kScope) && scope.ends_with(")")) {
        size_t open = scope.find('(');
        if (open == StringRef::npos || open == 0 ||
            !llvm::all_of(scope.take_front(open),
                          [](char c) { return llvm::isAlnum(c) || c == '_'; }))
          return;
        scope = scope.slice(open + 1, scope.size() - 1);
      }
      if (!scope.consume_front(kScope) || !scope.consume_back("]"))
        return;
      auto [scheduleName, budgetText] = scope.split(',');
      int64_t schedule = enzyme_ckpt_schedule_from_name(scheduleName.data(),
                                                        scheduleName.size()),
              budget = 0;
      if (schedule < 0 ||
          (!budgetText.empty() && budgetText.trim().getAsInteger(10, budget))) {
        loop.emitWarning() << "unknown checkpointing request '" << kScope
                           << scope << "]'";
        return;
      }
      (void)enzyme::setCheckpointSchedule(loop, schedule, budget);
    });
  }
};

} // namespace
