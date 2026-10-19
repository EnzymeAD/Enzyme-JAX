//===- GpuModuleToBinarySink.cpp - GPU serialization with late sinking ----===//
//
// Serializes GPU modules like the upstream `gpu-module-to-binary` pass, but
// installs an `optimizedLlvmIRCallback` that rematerializes cheap integer and
// address computations next to their uses after the target's LLVM
// optimization pipeline has run. At that point no further IR-level CSE/GVN
// executes, so the shortened live ranges survive to instruction selection.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPUMODULETOBINARYSINK
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

bool isSinkCandidate(llvm::Instruction &I) {
  if (isa<llvm::GetElementPtrInst>(I) || isa<llvm::CastInst>(I))
    return true;
  if (auto *bin = dyn_cast<llvm::BinaryOperator>(&I))
    return bin->getType()->isIntegerTy();
  return false;
}

bool sinkCheapOpsInFunction(llvm::Function &F, int sinkMode) {
  llvm::DominatorTree DT(F);
  llvm::LoopInfo LI(DT);
  bool anyChange = false;
  bool changed = true;
  unsigned iter = 0;
  while (changed && iter++ < 32) {
    changed = false;
    llvm::SmallVector<llvm::Instruction *> insts;
    for (auto &BB : F)
      for (auto &I : BB)
        insts.push_back(&I);
    // Reverse order so late instructions of a chain sink before the values
    // feeding them, letting whole chains migrate in one fixpoint round.
    for (llvm::Instruction *I : llvm::reverse(insts)) {
      if (!isSinkCandidate(*I))
        continue;
      llvm::Loop *defLoop = LI.getLoopFor(I->getParent());
      llvm::MapVector<llvm::BasicBlock *,
                      llvm::SmallVector<llvm::Instruction *>>
          usersByBlock;
      for (llvm::Use &use : I->uses()) {
        auto *user = cast<llvm::Instruction>(use.getUser());
        if (isa<llvm::PHINode>(user) || user->getParent() == I->getParent())
          continue;
        usersByBlock[user->getParent()].push_back(user);
      }
      for (auto &[BB, users] : usersByBlock) {
        llvm::Loop *useLoop = LI.getLoopFor(BB);
        if (useLoop != defLoop) {
          if (sinkMode < 2)
            continue;
          // Only rematerialize into strictly deeper loops on the same nest;
          // never hoist across into an unrelated loop.
          if (!useLoop || (defLoop && !defLoop->contains(useLoop)))
            continue;
        }
        llvm::Instruction *first = users.front();
        for (llvm::Instruction *user : users)
          if (user->comesBefore(first))
            first = user;
        llvm::Instruction *clone = I->clone();
        clone->insertBefore(first->getIterator());
        if (I->hasName())
          clone->setName(I->getName() + ".sunk");
        for (llvm::Instruction *user : users)
          user->replaceUsesOfWith(I, clone);
        changed = true;
        anyChange = true;
      }
      if (I->use_empty()) {
        I->eraseFromParent();
        changed = true;
      }
    }
  }
  return anyChange;
}

void sinkCheapOpsLate(llvm::Module &M, int sinkMode, bool dumpIR) {
  for (llvm::Function &F : M)
    if (sinkMode > 0 && !F.isDeclaration())
      sinkCheapOpsInFunction(F, sinkMode);
  // The IR as handed to instruction selection; lit tests key off this.
  if (dumpIR)
    M.print(llvm::errs(), nullptr);
}

class GpuModuleToBinarySink
    : public enzyme::impl::GpuModuleToBinarySinkBase<GpuModuleToBinarySink> {
public:
  using Base::Base;
  void runOnOperation() override {
    auto targetFormat =
        llvm::StringSwitch<std::optional<gpu::CompilationTarget>>(
            compilationTarget)
            .Cases({"offloading", "llvm"}, gpu::CompilationTarget::Offload)
            .Cases({"assembly", "isa"}, gpu::CompilationTarget::Assembly)
            .Cases({"binary", "bin"}, gpu::CompilationTarget::Binary)
            .Cases({"fatbinary", "fatbin"}, gpu::CompilationTarget::Fatbin)
            .Default(std::nullopt);
    if (!targetFormat) {
      getOperation()->emitError()
          << "Invalid format specified: '" << compilationTarget << "'";
      return signalPassFailure();
    }

    std::optional<SymbolTable> parentTable;
    auto lazyTableBuilder = [&]() -> SymbolTable * {
      if (!parentTable) {
        Operation *table = SymbolTable::getNearestSymbolTable(getOperation());
        if (!table)
          return nullptr;
        parentTable = SymbolTable(table);
      }
      return &parentTable.value();
    };
    SmallVector<Attribute> librariesToLink;
    for (const std::string &path : linkFiles)
      librariesToLink.push_back(StringAttr::get(&getContext(), path));

    int mode = sinkMode;
    bool dump = dumpIR;
    auto sinkCallback = [mode, dump](llvm::Module &M) {
      sinkCheapOpsLate(M, mode, dump);
    };
    gpu::TargetOptions targetOptions(
        toolkitPath, librariesToLink, cmdOptions, elfSection, *targetFormat,
        lazyTableBuilder, /*initialLlvmIRCallback=*/{},
        /*linkedLlvmIRCallback=*/{}, sinkCallback);
    if (failed(gpu::transformGpuModulesToBinaries(
            getOperation(),
            gpu::OffloadingLLVMTranslationAttrInterface(nullptr),
            targetOptions)))
      return signalPassFailure();
  }
};

} // namespace
