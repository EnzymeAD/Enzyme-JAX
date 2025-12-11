#ifndef TESSERA_ATTRIBUTES_H
#define TESSERA_ATTRIBUTES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {
class ModulePass;
}

// Legacy PM
llvm::ModulePass *createTesseraAttributesPass();

// New PM
class TesseraAttributesNewPM final
    : public llvm::AnalysisInfoMixin<TesseraAttributesNewPM> {
  friend struct llvm::AnalysisInfoMixin<TesseraAttributesNewPM>;
  
private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  
  TesseraAttributesNewPM() {}
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};

#endif