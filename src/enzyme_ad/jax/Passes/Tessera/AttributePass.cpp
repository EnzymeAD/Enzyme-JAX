//===----------------------------------------------------------------------===//
//
// This file contains a pass that processes custom Enzyme tessera_op annotations 
// and turns them into LLVM tessera.convert function attributes.
//
//===----------------------------------------------------------------------===//

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"
#include "Utils.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Attributes.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif

bool processTesseraAttributes(Module &M) {
  bool changed = false;

  // Modern path for processing annotations
  if (GlobalVariable *GA = M.getGlobalVariable("llvm.global.annotations")) {
    if (GA->hasInitializer()) {
      auto AOp = GA->getInitializer();
      // all metadata are stored in an array of struct of metadata
      if (ConstantArray *CA = dyn_cast<ConstantArray>(AOp)) {
        // so iterate over the operands
        SmallVector<Constant *, 1> replacements;
        for (Value *CAOp : CA->operands()) {
          // get the struct, which holds a pointer to the annotated function
          // as first field, and the annotation as second field
          ConstantStruct *CS = dyn_cast<ConstantStruct>(CAOp);
          if (!CS || CS->getNumOperands() < 2)
          replacements.push_back(cast<Constant>(CAOp));
            continue;

          // the second field is a pointer to a global constant Array that
          // holds the string
          GlobalVariable *GAnn =
              dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0));

          ConstantDataArray *A = nullptr;

          if (GAnn)
            A = dyn_cast<ConstantDataArray>(GAnn->getOperand(0));
          else
            A = dyn_cast<ConstantDataArray>(CS->getOperand(1)->getOperand(0));

          if (!A) {
            replacements.push_back(cast<Constant>(CAOp));
            continue;
          }

          // we have the annotation! Check it's an epona annotation
          // and process
          StringRef AS = A->getAsCString();

          Constant *Val = cast<Constant>(CS->getOperand(0));
          while (auto CE = dyn_cast<ConstantExpr>(Val))
            Val = CE->getOperand(0);

          Function *Func = dyn_cast<Function>(Val);
          GlobalVariable *Glob = dyn_cast<GlobalVariable>(Val);

          // check for tessera_op annotation
          if (startsWith(AS, "tessera_op") && Func) {
            // extract value after '='
            auto val = AS.substr(1 + AS.find('='));
            Func->addAttribute(
                AttributeList::FunctionIndex,
                Attribute::get(Func->getContext(), "tessera.convert", val));
            changed = true;
            replacements.push_back(Constant::getNullValue(CAOp->getType()));
            continue;
          }

          replacements.push_back(cast<Constant>(CAOp));
        }
        GA->setInitializer(ConstantArray::get(CA->getType(), replacements));
      }
    }
  }

  SmallVector<GlobalVariable *, 1> toErase;
  for (GlobalVariable &g : M.globals()) {
    if (g.getName().contains("__tessera_op")) {
      if (g.hasInitializer()) {
        auto CA = dyn_cast<ConstantAggregate>(g.getInitializer());
        if (!CA || CA->getNumOperands() < 2) {
          llvm::errs() << "Use of "
                       << "tessera_op"
                       << " must be a "
                          "constant of size at least "
                       << 2 << " " << g << "\n";
          llvm_unreachable("tessera_op");
        }
        Value *V = CA->getOperand(0);
        Value *name = CA->getOperand(1);
        while (auto CE = dyn_cast<ConstantExpr>(V)) {
          V = CE->getOperand(0);
        }
        while (auto CE = dyn_cast<ConstantExpr>(name)) {
          name = CE->getOperand(0);
        }
        StringRef nameVal;
        if (auto GV = dyn_cast<GlobalVariable>(name))
          if (GV->isConstant())
            if (auto C = GV->getInitializer())
              if (auto CA = dyn_cast<ConstantDataArray>(C))
                if (CA->getType()->getElementType()->isIntegerTy(8) &&
                    CA->isCString())
                  nameVal = CA->getAsCString();

        if (nameVal == "") {
          llvm::errs() << *name << "\n";
          llvm::errs() << "Use of "
                       << "tessera_op"
                       << "requires a non-empty function name"
                       << "\n";
          llvm_unreachable("tessera_op");
        }
        if (auto F = cast<Function>(V)) {
          F->addAttribute(
              AttributeList::FunctionIndex,
              Attribute::get(g.getContext(), "tessera.convert", nameVal));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __tessera_op must be a "
                          "constant function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__tessera_op");
        }
      }
    }
  }

  // Cleanup: remove global variables
  for (auto G : toErase) {
    for (auto name : {"llvm.used", "llvm.compiler.used"}) {
      if (auto V = M.getGlobalVariable(name)) {
        auto C = cast<ConstantArray>(V->getInitializer());
        SmallVector<Constant *, 1> toKeep;
        bool found = false;
        for (unsigned i = 0; i < C->getNumOperands(); i++) {
          Value *Op = C->getOperand(i)->stripPointerCasts();
          if (Op == G)
            found = true;
          else
            toKeep.push_back(C->getOperand(i));
        }
        if (found) {
          if (toKeep.size()) {
            auto CA = ConstantArray::get(
                ArrayType::get(C->getType()->getElementType(), toKeep.size()),
                toKeep);
            GlobalVariable *NGV = new GlobalVariable(
                CA->getType(), V->isConstant(), V->getLinkage(), CA, "",
                V->getThreadLocalMode());
#if LLVM_VERSION_MAJOR > 16
            V->getParent()->insertGlobalVariable(V->getIterator(), NGV);
#else
            V->getParent()->getGlobalList().insert(V->getIterator(), NGV);
#endif
            NGV->takeName(V);

            // Nuke the old list, replacing any uses with the new one.
            if (!V->use_empty()) {
              Constant *VV = NGV;
              if (VV->getType() != V->getType())
                VV = ConstantExpr::getBitCast(VV, V->getType());
              V->replaceAllUsesWith(VV);
            }
          }
          V->eraseFromParent();
        }
      }
    }
    changed = true;
    G->replaceAllUsesWith(ConstantPointerNull::get(G->getType()));
    G->eraseFromParent();
  }

  return changed;
}

namespace {

class TesseraAttributes final : public ModulePass {
public:
  static char ID;
  TesseraAttributes() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnModule(Module &M) override { return processTesseraAttributes(M); }
  
  StringRef getPassName() const override { return "process-tessera-attributes"; }
};

} // namespace

char TesseraAttributes::ID = 0;

static RegisterPass<TesseraAttributes> X("tessera-attributes", "Process Tessera Attributes Pass");

ModulePass *createTesseraAttributesPass() {
  return new TesseraAttributes();
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>
#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddTesseraAttributesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createTesseraAttributesPass());
}

TesseraAttributesNewPM::Result
TesseraAttributesNewPM::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  bool changed = processTesseraAttributes(M);
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey TesseraAttributesNewPM::Key;