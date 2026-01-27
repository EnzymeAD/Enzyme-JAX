//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise the generic tessera_op attribute
// into a proper MLIR Tessera convert attribute on the operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"

using namespace mlir;

namespace {

struct FuncAttrToTesseraAttrPass
    : public PassWrapper<FuncAttrToTesseraAttrPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const final { return "func-attr-to-tessera-attr"; }

  StringRef getDescription() const final {
    return "Convert generic tessera_op func attribute into tessera.convert "
           "attribute.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<enzyme::tessera::TesseraDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      llvm::errs() << "Checking function: " << func.getName() << "\n";
      if (auto attr = func->getAttrOfType<StringAttr>("tessera_op")) {
        llvm::errs() << "  Found tessera_op attribute\n";
        llvm::errs() << "  String value: '" << attr.getValue() << "'\n";
        llvm::errs() << "  About to call ConvertAttr::get...\n";

        auto tesseraAttr = enzyme::tessera::ConvertAttr::get(
            builder.getContext(), attr.getValue().str());

        llvm::errs() << "  ConvertAttr created successfully\n";
        llvm::errs() << "  About to set attribute...\n";

        func->setAttr("tessera.convert", tesseraAttr);

        llvm::errs() << "  Attribute set successfully\n";
        func->removeAttr("tessera_op");
      }
    }

    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      llvm::errs() << "Function: " << func.getName() << "\n";
      llvm::errs() << "All attributes:\n";
      for (auto attr : func->getAttrs()) {
        llvm::errs() << "  " << attr.getName() << " = " << attr.getValue()
                     << "\n";
      }
    }
  }
};

} // namespace

namespace mlir::enzyme::tessera {

std::unique_ptr<mlir::Pass> createFuncAttrToTesseraAttrPass() {
  return std::make_unique<FuncAttrToTesseraAttrPass>();
}

} // namespace mlir::enzyme::tessera
