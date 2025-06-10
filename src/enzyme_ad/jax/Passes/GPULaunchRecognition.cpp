
#include "AffineUtils.h"
#include "Passes.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#define DEBUG_TYPE "gpu-launch-recognition"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPULAUNCHRECOGNITION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace affine;

constexpr char gpuModuleName[] = "__mlir_gpu_module";
constexpr char kernelPrefix[] = "__mlir_launch_kernel_";

LogicalResult mergeDeviceIntoHost(mlir::ModuleOp hostModule,
                                  mlir::ModuleOp deviceModule) {
}


struct GPULaunchRecognitionPass
    : public enzyme::impl::GPULaunchRecognitionPassBase<
          GPULaunchRecognitionPass> {
  using GPULaunchRecognitionPassBase::GPULaunchRecognitionPassBase;
  void runOnOperation() override {
  
  llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
  getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
    auto symName = funcOp.getName();
    if (symName.starts_with(kernelPrefix))
      launchFuncs.push_back(funcOp);
  });

  auto ctx = getOperation().getContext();

  auto moduleBuilder = OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(
      deviceModule->getLoc(), gpuModuleName);
  // TODO get these target attrs from somewhere
  auto target = moduleBuilder.getAttr<NVVM::NVVMTargetAttr>(
      /*optLevel=*/2, /*triple=*/"nvptx64-nvidia-cuda", "sm_80", "+ptx60",
      /*flags=*/nullptr,
      /*linkLibs=*/nullptr);
  gpuModule.setTargetsAttr(moduleBuilder.getArrayAttr({target}));

  DataLayoutSpecInterface dataLayout = {};
  // Set index type size to 32 bits
  {
    llvm::DenseMap<mlir::TypeAttr, mlir::DataLayoutEntryInterface> typeEntries;
    auto type = IndexType::get(ctx);
    auto key = mlir::TypeAttr::get(type);
    uint64_t size = 32;
    auto params = IntegerAttr::get(mlir::IntegerType::get(ctx, 64), size);
    typeEntries.try_emplace(key, DataLayoutEntryAttr::get(type, params));
    SmallVector<DataLayoutEntryInterface> entries;
    entries.reserve(typeEntries.size());
    for (const auto &it : typeEntries)
      entries.push_back(it.second);
    dataLayout = DataLayoutSpecAttr::get(ctx, entries);
  }
  gpuModule->setAttr(
      LLVM::LLVMDialect::getDataLayoutAttrName(),
      deviceModule->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
  gpuModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayout);
  OpBuilder builder(getOperation().getContext());

  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  SmallVector<Operation*> tocopy;
  for (auto launchFunc : launchFuncs) {
    auto launchFuncUses = launchFunc.getSymbolUses(hostModule);
    for (auto use : *launchFuncUses) {
      if (auto cop = dyn_cast<CallOpInterface>(use)) {
        
      if (auto cur = cop.resolveCallable()) {
	 cur->moveBefore(&gpuModule.getBodyRegion().front(), gpuModule.getBodyRegion().front().begin());
      done.insert(cur);
      builder.clone(*cur);
      cur->walk([&](CallOpInterface cop) {
        if (auto op2 = cop.resolveCallable())
          tocopy.push_back(op2);
      });
      cur->walk([&](LLVM::AddressOfOp cop) {
        if (auto op2 = cop.getGlobal(symbolTable))
          tocopy.push_back(op2);
        else if (auto op2 = cop.getFunction(symbolTable))
          tocopy.push_back(op2);
      });
      CallInterfaceCallable callable = call.getCallableForCallee();

  	auto symbolRef = cast<SymbolRefAttr>(callable);

        SymbolRefAttr gpuFuncSymbol = SymbolRefAttr::get(
            StringAttr::get(ctx, gpuModuleName),
            {symbolRef});
        
	auto shMemSize = builder.create<LLVM::TruncOp>(
            loc, builder.getI32Type(), callOp.getArgOperands()[7]);
        auto stream = callOp.getArgOperands()[8];
        // TODO stream is arg 8
        llvm::SmallVector<mlir::Value> args;
        for (unsigned i = 9; i < callOp.getArgOperands().size(); i++)
          args.push_back(callOp.getArgOperands()[i]);
        builder.create<gpu::LaunchFuncOp>(
            loc, gpuFuncSymbol,
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[1]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[2]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[3])}),
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[4]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[5]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              callOp.getArgOperands()[6])}),
            shMemSize, ValueRange(args), stream);
        callOp->erase();
      }
    }
    }
  }
  SmallSet<Operation*, 1> done;
  while (tocopy.size()) {
      auto cur = tocopy.pop_back_val();
      if (done.count(cur))
        continue;
      done.insert(cur);
      builder.clone(*cur);
      cur->walk([&](CallOpInterface cop) {
        if (auto op2 = cop.resolveCallable())
          tocopy.push_back(op2);
      });
      cur->walk([&](LLVM::AddressOfOp cop) {
        if (auto op2 = cop.getGlobal(symbolTable))
          tocopy.push_back(op2);
        else if (auto op2 = cop.getFunction(symbolTable))
          tocopy.push_back(op2);
      });
    }

  if (launchFuncs.size())
    getOperation()->setAttr("gpu.container_module", OpBuilder(ctx).getUnitAttr());
  return success();
  }
};

template <class T>
struct SimplifyAccessAffineExprs : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  IslAnalysis &islAnalysis;
  SimplifyAccessAffineExprs(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<T>(&context), islAnalysis(islAnalysis) {}
  LogicalResult matchAndRewrite(T access,
                                PatternRewriter &rewriter) const override {
    return handleAffineOp(islAnalysis, access);
  }
};

void mlir::populateAffineExprSimplificationPatterns(
    IslAnalysis &islAnalysis, RewritePatternSet &patterns) {
  // clang-format off
  patterns.insert<
    SimplifyAccessAffineExprs<affine::AffineLoadOp>,
    SimplifyAccessAffineExprs<affine::AffineStoreOp>,
    SimplifyAccessAffineExprs<affine::AffineVectorLoadOp>,
    SimplifyAccessAffineExprs<affine::AffineVectorStoreOp>
  >(*patterns.getContext(), islAnalysis);
  // clang-format on
}
