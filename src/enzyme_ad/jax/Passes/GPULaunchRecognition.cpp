
#include "AffineUtils.h"
#include "Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "gpu-launch-recognition"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPULAUNCHRECOGNITION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

constexpr char gpuModuleName[] = "__mlir_gpu_module";
constexpr char kernelPrefix[] = "__mlir_launch_kernel_";

struct GPULaunchRecognitionPass
    : public enzyme::impl::GPULaunchRecognitionBase<
          GPULaunchRecognitionPass> {
  using GPULaunchRecognitionBase::GPULaunchRecognitionBase;
  void runOnOperation() override {
  
  llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
  getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
    auto symName = funcOp.getName();
    if (symName.starts_with(kernelPrefix))
      launchFuncs.push_back(funcOp);
  });

  auto ctx = getOperation()->getContext();

  auto moduleBuilder = OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
  auto gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(getOperation()->getLoc(), gpuModuleName);
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
  //gpuModule->setAttr(
  //    LLVM::LLVMDialect::getDataLayoutAttrName(),
  //    deviceModule->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
  gpuModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayout);
  OpBuilder builder(getOperation()->getContext());


    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

  SmallVector<Operation*> tocopy;
  for (auto launchFunc : launchFuncs) {
    auto launchFuncUses = launchFunc.getSymbolUses(getOperation());
    for (auto use : *launchFuncUses) {
      if (auto cop = dyn_cast<CallOpInterface>(use.getUser())) {
        
      if (auto cur = dyn_cast_if_present<FunctionOpInterface>(cop.resolveCallable())) {


  FunctionType gpuTy0 = dyn_cast<FunctionType>(cur.getFunctionType());
  if (!gpuTy0) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(cur.getFunctionType())) {
      SmallVector<Type> restys;
      if (!isa<LLVM::LLVMVoidType>(lty.getReturnType()))
	   restys.push_back(lty.getReturnType());

      gpuTy0 = builder.getFunctionType(lty.getParams(), restys);
    } else {
      cur.emitError(
          "Require target operand to have functiontype or llvmfunctiontype");
      continue;
    }
  }

    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
    auto gpufunc = builder.create<gpu::GPUFuncOp>(cur->getLoc(), cur.getName(), gpuTy0);
    {
      auto entry = &gpufunc.getBody().front();
      builder.setInsertionPointToEnd(entry);
      IRMapping map;
      for (auto &&[oldarg, newarg] :
           zip(cur.getArguments(), gpufunc.getArguments())) {
        Value newval = newarg;

        map.map(oldarg, newval);
      }

      cur.getFunctionBody().cloneInto(&gpufunc.getBody(), map);
  
  	  gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());

      gpufunc->walk([](LLVM::ReturnOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
      });

      gpufunc->walk([](LLVM::UnreachableOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
      });

      gpufunc->walk([](func::ReturnOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
      });


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
       
     auto loc = launchFunc->getLoc();
	builder.setInsertionPointAfter(cop);

	auto shMemSize = builder.create<LLVM::TruncOp>(
            loc, builder.getI32Type(), cop.getArgOperands()[7]);
        auto stream = cop.getArgOperands()[8];
        // TODO stream is arg 8
        llvm::SmallVector<mlir::Value> args;
        for (unsigned i = 9; i < cop.getArgOperands().size(); i++)
          args.push_back(cop.getArgOperands()[i]);
        builder.create<gpu::LaunchFuncOp>(
            loc, gpufunc,
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[1]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[2]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[3])}),
            gpu::KernelDim3(
                {builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[4]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[5]),
                 builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                              cop.getArgOperands()[6])}),
            shMemSize, ValueRange(args), stream.getType(), ValueRange(stream));
        cop->erase();
      }
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  llvm::SmallSet<Operation*, 1> done;
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
      }
    }
    }
  }

  if (launchFuncs.size())
    getOperation()->setAttr("gpu.container_module", OpBuilder(ctx).getUnitAttr());
  }
};

