
#include "AffineUtils.h"
#include "Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
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
    : public enzyme::impl::GPULaunchRecognitionBase<GPULaunchRecognitionPass> {
  using GPULaunchRecognitionBase::GPULaunchRecognitionBase;
  void runOnOperation() override {

    llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      auto symName = funcOp.getName();
      if (symName.starts_with(kernelPrefix))
        launchFuncs.push_back(funcOp);
    });

    auto ctx = getOperation()->getContext();

    gpu::GPUModuleOp gpuModule;
    if (use_launch_func) {
      auto moduleBuilder =
          OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
      gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(
          getOperation()->getLoc(), gpuModuleName);
      // TODO get these target attrs from somewhere
      auto target = moduleBuilder.getAttr<NVVM::NVVMTargetAttr>(
          /*optLevel=*/2, /*triple=*/"nvptx64-nvidia-cuda", "sm_80", "+ptx60",
          /*flags=*/nullptr,
          /*linkLibs=*/nullptr);
      gpuModule.setTargetsAttr(moduleBuilder.getArrayAttr({target}));

      DataLayoutSpecInterface dataLayout = {};
      // Set index type size to 32 bits
      {
        llvm::DenseMap<mlir::TypeAttr, mlir::DataLayoutEntryInterface>
            typeEntries;
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
      // gpuModule->setAttr(
      //     LLVM::LLVMDialect::getDataLayoutAttrName(),
      //     deviceModule->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
      gpuModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayout);
    }

    OpBuilder builder(getOperation()->getContext());

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    SetVector<Operation *> tocopy;
    for (auto launchFunc : launchFuncs) {
      auto launchFuncUses = launchFunc.getSymbolUses(getOperation());
      for (auto use : *launchFuncUses) {
        if (auto cop = dyn_cast<CallOpInterface>(use.getUser())) {
          if (cop.getArgOperands().size() == 0)
            continue;
          auto argop =
              cop.getArgOperands()[0].getDefiningOp<LLVM::AddressOfOp>();
          if (!argop)
            continue;
          auto cur = argop.getFunction(symbolTable);
          if (!cur)
            continue;

          FunctionType gpuTy0 = dyn_cast<FunctionType>(cur.getFunctionType());
          if (!gpuTy0) {
            if (auto lty =
                    dyn_cast<LLVM::LLVMFunctionType>(cur.getFunctionType())) {
              SmallVector<Type> restys;
              if (!isa<LLVM::LLVMVoidType>(lty.getReturnType()))
                restys.push_back(lty.getReturnType());

              gpuTy0 = builder.getFunctionType(lty.getParams(), restys);
            } else {
              cur.emitError("Require target operand to have functiontype or "
                            "llvmfunctiontype");
              continue;
            }
          }

          gpu::GPUFuncOp gpufunc;
          if (use_launch_func) {
            builder.setInsertionPointToStart(
                &gpuModule.getBodyRegion().front());
            gpufunc = builder.create<gpu::GPUFuncOp>(cur->getLoc(),
                                                     cur.getName(), gpuTy0);
            auto entry = &gpufunc.getBody().front();
            builder.setInsertionPointToEnd(entry);
            IRMapping map;
            gpufunc.getBody().getBlocks().clear();
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
                tocopy.insert(op2);
            });
            cur->walk([&](LLVM::AddressOfOp cop) {
              if (auto op2 = cop.getGlobal(symbolTable))
                tocopy.insert(op2);
              else if (auto op2 = cop.getFunction(symbolTable))
                tocopy.insert(op2);
            });
          }

          auto loc = cop->getLoc();
          builder.setInsertionPointAfter(cop);

          auto shMemSize = builder.create<LLVM::TruncOp>(
              loc, builder.getI32Type(), cop.getArgOperands()[7]);
          auto stream = cop.getArgOperands()[8];
          llvm::SmallVector<mlir::Value> args;
          for (unsigned i = 9; i < cop.getArgOperands().size(); i++)
            args.push_back(cop.getArgOperands()[i]);

          Value grid[3];
          for (int i = 0; i < 3; i++) {
            if (use_launch_func)
              grid[i] = builder.create<LLVM::SExtOp>(
                  loc, builder.getI64Type(), cop.getArgOperands()[i + 1]);
            else
              grid[i] = builder.create<arith::IndexCastOp>(
                  loc, builder.getIndexType(), cop.getArgOperands()[i + 1]);
          }
          Value block[3];
          for (int i = 0; i < 3; i++) {
            if (use_launch_func)
              block[i] = builder.create<LLVM::SExtOp>(
                  loc, builder.getI64Type(), cop.getArgOperands()[i + 4]);
            else
              block[i] = builder.create<arith::IndexCastOp>(
                  loc, builder.getIndexType(), cop.getArgOperands()[i + 4]);
          }
          if (stream.getDefiningOp<LLVM::ZeroOp>()) {
            if (use_launch_func) {
              builder.create<gpu::LaunchFuncOp>(
                  loc, gpufunc, gpu::KernelDim3{grid[0], grid[1], grid[2]},
                  gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                  ValueRange(args));
            } else {
              auto op = builder.create<mlir::gpu::LaunchOp>(
                  launchFunc->getLoc(), grid[0], grid[1], grid[2], block[0],
                  block[1], block[2], shMemSize, nullptr, ValueRange());
              builder.setInsertionPointToStart(&op.getRegion().front());
              builder.create<LLVM::CallOp>(loc, cur, args);
              builder.create<gpu::TerminatorOp>(loc);
            }
          } else {
            if (use_launch_func) {
              builder.create<gpu::LaunchFuncOp>(
                  loc, gpufunc, gpu::KernelDim3{grid[0], grid[1], grid[2]},
                  gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                  ValueRange(args), stream.getType(), ValueRange(stream));
            } else {
              assert(isa<LLVM::LLVMPointerType>(stream.getType()));
              stream = builder.create<enzymexla::StreamToTokenOp>(
                  loc, gpu::AsyncTokenType::get(ctx), stream);
              auto op = builder.create<mlir::gpu::LaunchOp>(
                  launchFunc->getLoc(), grid[0], grid[1], grid[2], block[0],
                  block[1], block[2], shMemSize, stream.getType(),
                  ValueRange(stream));
              builder.setInsertionPointToStart(&op.getRegion().front());
              builder.create<LLVM::CallOp>(loc, cur, args);
              builder.create<gpu::TerminatorOp>(loc);
            }
          }
          cop->erase();
        }
      }
    }
    if (use_launch_func) {
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      llvm::SmallSet<Operation *, 1> done;
      while (tocopy.size()) {
        auto cur = tocopy.pop_back_val();
        if (done.count(cur))
          continue;
        done.insert(cur);
        builder.clone(*cur);
        cur->walk([&](CallOpInterface cop) {
          if (auto op2 = cop.resolveCallable())
            tocopy.insert(op2);
        });
        cur->walk([&](LLVM::AddressOfOp cop) {
          if (auto op2 = cop.getGlobal(symbolTable))
            tocopy.insert(op2);
          else if (auto op2 = cop.getFunction(symbolTable))
            tocopy.insert(op2);
        });
      }
    }

    if (launchFuncs.size() && use_launch_func)
      getOperation()->setAttr("gpu.container_module",
                              OpBuilder(ctx).getUnitAttr());

    getOperation()->walk([](LLVM::CallOp call) {
      auto callee = call.getCallee();
      OpBuilder builder(call);
      auto i8 = builder.getIntegerType(8);
      if (callee == "cudaMalloc") {

        Value arg = call->getOperand(1);
        if (!isa<IndexType>(arg.getType()))
          arg = builder.create<arith::IndexCastOp>(call->getLoc(),
                                                   builder.getIndexType(), arg);

        auto res = builder.create<gpu::AllocOp>(
            call.getLoc(),
            MemRefType::get({ShapedType::kDynamic}, i8,
                            MemRefLayoutAttrInterface{},
                            builder.getI64IntegerAttr(1)),
            (mlir::Type) nullptr, ValueRange(), ValueRange(arg), ValueRange());
        auto ptr = builder.create<enzymexla::Memref2PointerOp>(
            call.getLoc(), LLVM::LLVMPointerType::get(call.getContext()),
            res.getResult(0));
        builder.create<LLVM::StoreOp>(call.getLoc(), ptr, call->getOperand(0));
        auto replace =
            builder.create<LLVM::ZeroOp>(call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }
      if (callee == "cudaMemcpy") {
        APInt directionA;

#if 0 
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};
#endif
        if (matchPattern(call->getOperand(3), m_ConstantInt(&directionA))) {
          auto dst = call->getOperand(0);
          if (directionA == 0 || directionA == 2)
            dst = builder.create<enzymexla::Pointer2MemrefOp>(
                call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                dst);
          else
            dst = builder.create<enzymexla::Pointer2MemrefOp>(
                call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                dst);

          auto src = call->getOperand(1);
          if (directionA == 0 || directionA == 1)
            src = builder.create<enzymexla::Pointer2MemrefOp>(
                call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{}),
                src);
          else
            src = builder.create<enzymexla::Pointer2MemrefOp>(
                call->getLoc(),
                MemRefType::get({ShapedType::kDynamic}, i8,
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1)),
                src);

          Value arg = call->getOperand(2);
          if (!isa<IndexType>(arg.getType()))
            arg = builder.create<arith::IndexCastOp>(
                call->getLoc(), builder.getIndexType(), arg);

          auto res = builder.create<enzymexla::MemcpyOp>(
              call.getLoc(), (mlir::Type) nullptr, ValueRange(), dst, src, arg);
          auto replace =
              builder.create<LLVM::ZeroOp>(call.getLoc(), call.getType(0));
          call->replaceAllUsesWith(replace);
          call->erase();
          return;
        }
      }
    });
  }
};
