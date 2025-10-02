
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
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "gpu-launch-recognition"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPULAUNCHRECOGNITION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

constexpr char gpuModuleName[] = "__mlir_gpu_module";

struct GPULaunchRecognitionPass
    : public enzyme::impl::GPULaunchRecognitionBase<GPULaunchRecognitionPass> {
  using GPULaunchRecognitionBase::GPULaunchRecognitionBase;

  void initGPUModule(gpu::GPUModuleOp &gpuModule, LLVM::LLVMFuncOp func) {
    if (gpuModule)
      return;
    auto ctx = getOperation()->getContext();
    auto moduleBuilder =
        OpBuilder::atBlockBegin(cast<ModuleOp>(getOperation()).getBody());
    gpuModule = moduleBuilder.create<gpu::GPUModuleOp>(getOperation()->getLoc(),
                                                       gpuModuleName);

    std::string sm;
    if (auto attr = dyn_cast_or_null<ArrayAttr>(func.getPassthroughAttr())) {
      for (auto a : attr) {
        if (auto ar = dyn_cast<ArrayAttr>(a)) {
          if (ar.size() != 2)
            continue;
          auto s0 = dyn_cast<StringAttr>(ar[0]);
          auto s1 = dyn_cast<StringAttr>(ar[1]);
          if (!s0 || !s1)
            continue;
          if (s0.getValue() == "target-cpu")
            sm = s1.getValue();
        }
      }
    }
    std::string feat;
    if (auto attr = dyn_cast_or_null<LLVM::TargetFeaturesAttr>(
            func.getTargetFeaturesAttr())) {
      feat = attr.getFeaturesString();
    }

    auto chip = sm;
    if (chip.size() == 0)
      chip = "sm_80";
    auto features = feat;
    if (features.size() == 0)
      features = "+ptx73";

    // TODO get these target attrs from somewhere
    auto target = moduleBuilder.getAttr<NVVM::NVVMTargetAttr>(
        /*optLevel=*/2, /*triple=*/"nvptx64-nvidia-cuda", chip, features,
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
  void replaceRuntime() {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());
    StringSet<> seenErrors;
    getOperation()->walk([&](LLVM::CallOp call) {
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

      if (callee == "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") {
        auto intType = call.getArgOperands()[2].getType();

        auto fnop = call.getArgOperands()[1].getDefiningOp<LLVM::AddressOfOp>();
        if (!fnop)
          return;

        auto curfn = fnop.getFunction(symbolTable);
        if (!curfn)
          return;

        auto repOp = builder.create<enzymexla::GPUOccupancyOp>(
            call.getLoc(), intType,
            mlir::SymbolRefAttr::get(curfn.getContext(),
                                     curfn.getSymName().str()),
            call.getArgOperands()[2], call.getArgOperands()[3],
            call.getArgOperands()[4]);
        builder.create<LLVM::StoreOp>(call.getLoc(), repOp->getResult(0),
                                      call.getArgOperands()[0]);
        auto replace =
            builder.create<LLVM::ZeroOp>(call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }

      /*
      if (callee == "cudaFuncGetAttributes" ||
          callee == "cudaFuncSetCacheConfig") {
        if (!seenErrors.count(*callee))
          call->emitWarning()
              << " Unsupported runtime function: " << *callee << "\n";
        seenErrors.insert(*callee);
        auto replace =
            builder.create<LLVM::ZeroOp>(call.getLoc(), call.getType(0));
        call->replaceAllUsesWith(replace);
        call->erase();
        return;
      }
      */
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

          builder.create<enzymexla::MemcpyOp>(
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
  void runOnOperation() override {
    replaceRuntime();
    llvm::SmallVector<LLVM::LLVMFuncOp> launchFuncs;
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      auto symName = funcOp.getName();
      if (symName == "__mlir_cuda_caller_phase3")
        launchFuncs.push_back(funcOp);
    });

    auto ctx = getOperation()->getContext();

    gpu::GPUModuleOp gpuModule = nullptr;

    OpBuilder builder(getOperation()->getContext());

    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    SetVector<Operation *> tocopy;

    DenseMap<LLVM::LLVMFuncOp, SmallVector<CallOpInterface>> kernelLaunches;

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

          kernelLaunches[cur].push_back(cop);
        }
      }
    }

    for (auto &launch : kernelLaunches) {
      bool captured = false;
      auto kernelUses = launch.first.getSymbolUses(getOperation());
      for (auto use : *kernelUses) {
        auto user = dyn_cast<LLVM::AddressOfOp>(use.getUser());
        if (!user) {
          captured = true;
          break;
        }
        for (auto user2 : user->getResult(0).getUsers()) {
          auto user3 = dyn_cast<CallOpInterface>(user2);
          if (!user3) {
            captured = true;
            break;
          }
          if (!llvm::is_contained(launch.second, user3)) {
            captured = true;
            break;
          }
        }
      }
      gpu::GPUFuncOp gpufunc = nullptr;
      for (auto cop : launch.second) {
        auto cur = launch.first;

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

        bool local_use_launch_func = use_launch_func || captured;
        if (local_use_launch_func && !gpufunc) {
          initGPUModule(gpuModule, launch.first);
          builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
          gpufunc = builder.create<gpu::GPUFuncOp>(cur->getLoc(), cur.getName(),
                                                   gpuTy0);
          auto entry = &gpufunc.getBody().front();
          builder.setInsertionPointToEnd(entry);
          IRMapping map;
          gpufunc.getBody().getBlocks().clear();
          cur.getFunctionBody().cloneInto(&gpufunc.getBody(), map);

          if (auto comdat = cur.getComdat()) {
            cur.setComdatAttr({});
            auto comdatSelector =
                SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
            if (auto cselect =
                    dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
              cselect->erase();
            }
          }

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

          auto kernelSymbol =
              SymbolRefAttr::get(gpuModule.getNameAttr(),
                                 {SymbolRefAttr::get(gpufunc.getNameAttr())});
          for (auto use : *kernelUses) {
            if (auto occ = dyn_cast<enzymexla::GPUOccupancyOp>(use.getUser())) {
              occ.setFnAttr(kernelSymbol);
              continue;
            }
            auto user = dyn_cast<LLVM::AddressOfOp>(use.getUser());
            if (!user) {
              llvm::errs()
                  << " Error, could not replace kernel symbol in user(1): "
                  << *use.getUser() << "\n";
              continue;
            }
            builder.setInsertionPoint(user);
            auto k2 = builder.create<enzymexla::GPUKernelAddressOp>(
                user->getLoc(), user.getType(), kernelSymbol);
            for (auto &user2 : user->getResult(0).getUses()) {
              auto user3 = dyn_cast<CallOpInterface>(user2.getOwner());
              if (user3 && llvm::is_contained(launch.second, user3)) {
                continue;
              }
              user2.assign(k2);
            }
            user->erase();
          }
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
          if (local_use_launch_func)
            grid[i] = builder.create<LLVM::SExtOp>(loc, builder.getI64Type(),
                                                   cop.getArgOperands()[i + 1]);
          else
            grid[i] = builder.create<arith::IndexCastOp>(
                loc, builder.getIndexType(), cop.getArgOperands()[i + 1]);
        }
        Value block[3];
        for (int i = 0; i < 3; i++) {
          if (local_use_launch_func)
            block[i] = builder.create<LLVM::SExtOp>(
                loc, builder.getI64Type(), cop.getArgOperands()[i + 4]);
          else
            block[i] = builder.create<arith::IndexCastOp>(
                loc, builder.getIndexType(), cop.getArgOperands()[i + 4]);
        }
        if (stream.getDefiningOp<LLVM::ZeroOp>()) {
          if (local_use_launch_func) {
            builder.create<gpu::LaunchFuncOp>(
                loc, gpufunc, gpu::KernelDim3{grid[0], grid[1], grid[2]},
                gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                ValueRange(args));
          } else {
            auto op = builder.create<mlir::gpu::LaunchOp>(
                launch.first->getLoc(), grid[0], grid[1], grid[2], block[0],
                block[1], block[2], shMemSize, nullptr, ValueRange());
            builder.setInsertionPointToStart(&op.getRegion().front());
            builder.create<LLVM::CallOp>(loc, cur, args);
            builder.create<gpu::TerminatorOp>(loc);
          }
        } else {
          if (local_use_launch_func) {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            stream = builder.create<enzymexla::StreamToTokenOp>(
                loc, gpu::AsyncTokenType::get(ctx), stream);
            builder.create<gpu::LaunchFuncOp>(
                loc, gpufunc, gpu::KernelDim3{grid[0], grid[1], grid[2]},
                gpu::KernelDim3{block[0], block[1], block[2]}, shMemSize,
                ValueRange(args), stream.getType(), ValueRange(stream));
          } else {
            assert(isa<LLVM::LLVMPointerType>(stream.getType()));
            stream = builder.create<enzymexla::StreamToTokenOp>(
                loc, gpu::AsyncTokenType::get(ctx), stream);
            auto op = builder.create<mlir::gpu::LaunchOp>(
                launch.first->getLoc(), grid[0], grid[1], grid[2], block[0],
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

    if (gpuModule) {
      builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
      llvm::SmallSet<Operation *, 1> done;
      while (tocopy.size()) {
        auto cur = tocopy.pop_back_val();
        if (done.count(cur))
          continue;
        done.insert(cur);
        auto cloned = builder.clone(*cur);
        if (auto glob = dyn_cast<LLVM::GlobalOp>(cur)) {
          if (auto comdat = glob.getComdat()) {
            glob.setComdatAttr({});

            auto comdatSelector =
                SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
            if (auto cselect =
                    dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
              cselect->erase();
            }

            cast<LLVM::GlobalOp>(cloned).setComdatAttr({});
          }
        }
        if (auto glob = dyn_cast<LLVM::LLVMFuncOp>(cur)) {
          if (auto comdat = glob.getComdat()) {
            glob.setComdatAttr({});

            auto comdatSelector =
                SymbolTable::lookupNearestSymbolFrom(cur, *comdat);
            if (auto cselect =
                    dyn_cast<LLVM::ComdatSelectorOp>(comdatSelector)) {
              cselect->erase();
            }

            cast<LLVM::LLVMFuncOp>(cloned).setComdatAttr({});
          }
        }
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

    if (launchFuncs.size() && gpuModule)
      getOperation()->setAttr("gpu.container_module",
                              OpBuilder(ctx).getUnitAttr());
  }
};
