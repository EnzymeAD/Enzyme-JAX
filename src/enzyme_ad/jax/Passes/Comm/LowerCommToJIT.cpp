#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
#include "src/enzyme_ad/jax/Passes/Comm/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LOWERCOMMTOJITPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

extern "C" void *EnzymeJaXLookupSymbol(const char *name);

const char *convertMlirTypeToMpiDatatypeName(Type type,
                                             bool allow_cast = false) {
  // case ffi::DataType::INVALID: return nullptr;
  if (type.isInteger(1)) /*ffi::DataType::PRED:*/
    return "MPI_C_BOOL";
  // case ffi::DataType::S1: return nullptr;
  // case ffi::DataType::S2: return nullptr;
  // case ffi::DataType::S4: return nullptr;
  if (type.isInteger(8)) /*ffi::DataType::S8:*/
    return "MPI_INT8_T";
  if (type.isInteger(16)) /*ffi::DataType::S16:*/
    return "MPI_INT16_T";
  if (type.isInteger(32)) /*ffi::DataType::S32:*/
    return "MPI_INT32_T";
  if (type.isInteger(64)) /*ffi::DataType::S64:*/
    return "MPI_INT64_T";
  // case ffi::DataType::U1: return nullptr;
  // case ffi::DataType::U2: return nullptr;
  // case ffi::DataType::U4: return nullptr;
  if (type.isUnsignedInteger(8)) /*ffi::DataType::U8:*/
    return "MPI_UINT8_T";
  if (type.isUnsignedInteger(16)) /*ffi::DataType::U16:*/
    return "MPI_UINT16_T";
  if (type.isUnsignedInteger(32)) /*ffi::DataType::U32:*/
    return "MPI_UINT32_T";
  if (type.isUnsignedInteger(64)) /*ffi::DataType::U64:*/
    return "MPI_UINT64_T";
  if (type.isFloat(16)) /*ffi::DataType::F16:*/
    return (allow_cast ? "MPI_UINT16_T" : nullptr);
  if (type.isF32()) /*ffi::DataType::F32:*/
    return "MPI_FLOAT";
  if (type.isF64()) /*ffi::DataType::F64:*/
    return "MPI_DOUBLE";
  if (type.isBF16()) /*ffi::DataType::BF16:*/
    return (allow_cast ? "MPI_UINT16_T" : nullptr);
  if (auto complex_type = dyn_cast<ComplexType>(type)) {
    if (complex_type.getElementType().isF32()) /*ffi::DataType::C64:*/
      return "MPI_C_FLOAT_COMPLEX";
    if (complex_type.getElementType().isF64()) /*ffi::DataType::C128:*/
      return "MPI_C_DOUBLE_COMPLEX";
    else
      return nullptr;
  }
  // case ffi::DataType::TOKEN: return nullptr;
  //   if (type) /*ffi::DataType::F8E5M2:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E4M3:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E4M3FN:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E4M3B11FNUZ:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E5M2FNUZ:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E4M3FNUZ:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  //   if (type) /*ffi::DataType::F8E3M4:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  // case ffi::DataType::F4E2M1FN: return nullptr;
  //   if (isa<>(type)) /*ffi::DataType::F8E8M0FNU:*/
  //     return (allow_cast ? "MPI_UINT8_T" : nullptr);
  else
    return nullptr;
}

// void *convertMlirTypeToMpiDatatype(Type type, bool allow_cast = false) {
//   const char *name = convertMlirTypeToMpiDatatypeName(type, allow_cast);
//   return nullptr;

//   auto dt = reinterpret_cast<MPI_Datatype>(EnzymeJaXLookupSymbol(name));
//   if (dt == nullptr) {
//     return ffi::Error::Internal(
//         absl::StrFormat("MPI: symbol `%s` not found", name));
//   }

//   return dt;
// }

struct LowerCommMpiConstantOpToJIT
    : public OpConversionPattern<comm::MpiConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    auto restype = converter->convertType(op.getResult().getType());
    if (restype == nullptr)
      return failure();

    llvm::StringRef name;
    auto value_attr = op.getValue();
    if (auto attr = cast<comm::MpiCommAttr>(value_attr)) {
      name = comm::stringifyMpiCommEnum(attr.getValue());
    } else if (auto attr = cast<comm::MpiOpAttr>(value_attr)) {
      name = comm::stringifyMpiOpEnum(attr.getValue());
    } else {
      return rewriter.notifyMatchFailure(
          op, "MPI constant is not a valid attribute");
    }

    void *value_abi = EnzymeJaXLookupSymbol(name.data());
    if (value_abi == nullptr) {
      return rewriter.notifyMatchFailure(op, "MPI constant `" + name +
                                                 "` not found");
    }

    uint64_t value = reinterpret_cast<uint64_t>(value_abi);
    auto constant_attr = SplatElementsAttr::get(
        RankedTensorType::get({}, rewriter.getIntegerType(64)),
        ArrayRef(APInt(64, value)));

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, restype, cast<ElementsAttr>(constant_attr));

    return success();
  }
};

struct LowerCommMpiCommRankOpToJIT
    : public OpConversionPattern<comm::MpiCommRankOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);

    std::string mpiFunctionName = "MPI_Comm_rank";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_void, {type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"),
      // rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects",
      // memoryEffectsAttr);

      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_comm_ptr = entryBlock->getArgument(0);
      Value arg_rank_ptr = entryBlock->getArgument(1);

      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{comm, arg_rank_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32, {type_ptr, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    auto comm = adaptor.getComm();
    auto rank_placeholder = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, ArrayRef<int32_t>{-1}));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/1,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i32,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{comm, rank_placeholder},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    return success();
  }
};

struct LowerCommMpiCommSizeOpToJIT
    : public OpConversionPattern<comm::MpiCommSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);

    std::string mpiFunctionName = "MPI_Comm_size";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_void, {type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_comm_ptr = entryBlock->getArgument(0);
      Value arg_size_ptr = entryBlock->getArgument(1);

      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{comm, arg_size_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32, {type_ptr, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    auto comm = adaptor.getComm();
    auto size_placeholder = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, ArrayRef<int32_t>{-1}));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/1,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i32,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{comm, size_placeholder},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    return success();
  }
};

struct LowerCommMpiCommSplitOpToJIT
    : public OpConversionPattern<comm::MpiCommSplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Comm_split";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_void, {type_ptr, type_ptr, type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_comm_ptr = entryBlock->getArgument(0);
      Value arg_color_ptr = entryBlock->getArgument(1);
      Value arg_key_ptr = entryBlock->getArgument(2);
      Value arg_newcomm_ptr = entryBlock->getArgument(3);

      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();
      Value color =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_color_ptr)
              .getResult();
      Value key =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_key_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{comm, color, key, arg_newcomm_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_i32, {type_ptr, type_i32, type_i32, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    auto comm = adaptor.getComm();
    auto color = adaptor.getColor();
    auto key = adaptor.getKey();
    auto newcomm_placeholder = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64, ArrayRef<int64_t>{-1}));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/3,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i64,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{comm, color, key, newcomm_placeholder},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiBarrierOpToJIT
    : public OpConversionPattern<comm::MpiBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);

    std::string mpiFunctionName = "MPI_Barrier";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(type_void, {type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_comm_ptr = entryBlock->getArgument(0);

      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{comm});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(type_i32, {type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    auto comm = adaptor.getComm();

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, TypeRange{},
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{comm},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiSendOpToJIT : public OpConversionPattern<comm::MpiSendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Send";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_void,
          {type_ptr, type_ptr, type_ptr, type_ptr, type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_buffer_ptr = entryBlock->getArgument(0);
      Value arg_count_ptr = entryBlock->getArgument(1);
      Value arg_datatype_ptr = entryBlock->getArgument(2);
      Value arg_dest_ptr = entryBlock->getArgument(3);
      Value arg_tag_ptr = entryBlock->getArgument(4);
      Value arg_comm_ptr = entryBlock->getArgument(5);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value dest =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_dest_ptr)
              .getResult();
      Value tag =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_tag_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(
          rewriter, op.getLoc(), TypeRange{type_i32},
          SymbolRefAttr::get(context, mpiFunctionName),
          ValueRange{arg_buffer_ptr, count, datatype, dest, tag, comm});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_i32,
          {type_ptr, type_i32, type_ptr, type_i32, type_i32, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getBuffer().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "support for dynamic buffer shape is not implemented yet");
    }

    auto len = std::reduce(op.getBuffer().getType().getShape().begin(),
                           op.getBuffer().getType().getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto buffer = adaptor.getBuffer();
    auto dest = adaptor.getDest();
    auto tag = adaptor.getTag();
    auto comm = adaptor.getComm();

    // TODO can we pass `datatype` as attribute?
    auto datatype_name = convertMlirTypeToMpiDatatypeName(
        op.getBuffer().getType().getElementType(),
        /*allow_cast=*/true);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i64,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{buffer, count, datatype, dest, tag, comm},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiIsendOpToJIT : public OpConversionPattern<comm::MpiIsendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIsendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Isend";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_void,
                                      {type_ptr, type_ptr, type_ptr, type_ptr,
                                       type_ptr, type_ptr, type_ptr},
                                      false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_buffer_ptr = entryBlock->getArgument(0);
      Value arg_count_ptr = entryBlock->getArgument(1);
      Value arg_datatype_ptr = entryBlock->getArgument(2);
      Value arg_dest_ptr = entryBlock->getArgument(3);
      Value arg_tag_ptr = entryBlock->getArgument(4);
      Value arg_comm_ptr = entryBlock->getArgument(5);
      Value arg_request_ptr = entryBlock->getArgument(6);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value dest =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_dest_ptr)
              .getResult();
      Value tag =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_tag_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{arg_buffer_ptr, count, datatype, dest,
                                      tag, comm, arg_request_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_i32,
          {type_ptr, type_i32, type_ptr, type_i32, type_i32, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getBuffer().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "support for dynamic buffer shape is not implemented yet");
    }

    auto len = std::reduce(op.getBuffer().getType().getShape().begin(),
                           op.getBuffer().getType().getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto buffer = adaptor.getBuffer();
    auto dest = adaptor.getDest();
    auto tag = adaptor.getTag();
    auto comm = adaptor.getComm();

    auto request_placeholder = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64, -1));

    // TODO can we pass `datatype` as attribute?
    auto datatype_name = convertMlirTypeToMpiDatatypeName(
        op.getBuffer().getType().getElementType(),
        /*allow_cast=*/true);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/6,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i64,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{buffer, count, datatype, dest, tag, comm,
                   request_placeholder},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiRecvOpToJIT : public OpConversionPattern<comm::MpiRecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Recv";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      // we are ignoring `MPI_Status` argument for now
      auto funcType = LLVM::LLVMFunctionType::get(
          type_void,
          {type_ptr, type_ptr, type_ptr, type_ptr, type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_buffer_ptr = entryBlock->getArgument(0);
      Value arg_count_ptr = entryBlock->getArgument(1);
      Value arg_datatype_ptr = entryBlock->getArgument(2);
      Value arg_src_ptr = entryBlock->getArgument(3);
      Value arg_tag_ptr = entryBlock->getArgument(4);
      Value arg_comm_ptr = entryBlock->getArgument(5);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value src =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_src_ptr)
              .getResult();
      Value tag =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_tag_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();
      Value status =
          LLVM::ZeroOp::create(rewriter, op.getLoc(), type_ptr).getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(
          rewriter, op.getLoc(), TypeRange{type_i32},
          SymbolRefAttr::get(context, mpiFunctionName),
          ValueRange{arg_buffer_ptr, count, datatype, src, tag, comm, status});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32,
                                      {type_ptr, type_i32, type_ptr, type_i32,
                                       type_i32, type_ptr, type_ptr},
                                      false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getResult().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic buffer shape is not supported");
    }

    auto type_buffer = op.getResult().getType();
    auto buffer_placeholder = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_buffer, DenseIntElementsAttr::get(type_buffer, -1));

    auto len = std::reduce(type_buffer.getShape().begin(),
                           type_buffer.getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto src = adaptor.getSource();
    auto tag = adaptor.getTag();
    auto comm = adaptor.getComm();

    // TODO can we pass `datatype` as attribute?
    auto datatype_name =
        convertMlirTypeToMpiDatatypeName(type_buffer.getElementType(),
                                         /*allow_cast=*/true);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/0,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i64,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{buffer_placeholder, count, datatype, src, tag, comm},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiIrecvOpToJIT : public OpConversionPattern<comm::MpiIrecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIrecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Irecv";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_void,
                                      {type_ptr, type_ptr, type_ptr, type_ptr,
                                       type_ptr, type_ptr, type_ptr},
                                      false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_buffer_ptr = entryBlock->getArgument(0);
      Value arg_count_ptr = entryBlock->getArgument(1);
      Value arg_datatype_ptr = entryBlock->getArgument(2);
      Value arg_src_ptr = entryBlock->getArgument(3);
      Value arg_tag_ptr = entryBlock->getArgument(4);
      Value arg_comm_ptr = entryBlock->getArgument(5);
      Value arg_request_ptr = entryBlock->getArgument(6);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value src =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_src_ptr)
              .getResult();
      Value tag =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_tag_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{arg_buffer_ptr, count, datatype, src, tag,
                                      comm, arg_request_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32,
                                      {type_ptr, type_i32, type_ptr, type_i32,
                                       type_i32, type_ptr, type_ptr},
                                      false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getBuffer().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic buffer shape is not supported");
    }

    auto type_buffer = op.getBuffer().getType();
    auto buffer_placeholder = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_buffer, DenseIntElementsAttr::get(type_buffer, -1));

    auto len = std::reduce(type_buffer.getShape().begin(),
                           type_buffer.getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto src = adaptor.getSource();
    auto tag = adaptor.getTag();
    auto comm = adaptor.getComm();

    auto request_placeholder = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64, -1));

    // TODO can we pass `datatype` as attribute?
    auto datatype_name =
        convertMlirTypeToMpiDatatypeName(type_buffer.getElementType(),
                                         /*allow_cast=*/true);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    auto aliases = rewriter.getArrayAttr({
        /* buffer */
        stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/0,
            /*operandTupleIndices=*/ArrayRef<int64_t>{}),
        /* request */
        stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/6,
            /*operandTupleIndices=*/ArrayRef<int64_t>{}),
    });

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_tensor_i64,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{buffer_placeholder, count, datatype, src, tag, comm,
                   request_placeholder},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiWaitOpToJIT : public OpConversionPattern<comm::MpiWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);

    std::string mpiFunctionName = "MPI_Wait";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      // we are ignoring `MPI_Status` argument for now
      auto funcType = LLVM::LLVMFunctionType::get(type_void, {type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_request_ptr = entryBlock->getArgument(0);
      Value status_ptr =
          LLVM::ZeroOp::create(rewriter, op.getLoc(), type_ptr).getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{arg_request_ptr, status_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32, {type_ptr, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    auto request = adaptor.getRequest();

    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, TypeRange{},
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{request},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiWaitallOpToJIT
    : public OpConversionPattern<comm::MpiWaitallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);

    auto num_requests = op.getNumOperands();
    std::string mpiFunctionName = "MPI_Waitall_" + std::to_string(num_requests);
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      // we are ignoring `MPI_Status` argument for now
      SmallVector<Type> argTypes(num_requests, type_ptr);
      auto funcType = LLVM::LLVMFunctionType::get(type_void, argTypes, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value count = rewriter
                        .create<LLVM::ConstantOp>(
                            op.getLoc(), type_i32,
                            rewriter.getI32IntegerAttr(num_requests))
                        .getResult();

      Value array_of_requests =
          LLVM::AllocaOp::create(rewriter, op.getLoc(), type_ptr, type_ptr,
                                 count)
              .getResult();

      for (int i = 0; i < num_requests; ++i) {
        auto gep_op =
            LLVM::GEPOp::create(rewriter, op.getLoc(), type_ptr, type_ptr,
                                array_of_requests, ValueRange{});
        gep_op.setRawConstantIndices({i});
        LLVM::StoreOp::create(rewriter, op.getLoc(), entryBlock->getArgument(i),
                              gep_op.getResult());
      }

      Value array_of_requests_ptr =
          LLVM::GEPOp::create(rewriter, op.getLoc(), type_ptr, type_ptr,
                              array_of_requests, ValueRange{})
              .getResult();

      // TODO get MPI_STATUS_IGNORE constant
      Value status_ptr =
          LLVM::ZeroOp::create(rewriter, op.getLoc(), type_ptr).getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(
          rewriter, op.getLoc(), TypeRange{type_i32},
          SymbolRefAttr::get(context, mpiFunctionName),
          ValueRange{count, array_of_requests_ptr, status_ptr});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(type_i32, {type_ptr, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, TypeRange{},
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        adaptor.getRequests(),
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiAllreduceOpToJIT
    : public OpConversionPattern<comm::MpiAllreduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiAllreduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Allreduce";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_void,
          {type_ptr, type_ptr, type_ptr, type_ptr, type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_sendbuf_ptr = entryBlock->getArgument(0);
      Value arg_recvbuf_ptr = entryBlock->getArgument(1);
      Value arg_count_ptr = entryBlock->getArgument(2);
      Value arg_datatype_ptr = entryBlock->getArgument(3);
      Value arg_op_ptr = entryBlock->getArgument(4);
      Value arg_comm_ptr = entryBlock->getArgument(5);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value op_val =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_op_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{type_i32},
                           SymbolRefAttr::get(context, mpiFunctionName),
                           ValueRange{arg_sendbuf_ptr, arg_recvbuf_ptr, count,
                                      datatype, op_val, comm});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_i32,
          {type_ptr, type_ptr, type_i32, type_ptr, type_ptr, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getResult().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic buffer shape is not supported");
    }

    auto sendbuf = adaptor.getSendbuf();
    auto type_buffer = op.getResult().getType();
    auto len = std::reduce(type_buffer.getShape().begin(),
                           type_buffer.getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto recvbuf_placeholder = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_buffer, DenseIntElementsAttr::get(type_buffer, -1));

    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto comm = adaptor.getComm();

    // TODO can we pass `datatype` and `op` as attribute?
    auto datatype_name =
        convertMlirTypeToMpiDatatypeName(type_buffer.getElementType(),
                                         /*allow_cast=*/false);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    auto mpi_op_name =
        comm::stringifyMpiOpEnum(adaptor.getReduceOp().getValue());
    auto mpi_op_val = EnzymeJaXLookupSymbol(mpi_op_name.data());
    if (mpi_op_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(mpi_op_name) + "` not found");
    }

    Value mpi_op = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(mpi_op_val)));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/1,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_buffer,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{sendbuf, recvbuf_placeholder, count, datatype, mpi_op, comm},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommMpiBcastOpToJIT : public OpConversionPattern<comm::MpiBcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto type_ptr = LLVM::LLVMPointerType::get(context);
    auto type_void = LLVM::LLVMVoidType::get(context);
    auto type_i32 = IntegerType::get(context, 32);
    auto type_i64 = IntegerType::get(context, 64);
    auto type_tensor_i32 = RankedTensorType::get({}, type_i32);
    auto type_tensor_i64 = RankedTensorType::get({}, type_i64);

    std::string mpiFunctionName = "MPI_Bcast";
    std::string wrapperFunctionName = "enzymexla_jitwrap_" + mpiFunctionName;

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_void, {type_ptr, type_ptr, type_ptr, type_ptr, type_ptr}, false);

      auto wrapperFunc = LLVM::LLVMFuncOp::create(
          rewriter, op.getLoc(), wrapperFunctionName, funcType);

      Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);

      // Add function-level memory effects attribute
      // auto memoryEffectsAttr = rewriter.getArrayAttr(
      //     {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
      //      rewriter.getStringAttr("allocate"),
      //      rewriter.getStringAttr("free")});
      // wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

      // Add argument-level memory effects attribute
      // wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
      // memoryEffectsAttr);

      Value arg_buffer_ptr = entryBlock->getArgument(0);
      Value arg_count_ptr = entryBlock->getArgument(1);
      Value arg_datatype_ptr = entryBlock->getArgument(2);
      Value arg_root_ptr = entryBlock->getArgument(3);
      Value arg_comm_ptr = entryBlock->getArgument(4);

      Value count =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_count_ptr)
              .getResult();
      Value datatype = LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr,
                                            arg_datatype_ptr)
                           .getResult();
      Value root =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_i32, arg_root_ptr)
              .getResult();
      Value comm =
          LLVM::LoadOp::create(rewriter, op.getLoc(), type_ptr, arg_comm_ptr)
              .getResult();

      // TODO error checking
      // currently, we ignore the int return code
      LLVM::CallOp::create(
          rewriter, op.getLoc(), TypeRange{type_i32},
          SymbolRefAttr::get(context, mpiFunctionName),
          ValueRange{arg_buffer_ptr, count, datatype, root, comm});

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          type_i32, {type_ptr, type_i32, type_ptr, type_i32, type_ptr}, false);

      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName, funcType,
                               LLVM::Linkage::External);
    }

    if (!op.getResult().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic buffer shape is not supported");
    }

    auto buffer = adaptor.getInBuffer();
    auto type_buffer = op.getResult().getType();

    auto len = std::reduce(type_buffer.getShape().begin(),
                           type_buffer.getShape().end(), 1,
                           std::multiplies<int64_t>());
    auto count = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i32,
        DenseIntElementsAttr::get(type_tensor_i32, len));

    auto root = adaptor.getRoot();
    auto comm = adaptor.getComm();

    // TODO can we pass `datatype` as attribute?
    auto datatype_name =
        convertMlirTypeToMpiDatatypeName(type_buffer.getElementType(),
                                         /*allow_cast=*/true);
    auto datatype_val = EnzymeJaXLookupSymbol(datatype_name);
    if (datatype_val == nullptr) {
      return rewriter.notifyMatchFailure(
          op, "Symbol `" + std::string(datatype_name) + "` not found");
    }

    Value datatype = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tensor_i64,
        DenseIntElementsAttr::get(type_tensor_i64,
                                  reinterpret_cast<int64_t>(datatype_val)));

    auto aliases =
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            context,
            /*outputTupleIndices=*/ArrayRef<int64_t>{},
            /*operandIndex=*/0,
            /*operandTupleIndices=*/ArrayRef<int64_t>{})});

    // TODO revise if it is side effect free
    rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, type_buffer,
        mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
        ValueRange{buffer, count, datatype, root, comm},
        /*backend_config=*/rewriter.getStringAttr(""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/aliases,
        /*xla_side_effect_free=*/nullptr);

    return success();
  }
};

struct LowerCommToJITPass
    : public mlir::comm::impl::LowerCommToJITPassBase<LowerCommToJITPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<comm::CommDialect>();

    comm::StablehloTypeConverter converter;

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });

    // lower comm.mpi ops to stablehlo.custom_call ops
    RewritePatternSet patterns(context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);

    patterns.add<LowerCommMpiConstantOpToJIT, LowerCommMpiCommRankOpToJIT,
                 LowerCommMpiCommSizeOpToJIT, LowerCommMpiCommSplitOpToJIT,
                 LowerCommMpiBarrierOpToJIT, LowerCommMpiSendOpToJIT,
                 LowerCommMpiIsendOpToJIT, LowerCommMpiRecvOpToJIT,
                 LowerCommMpiIrecvOpToJIT, LowerCommMpiWaitOpToJIT,
                 LowerCommMpiWaitallOpToJIT, LowerCommMpiAllreduceOpToJIT,
                 LowerCommMpiBcastOpToJIT>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
