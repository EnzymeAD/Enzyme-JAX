#ifndef GPULOWERING_H_
#define GPULOWERING_H_

namespace mlir {
class RewritePatternSet;
class LLVMTypeConverter;
void populateGPULoweringPatterns(RewritePatternSet &patterns,
                                 LLVMTypeConverter &typeConverter);
} // namespace mlir


#endif // GPULOWERING_H_
