// RUN: enzymexlamlir-opt --split-input-file --preserve-triton-warps-ctas="save=true restore=false" %s | FileCheck %s --check-prefix=SAVE
// RUN: enzymexlamlir-opt --split-input-file --convert-triton-to-tritongpu="target=cuda:100" --preserve-triton-warps-ctas="save=false restore=true" %s | FileCheck %s --check-prefix=RESTORE

module attributes {"ttg.num-ctas" = 5 : i32, "ttg.num-warps" = 5 : i32, "ttg.threads-per-warp" = 64 : i32} {
    func.func @main() -> tensor<1x1x1xf32> {
        %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x1xf32>
        return %0 : tensor<1x1x1xf32>
    }
}

// SAVE: module attributes {"enzymexla.ttg.num-ctas" = 5 : i32, "enzymexla.ttg.num-warps" = 5 : i32, "enzymexla.ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 5 : i32, "ttg.num-warps" = 5 : i32, "ttg.threads-per-warp" = 64 : i32} {

// -----

module attributes {"enzymexla.ttg.num-ctas" = 5 : i32, "enzymexla.ttg.num-warps" = 5 : i32, "enzymexla.ttg.threads-per-warp" = 12 : i32} {
    func.func @main() -> tensor<1x1x1xf32> {
        %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x1xf32>
        return %0 : tensor<1x1x1xf32>
    }
}

// RESTORE: module attributes {"ttg.num-ctas" = 5 : i32, "ttg.num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 12 : i32} {
