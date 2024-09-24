fn main() {
    println!("cargo:rerun-if-changed=src/input.rs");
    println!("cargo:rerun-if-changed=src/graph.cc");
    println!("cargo:rerun-if-changed=include/tensat.h");

    // C++ graph input bindings
    cxx_build::bridge("src/input.rs")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-lc++")
        .flag_if_supported("-lc++abi")
        .compile("tensatcpp");
}
