#include "rust/cxx.h"
#include "cxxbridge/deps/tensat/src/input.rs.h"
#include <utility>

int main() {
    auto graphBox = tensat::new_converter();
    // int dims[2] = {1024, 1024};
    int dims_1[0] = {};
    int dims_2[0] = {};
    auto input_slice_1 = rust::Slice<const int32_t>{dims_1, 0};
    auto input_slice_2 = rust::Slice<const int32_t>{dims_2, 0};
    auto inp1 = graphBox->new_input(input_slice_1);
    auto inp2 = graphBox->new_input(input_slice_2);
    auto mul = graphBox->new_mul_op(*inp1, *inp2, 0);
    // auto relu = graphBox->new_tanh_op(*inp, 0);

    graphBox->print_rec_expr();
}
