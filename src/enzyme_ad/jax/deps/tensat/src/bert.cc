// NOTE: This model is for the original Tensat API, which is incompatible with the new StableHLO-based ops.
// TODO: Change BERT model to use StableHLO ops

#include "rust/cxx.h"
#include "cxxbridge/deps/tensat/src/input.rs.h"
#include <utility>
#include <bits/stdc++.h>

using namespace rust::cxxbridge1;


std::pair<TensorInfo*, TensorInfo*> attention(
    Box<CppGraphConverter> &graph,
    TensorInfo &input,
    int32_t heads,
    int32_t input_dim_1
) { 
    int d_model = input_dim_1;
    int d_k = d_model / heads;
    assert(input_dim_1 % heads == 0);
    std::vector<Box<TensorInfo>> weights;
    for (int i = 0; i < 3; i++) {
        int32_t dims[] = {d_model, d_model};
        weights.push_back(graph->new_weight(Slice<const int32_t>{dims, 2}));
    }

    // compute query, key, value tensors
    auto q = graph->matmul(input, *weights[0]);
    auto k = graph->matmul(input, *weights[1]);
    auto v = graph->matmul(input, *weights[2]);

    int32_t reshape_shape[] = {64, 16, 64};
    auto reshape_slice = Slice<const int32_t>{reshape_shape, 2};
    q = graph->reshape(*q, reshape_slice);
    k = graph->reshape(*k, reshape_slice);
    v = graph->reshape(*v, reshape_slice);

    int32_t transpose_perm[] = {1, 0, 2};
    auto transpose_slice = Slice<const int32_t>{transpose_perm, 3};
    q = graph->transpose(*q, transpose_slice, true);
    k = graph->transpose(*k, transpose_slice, true);
    v = graph->transpose(*v, transpose_slice, true);

    auto logits = graph->matmul(*q, *k);
    auto output = graph->matmul(*logits, *v);

    output = graph->transpose(*output, transpose_slice, true);
    int32_t reshape_shape2[] = {64, 1024};
    auto reshape_slice2 = Slice<const int32_t>{reshape_shape2, 2};
    output = graph->reshape(*output, reshape_slice2);

    int32_t dims[] = {d_model, d_model};
    auto linear = graph->new_weight(Slice<const int32_t>{dims, 2});
    auto next_in = graph->matmul(input, *linear);
    
    // return std::make_pair(next_in, output);
    return std::make_pair(&(*next_in), &(*output));
}


const int SEQ_LENGTH = 64;
const int HIDDEN_DIMS = 1024;

int main() {
    auto graph = new_converter();
    
    int dims[2] = {SEQ_LENGTH, HIDDEN_DIMS};
    auto input_slice = rust::Slice<const int32_t>{dims, 2};
    
    auto input = graph->new_input(input_slice);
    auto relu = graph->relu(*input);
    
    auto [next_in, output] = attention(graph, *input, 16, HIDDEN_DIMS);

    auto tmp = next_in;
    auto current = output;
    
    for (int i = 1; i < 8; i++) {
        auto [next_in, output] = attention(graph, *tmp, 16, HIDDEN_DIMS);
        tmp = next_in;
        current = &*graph->noop(*current, *output);
    }

    current = &*graph->noop(*current, *tmp);
    
    graph->print_rec_expr();
}
