#ifndef CUBERT_GRAPH_H
#define CUBERT_GRAPH_H

#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/graph.pb.h"
#include "cuBERT/common.h"

namespace cuBERT {
    template <typename T>
    class Graph {
    public:
        explicit Graph(const char *filename);
        explicit Graph(int8_t *model_bytes, size_t byte_len);


        ~Graph();

        std::unordered_map<std::string, T *> var;
        size_t vocab_size;
        size_t type_vocab_size;
        size_t hidden_size;
        size_t intermediate_size;
        size_t num_labels;
    };
}

#endif //CUBERT_GRAPH_H
