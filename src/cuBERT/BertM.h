#ifndef CUBERT_BERTMGPU_H
#define CUBERT_BERTMGPU_H

#include <vector>
#include <string>
#include <atomic>
#include <mutex>

#include "cuBERT.h"
#include "cuBERT/Bert.h"
#include "cuBERT/tensorflow/Graph.h"

namespace cuBERT {
    template <typename T>
    class BertM {
    public:
        explicit BertM(const char *model_file,
                          size_t max_batch_size,
                          size_t seq_length,
                          size_t num_hidden_layers = 12,
                          size_t num_attention_heads = 12);
                          
        explicit BertM(int8_t *model_bytes,
                          size_t byte_len,
                          size_t max_batch_size,
                          size_t seq_length,
                          size_t num_hidden_layers = 12,
                          size_t num_attention_heads = 12);

        virtual ~BertM();

        unsigned int compute(size_t batch_size,
                             int *input_ids, int8_t *input_mask, int8_t *segment_ids,
                             T *output,
                             cuBERT_OutputType output_type = cuBERT_LOGITS);

        size_t seq_length;

    private:
        Graph<T> graph;
        std::vector<Bert<T> *> bert_instances;
        std::vector<std::mutex *> mutex_instances;

        std::atomic<uint8_t> rr;
    };
}

#endif //CUBERT_BERTMGPU_H
