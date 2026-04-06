#include "weight_io.h"
#include "../include/cuda_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Load a single tensor from a .bin file.
// Format: [int32 ndims][int32 shape...]
//         [float32 data...]
int load_tensor_from_file(
    const char* path, float** d_ptr
) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr,
            "weight_io: cannot open %s\n", path);
        return -1;
    }

    int ndims = 0;
    if (fread(&ndims, sizeof(int), 1, f) != 1) {
        fprintf(stderr,
            "weight_io: bad header %s\n", path);
        fclose(f);
        return -1;
    }

    // Read shape and compute total count
    std::vector<int> shape(ndims);
    if (fread(shape.data(), sizeof(int),
              ndims, f) != (size_t)ndims) {
        fprintf(stderr,
            "weight_io: bad shape %s\n", path);
        fclose(f);
        return -1;
    }

    int count = 1;
    for (int i = 0; i < ndims; i++) {
        count *= shape[i];
    }

    // Read float data
    std::vector<float> h(count);
    size_t read = fread(
        h.data(), sizeof(float), count, f);
    fclose(f);

    if ((int)read != count) {
        fprintf(stderr,
            "weight_io: truncated %s "
            "(got %d / %d)\n",
            path, (int)read, count);
        return -1;
    }

    // Upload to GPU
    CUDA_CHECK(cudaMalloc(
        d_ptr, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        *d_ptr, h.data(),
        count * sizeof(float),
        cudaMemcpyHostToDevice));

    return count;
}

// Save a single tensor to a .bin file.
void save_tensor_to_file(
    const char* path,
    const float* d_ptr,
    int ndims, const int* shape
) {
    int count = 1;
    for (int i = 0; i < ndims; i++) {
        count *= shape[i];
    }

    std::vector<float> h(count);
    CUDA_CHECK(cudaMemcpy(
        h.data(), d_ptr,
        count * sizeof(float),
        cudaMemcpyDeviceToHost));

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr,
            "weight_io: cannot write %s\n", path);
        return;
    }
    fwrite(&ndims, sizeof(int), 1, f);
    fwrite(shape, sizeof(int), ndims, f);
    fwrite(h.data(), sizeof(float), count, f);
    fclose(f);
}

// Try to load a tensor; return false if missing.
static bool try_load(
    const char* path, float** d_ptr, int expected
) {
    int n = load_tensor_from_file(path, d_ptr);
    if (n < 0) return false;
    if (n != expected) {
        fprintf(stderr,
            "weight_io: size mismatch %s "
            "(got %d, expected %d)\n",
            path, n, expected);
        cudaFree(*d_ptr);
        *d_ptr = nullptr;
        return false;
    }
    return true;
}

// Load all model weights from a directory.
bool load_model_weights_from_dir(
    const char* dir_path,
    const ModelConfig& cfg,
    ModelWeights& w
) {
    int E = cfg.embed_dim;
    int M = cfg.mlp_dim;
    int V = cfg.vocab_size;
    int L = cfg.num_layers;
    std::string dir(dir_path);
    if (!dir.empty() && dir.back() != '/'
        && dir.back() != '\\') {
        dir += '/';
    }

    w.layers.resize(L);

    // Per-layer weights
    const char* names[] = {
        "W_Q", "W_K", "W_V", "W_O",
        "W_mlp1", "W_mlp2",
        "rms_attn", "rms_mlp"
    };
    int sizes[] = {
        E*E, E*E, E*E, E*E,
        E*M, M*E, E, E
    };

    for (int l = 0; l < L; l++) {
        float** ptrs[] = {
            &w.layers[l].W_Q,
            &w.layers[l].W_K,
            &w.layers[l].W_V,
            &w.layers[l].W_O,
            &w.layers[l].W_mlp1,
            &w.layers[l].W_mlp2,
            &w.layers[l].rms_attn,
            &w.layers[l].rms_mlp
        };
        for (int j = 0; j < 8; j++) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                "%slayer%d_%s.bin",
                dir.c_str(), l, names[j]);
            if (!try_load(buf, ptrs[j], sizes[j])) {
                fprintf(stderr,
                    "weight_io: failed layer %d "
                    "%s, aborting load\n",
                    l, names[j]);
                return false;
            }
        }
    }

    // Final projection + norm
    char buf[512];
    snprintf(buf, sizeof(buf),
        "%sW_out.bin", dir.c_str());
    if (!try_load(buf, &w.W_out, E * V)) {
        return false;
    }

    snprintf(buf, sizeof(buf),
        "%srms_final.bin", dir.c_str());
    if (!try_load(buf, &w.rms_final, E)) {
        return false;
    }

    printf("  Loaded weights from %s "
           "(%d layers)\n", dir_path, L);
    return true;
}
