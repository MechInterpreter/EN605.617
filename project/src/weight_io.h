#ifndef WEIGHT_IO_H
#define WEIGHT_IO_H

#include "../include/types.h"
#include <string>

// Load a single tensor from a .bin file.
// Format: [int32 ndims] [int32 shape[0]]...
//         [float32 data...]
// Returns element count; allocates GPU memory.
int load_tensor_from_file(
    const char* path, float** d_ptr);

// Load all model weights from a directory.
// Expects files named: layer{L}_{name}.bin
//   where name = W_Q, W_K, W_V, W_O,
//                W_mlp1, W_mlp2, rms_attn, rms_mlp
// Plus: W_out.bin, rms_final.bin
// Returns true on success.
bool load_model_weights_from_dir(
    const char* dir_path,
    const ModelConfig& cfg,
    ModelWeights& w);

// Save a single tensor to a .bin file.
// Reads from GPU memory, writes to disk.
void save_tensor_to_file(
    const char* path,
    const float* d_ptr,
    int ndims, const int* shape);

#endif // WEIGHT_IO_H
