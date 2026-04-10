// main.cpp -- CLI entry point for the OpenCL
// batched causal ablation engine.

#include <iostream>
#include <cstring>
#include <cstdlib>
#include "../include/config.h"
#include "../include/types.h"
#include "cl_setup.h"
#include "device_info.h"
#include "ablation_engine.h"

// Print usage
static void print_usage(const char *prog)
{
    std::cout
        << "\nUsage: " << prog << " [options]\n"
        << "\nOptions:\n"
        << "  --platform <N>       "
        << "Platform index (default: "
        << DEFAULT_PLATFORM << ")\n"
        << "  --device <N>         "
        << "Device index (default: "
        << DEFAULT_DEVICE << ")\n"
        << "  --seq-len <N>        "
        << "Sequence length (default: "
        << DEFAULT_SEQ_LEN << ")\n"
        << "  --embed-dim <N>      "
        << "Embedding dim, must be %4==0 "
        << "(default: " << DEFAULT_EMBED_DIM
        << ")\n"
        << "  --num-heads <N>      "
        << "Number of attention heads "
        << "(default: " << DEFAULT_NUM_HEADS
        << ")\n"
        << "  --interventions <N>  "
        << "Number of interventions "
        << "(default: "
        << DEFAULT_NUM_INTERVENTIONS << ")\n"
        << "  --batch-size <N>     "
        << "Interventions per batch "
        << "(default: "
        << DEFAULT_BATCH_SIZE << ")\n"
        << "  --iterations <N>     "
        << "Timing iterations (default: "
        << DEFAULT_ITERATIONS << ")\n"
        << "  --use-map            "
        << "Use map/unmap instead of R/W\n"
        << "  --profile            "
        << "Enable kernel profiling\n"
        << "  --info               "
        << "Display device info and exit\n"
        << "  --help               "
        << "Show this message\n"
        << std::endl;
}

// Parse CLI arguments
static RunConfig parse_args(
    int argc, char **argv)
{
    RunConfig cfg;
    cfg.platform_idx      = DEFAULT_PLATFORM;
    cfg.device_idx        = DEFAULT_DEVICE;
    cfg.seq_len           = DEFAULT_SEQ_LEN;
    cfg.embed_dim         = DEFAULT_EMBED_DIM;
    cfg.num_heads         = DEFAULT_NUM_HEADS;
    cfg.num_interventions =
        DEFAULT_NUM_INTERVENTIONS;
    cfg.batch_size        = DEFAULT_BATCH_SIZE;
    cfg.iterations        = DEFAULT_ITERATIONS;
    cfg.use_map           = 0;
    cfg.profile           = 0;
    cfg.show_info         = 0;
    cfg.show_help         = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--platform") == 0
            && i + 1 < argc) {
            cfg.platform_idx = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--device") == 0
                 && i + 1 < argc) {
            cfg.device_idx = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seq-len") == 0
                 && i + 1 < argc) {
            cfg.seq_len = atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--embed-dim") == 0
            && i + 1 < argc) {
            cfg.embed_dim = atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--num-heads") == 0
            && i + 1 < argc) {
            cfg.num_heads = atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--interventions") == 0
            && i + 1 < argc) {
            cfg.num_interventions =
                atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--batch-size") == 0
            && i + 1 < argc) {
            cfg.batch_size = atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--iterations") == 0
            && i + 1 < argc) {
            cfg.iterations = atoi(argv[++i]);
        }
        else if (
            strcmp(argv[i], "--use-map") == 0) {
            cfg.use_map = 1;
        }
        else if (
            strcmp(argv[i], "--profile") == 0) {
            cfg.profile = 1;
        }
        else if (
            strcmp(argv[i], "--info") == 0) {
            cfg.show_info = 1;
        }
        else if (
            strcmp(argv[i], "--help") == 0) {
            cfg.show_help = 1;
        }
        else {
            std::cerr << "Unknown option: "
                      << argv[i] << std::endl;
            cfg.show_help = 1;
        }
    }
    return cfg;
}

// Validate configuration (float4 alignment)
static int validate_config(const RunConfig *cfg)
{
    if (cfg->embed_dim % 4 != 0) {
        std::cerr
            << "embed-dim must be divisible by 4"
            << std::endl;
        return -1;
    }
    if (cfg->embed_dim % cfg->num_heads != 0) {
        std::cerr
            << "embed-dim must be divisible by "
            << "num-heads" << std::endl;
        return -1;
    }
    int hd = cfg->embed_dim / cfg->num_heads;
    if (hd % 4 != 0) {
        std::cerr
            << "head-dim (embed/heads=" << hd
            << ") must be divisible by 4"
            << std::endl;
        return -1;
    }
    if (cfg->batch_size <= 0
        || cfg->num_interventions <= 0) {
        std::cerr
            << "batch-size and interventions "
            << "must be > 0" << std::endl;
        return -1;
    }
    return 0;
}

// Main
int main(int argc, char **argv)
{
    RunConfig cfg = parse_args(argc, argv);

    if (cfg.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (cfg.show_info) {
        display_all_device_info();
        return 0;
    }

    if (validate_config(&cfg) != 0) {
        return 1;
    }

    CLState st;
    memset(&st, 0, sizeof(st));

    if (cl_setup_init(&st, &cfg) != 0) {
        std::cerr << "OpenCL init failed."
                  << std::endl;
        return 1;
    }

    if (cl_setup_build_program(
            &st, "kernels") != 0) {
        std::cerr << "Kernel build failed."
                  << std::endl;
        cl_setup_cleanup(&st);
        return 1;
    }

    int rc = run_ablation_engine(&st, &cfg);

    cl_setup_cleanup(&st);
    return rc;
}
