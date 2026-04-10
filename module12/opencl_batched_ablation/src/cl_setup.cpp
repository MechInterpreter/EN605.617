// cl_setup.cpp -- OpenCL setup and program build.

#include "cl_setup.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

// Check OpenCL error; exit on failure
void cl_check(cl_int err, const char *op)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << op
                  << " (" << err << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Context error callback
static void CL_CALLBACK context_error_cb(
    const char *errinfo,
    const void * /*private_info*/,
    size_t       /*cb*/,
    void *       /*user_data*/)
{
    std::cerr << "[CL Context Error] "
              << errinfo << std::endl;
}

// Select platform by index
static cl_platform_id select_platform(int idx)
{
    cl_uint n = 0;
    cl_check(
        clGetPlatformIDs(0, NULL, &n),
        "clGetPlatformIDs(count)");
    if (n == 0 || idx < 0 || (cl_uint)idx >= n) {
        std::cerr << "Invalid platform index "
                  << idx << " (have " << n << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    cl_platform_id *ps = new cl_platform_id[n];
    clGetPlatformIDs(n, ps, NULL);
    cl_platform_id p = ps[idx];
    delete[] ps;
    return p;
}

// Select device by index
static cl_device_id select_device(
    cl_platform_id plat, int idx)
{
    cl_uint n = 0;
    cl_check(
        clGetDeviceIDs(
            plat, CL_DEVICE_TYPE_ALL,
            0, NULL, &n),
        "clGetDeviceIDs(count)");
    if (n == 0 || idx < 0 || (cl_uint)idx >= n) {
        std::cerr << "Invalid device index "
                  << idx << " (have " << n << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    cl_device_id *ds = new cl_device_id[n];
    clGetDeviceIDs(
        plat, CL_DEVICE_TYPE_ALL, n, ds, NULL);
    cl_device_id d = ds[idx];
    delete[] ds;
    return d;
}

// Read file contents
static std::string read_file(const char *path)
{
    std::ifstream f(path, std::ios::in);
    if (!f.is_open()) {
        std::cerr << "Cannot open: " << path
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Initialize OpenCL state
int cl_setup_init(
    CLState *st, const RunConfig *cfg)
{
    st->platform = select_platform(
        cfg->platform_idx);
    st->device   = select_device(
        st->platform, cfg->device_idx);

    cl_int err;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)st->platform,
        0
    };
    st->context = clCreateContext(
        props, 1, &st->device,
        context_error_cb, NULL, &err);
    cl_check(err, "clCreateContext");

    cl_command_queue_properties qprops =
        CL_QUEUE_PROFILING_ENABLE;
    st->queue = clCreateCommandQueue(
        st->context, st->device, qprops, &err);
    cl_check(err, "clCreateCommandQueue");

    st->program = NULL;
    return 0;
}

// Show build log on failure
static void show_build_log(
    cl_program prog, cl_device_id dev)
{
    size_t len = 0;
    clGetProgramBuildInfo(
        prog, dev, CL_PROGRAM_BUILD_LOG,
        0, NULL, &len);
    char *log = new char[len + 1];
    clGetProgramBuildInfo(
        prog, dev, CL_PROGRAM_BUILD_LOG,
        len, log, NULL);
    log[len] = '\0';
    std::cerr << "Build log:\n" << log << std::endl;
    delete[] log;
}

// Build program from kernel files
int cl_setup_build_program(
    CLState *st, const char *kernel_dir)
{
    const char *files[] = {
        "forward_pass.cl",
        "ablation.cl",
        "scoring.cl"
    };
    const int nfiles = 3;

    std::string combined;
    for (int i = 0; i < nfiles; ++i) {
        std::string path =
            std::string(kernel_dir) + "/" + files[i];
        combined += "// === " + std::string(files[i])
                  + " ===\n";
        combined += read_file(path.c_str());
        combined += "\n";
    }

    const char *src = combined.c_str();
    size_t len = combined.length();

    cl_int err;
    st->program = clCreateProgramWithSource(
        st->context, 1, &src, &len, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(
        st->program, 1, &st->device,
        "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel build failed!"
                  << std::endl;
        show_build_log(st->program, st->device);
        return -1;
    }
    return 0;
}

// Cleanup
void cl_setup_cleanup(CLState *st)
{
    if (st->program)
        clReleaseProgram(st->program);
    if (st->queue)
        clReleaseCommandQueue(st->queue);
    if (st->context)
        clReleaseContext(st->context);
}
