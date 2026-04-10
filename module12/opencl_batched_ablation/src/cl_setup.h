// cl_setup.h -- OpenCL context, queue, and
// program setup.

#ifndef CL_SETUP_H
#define CL_SETUP_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../include/types.h"

// OpenCL runtime state
typedef struct {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue  queue;
    cl_program       program;
} CLState;

// Check OpenCL error; exit on failure
void cl_check(cl_int err, const char *op);

// Initialize OpenCL (profiling-enabled queue)
int cl_setup_init(
    CLState *st,
    const RunConfig *cfg);

// Build program from .cl files in kernel_dir
int cl_setup_build_program(
    CLState *st,
    const char *kernel_dir);

// Release OpenCL resources
void cl_setup_cleanup(CLState *st);

#endif // CL_SETUP_H
