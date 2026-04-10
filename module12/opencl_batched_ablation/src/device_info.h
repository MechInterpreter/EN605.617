// device_info.h -- Platform/device enumeration.

#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Display all platforms and devices
void display_all_device_info(void);

// Display platform string attribute
void display_platform_info(
    cl_platform_id id,
    cl_platform_info param,
    const char *label);

// Display device string attribute
void display_device_info_str(
    cl_device_id id,
    cl_device_info param,
    const char *label);

// Display cl_uint device attribute
void display_device_info_uint(
    cl_device_id id,
    cl_device_info param,
    const char *label);

// Display cl_ulong device attribute
void display_device_info_ulong(
    cl_device_id id,
    cl_device_info param,
    const char *label);

#endif // DEVICE_INFO_H
