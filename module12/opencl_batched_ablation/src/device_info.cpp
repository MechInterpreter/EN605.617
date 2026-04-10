// device_info.cpp -- Platform/device enumeration.

#include "device_info.h"
#include <iostream>
#include <cstring>

// Display platform string attribute
void display_platform_info(
    cl_platform_id id,
    cl_platform_info param,
    const char *label)
{
    size_t size = 0;
    clGetPlatformInfo(id, param, 0, NULL, &size);
    if (size == 0) return;

    char *buf = new char[size];
    clGetPlatformInfo(id, param, size, buf, NULL);
    std::cout << "    " << label << ": "
              << buf << std::endl;
    delete[] buf;
}

// Display device string attribute
void display_device_info_str(
    cl_device_id id,
    cl_device_info param,
    const char *label)
{
    size_t size = 0;
    clGetDeviceInfo(id, param, 0, NULL, &size);
    if (size == 0) return;

    char *buf = new char[size];
    clGetDeviceInfo(id, param, size, buf, NULL);
    std::cout << "      " << label << ": "
              << buf << std::endl;
    delete[] buf;
}

// Display cl_uint device attribute
void display_device_info_uint(
    cl_device_id id,
    cl_device_info param,
    const char *label)
{
    cl_uint val = 0;
    clGetDeviceInfo(
        id, param, sizeof(val), &val, NULL);
    std::cout << "      " << label << ": "
              << val << std::endl;
}

// Display cl_ulong device attribute
void display_device_info_ulong(
    cl_device_id id,
    cl_device_info param,
    const char *label)
{
    cl_ulong val = 0;
    clGetDeviceInfo(
        id, param, sizeof(val), &val, NULL);
    std::cout << "      " << label << ": "
              << val << std::endl;
}

// Print device details
static void print_device_details(cl_device_id dev)
{
    display_device_info_str(
        dev, CL_DEVICE_NAME, "Name");
    display_device_info_str(
        dev, CL_DEVICE_VENDOR, "Vendor");
    display_device_info_str(
        dev, CL_DEVICE_VERSION, "Version");
    display_device_info_str(
        dev, CL_DRIVER_VERSION, "Driver");
    display_device_info_uint(
        dev, CL_DEVICE_MAX_COMPUTE_UNITS, "CUs");
    display_device_info_uint(
        dev, CL_DEVICE_MAX_CLOCK_FREQUENCY,
        "Max Clock (MHz)");
    display_device_info_ulong(
        dev, CL_DEVICE_GLOBAL_MEM_SIZE,
        "Global Mem (bytes)");
    display_device_info_ulong(
        dev, CL_DEVICE_LOCAL_MEM_SIZE,
        "Local Mem (bytes)");
    display_device_info_uint(
        dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        "Mem Align (bits)");

    size_t wg = 0;
    clGetDeviceInfo(
        dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(wg), &wg, NULL);
    std::cout << "      Max WG Size: "
              << wg << std::endl;
}

// Display all platforms and devices
void display_all_device_info(void)
{
    cl_uint nplat = 0;
    clGetPlatformIDs(0, NULL, &nplat);
    if (nplat == 0) {
        std::cerr << "No OpenCL platforms."
                  << std::endl;
        return;
    }

    cl_platform_id *plats =
        new cl_platform_id[nplat];
    clGetPlatformIDs(nplat, plats, NULL);

    std::cout << "=== OpenCL Platforms: "
              << nplat << " ===" << std::endl;

    for (cl_uint p = 0; p < nplat; ++p) {
        std::cout << "  Platform " << p << ":"
                  << std::endl;
        display_platform_info(
            plats[p], CL_PLATFORM_NAME, "Name");
        display_platform_info(
            plats[p], CL_PLATFORM_VENDOR, "Vendor");
        display_platform_info(
            plats[p], CL_PLATFORM_VERSION,
            "Version");

        cl_uint ndev = 0;
        clGetDeviceIDs(
            plats[p], CL_DEVICE_TYPE_ALL,
            0, NULL, &ndev);
        if (ndev == 0) continue;

        cl_device_id *devs =
            new cl_device_id[ndev];
        clGetDeviceIDs(
            plats[p], CL_DEVICE_TYPE_ALL,
            ndev, devs, NULL);

        for (cl_uint d = 0; d < ndev; ++d) {
            std::cout << "    Device " << d
                      << ":" << std::endl;
            print_device_details(devs[d]);
        }
        delete[] devs;
    }
    delete[] plats;
    std::cout << "============================"
              << std::endl;
}
