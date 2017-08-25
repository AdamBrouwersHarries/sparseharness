#include "GlobalArg.h"

#include "util/Logger.h"

namespace executor {

GlobalArg::GlobalArg(executor::Vector<char> &&vectorP, bool isOutputP)
    : vector(std::move(vectorP)), isOutput(isOutputP) {}

KernelArg *GlobalArg::create(void *data, size_t size, bool isOutput) {
  auto dataCharPtr = static_cast<char *>(data);
  executor::Vector<char> vector(dataCharPtr, dataCharPtr + size);
  return new GlobalArg{std::move(vector), isOutput};
}

KernelArg *GlobalArg::create(size_t size, bool isOutput) {
  executor::Vector<char> vector;
  vector.resize(size);
  return new GlobalArg{std::move(vector), isOutput};
}

void GlobalArg::assign(void *data, size_t size) {
  auto dataCharPtr = static_cast<char *>(data);
  vector.assign(dataCharPtr, dataCharPtr + size);
}

void GlobalArg::assign(std::vector<char> &data) {
  vector.dataOnDeviceModified();
  vector.assign(data.begin(), data.end());
}

executor::Vector<char> &GlobalArg::data() { return vector; }

void GlobalArg::clear() {
  if (isOutput) {
    vector.assign(vector.size());
    vector.dataOnHostModified();
  }
}

void GlobalArg::setAsKernelArg(cl::Kernel kernel, int i) {
  auto &devPtr = executor::globalDeviceList.front();

        size_t size;
        clGetKernelArgInfo(kernel(), i,  CL_KERNEL_ARG_NAME, 0, NULL, &size);
        char * name = (char*)malloc(size);
        clGetKernelArgInfo(kernel(), i,  CL_KERNEL_ARG_NAME, size, name, 0);
        std::cout << "setKernelArg for " << i << ": " << name << " buffer: " << 
                 vector.deviceBuffer(*devPtr).clBuffer()() << "\n";
        free(name);

  LOG_DEBUG_INFO("Setting GlobalArg with size ", vector.size(),
                 ", at position ", i, " and buffer ",
                 vector.deviceBuffer(*devPtr).clBuffer()());

  auto err = kernel.setArg(i, vector.deviceBuffer(*devPtr).clBuffer());
  if (err != CL_SUCCESS) {
    LOG_ERROR("ERROR SET KERNEL ARG: ", err);
  }
}

void GlobalArg::upload() {
  // create buffers on device
  vector.createDeviceBuffers();
  // start upload
  vector.startUpload();
}

void GlobalArg::download() {
  //if (isOutput) {
    vector.dataOnDeviceModified();
    vector.copyDataToHost();
  //}
}
}
