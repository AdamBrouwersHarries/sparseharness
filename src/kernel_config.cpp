#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// #include "file_utils.h"

#include "kernel_config.h"

template <typename T> KernelConfig<T>::KernelConfig(std::string filename) {
  start_timer(KernelConfig, KernelConfig);
  boost::property_tree::ptree tree;

  boost::property_tree::read_json(filename, tree);

  name = tree.get<std::string>("name");
  source = tree.get<std::string>("source");

  kprops = KernelProperties(name);

  // iterate over the kernel properties
  auto properties = tree.get_child("properties");
  auto outerMap = properties.get_optional<std::string>("outerMap");
  auto innerMap = properties.get_optional<std::string>("innerMap");
  auto innerMap2 = properties.get_optional<std::string>("innerMap2");
  auto splitSize = properties.get_optional<std::string>("splitSize");
  auto chunkSize = properties.get_optional<std::string>("chunkSize");
  auto arrayType = properties.get_optional<std::string>("arrayType");

  auto unwrap_map = [](boost::optional<std::string> value) {
    return value ? value.get() : std::string("nothing");
  };

  auto unwrap_param = [](boost::optional<std::string> value) {
    return value ? std::stoi(value.get()) : 1;
  };

  kprops = KernelProperties(unwrap_map(outerMap), unwrap_map(innerMap),
                            unwrap_map(innerMap2), unwrap_map(arrayType),
                            unwrap_param(splitSize), unwrap_param(chunkSize));

  std::cout << "Kernel: " << name << ", source: \n" << source << ENDL;

  // for (auto &arg : tree.get_child("args")) {
  //   std::string variable = arg.second.get<std::string>("variable");
  //   std::string addressSpace = arg.second.get<std::string>("addressSpace");
  //   std::string size = arg.second.get<std::string>("size");
  //   args.push_back(ArgDescr(variable, addressSpace, size));

  //   std::cout << "variable: " << variable << " address space: " <<
  //   addressSpace
  //             << " size: " << size << ENDL;
  // }
  std::cerr << "input arguments: " << ENDL;
  for (auto &arg : tree.get_child("inputArgs")) {
    std::string variable = arg.second.get<std::string>("variable");
    std::string addressSpace = arg.second.get<std::string>("addressSpace");
    std::string size = arg.second.get<std::string>("size");
    inputArgs.push_back(ArgDescr(variable, addressSpace, size));
    std::cout << "variable: " << variable << " address space: " << addressSpace
              << " size: " << size << ENDL;
  }
  std::cerr << "output argument: " << ENDL;
  {
    auto &outputArgJson = tree.get_child("outputArg");
    std::string variable = outputArgJson.get<std::string>("variable");
    std::string addressSpace = outputArgJson.get<std::string>("addressSpace");
    std::string size = outputArgJson.get<std::string>("size");
    outputArg = new ArgDescr(variable, addressSpace, size);
    std::cout << "variable: " << variable << " address space: " << addressSpace
              << " size: " << size << ENDL;
  }
  std::cerr << "temp global arguments: " << ENDL;
  for (auto &arg : tree.get_child("tempGlobals")) {
    std::string variable = arg.second.get<std::string>("variable");
    std::string addressSpace = arg.second.get<std::string>("addressSpace");
    std::string size = arg.second.get<std::string>("size");
    tempGlobals.push_back(ArgDescr(variable, addressSpace, size));
    std::cout << "variable: " << variable << " address space: " << addressSpace
              << " size: " << size << ENDL;
  }
  std::cerr << "temp local arguments: " << ENDL;
  for (auto &arg : tree.get_child("tempLocals")) {
    std::string variable = arg.second.get<std::string>("variable");
    std::string addressSpace = arg.second.get<std::string>("addressSpace");
    std::string size = arg.second.get<std::string>("size");
    tempLocals.push_back(ArgDescr(variable, addressSpace, size));
    std::cout << "variable: " << variable << " address space: " << addressSpace
              << " size: " << size << ENDL;
  }
}

template <typename T> std::string &KernelConfig<T>::getSource() {
  return source;
}

template <typename T> std::string &KernelConfig<T>::getName() { return name; }

template <typename T> std::vector<ArgDescr> KernelConfig<T>::getArgs() {
  return inputArgs;
}

template <typename T> std::vector<ArgDescr> KernelConfig<T>::getTempGlobals() {
  return tempGlobals;
}

template <typename T> std::vector<ArgDescr> KernelConfig<T>::getTempLocals() {
  return tempLocals;
}

template <typename T> ArgDescr *KernelConfig<T>::getOutputArg() {
  return outputArg;
}

template <typename T> KernelProperties KernelConfig<T>::getProperties() {
  return kprops;
}

// from
// https://stackoverflow.com/questions/38874605/generic-method-for-flattening-2d-vectors
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig) {
  std::vector<T> ret;
  for (const auto &v : orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

// KernelProperties

KernelProperties::KernelProperties() {
  // dummy initialiser
}

KernelProperties::KernelProperties(std::string kname) : argcache(kname) {
  // do nothing for now
}

KernelProperties::KernelProperties(std::string om, std::string im,
                                   std::string im2, std::string at, int ss,
                                   int cs)
    : outerMap(om), innerMap(im), innerMap2(im2), arrayType(at), splitSize(ss),
      chunkSize(cs) {
  // do nothing else for now
}

template class KernelConfig<float>;
template class KernelConfig<int>;
template class KernelConfig<bool>;
template class KernelConfig<double>;
