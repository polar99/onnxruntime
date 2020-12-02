// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/session/tensorhelper.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::experimental;
using namespace onnxruntime::common;

namespace onnxruntime {

size_t findIndex(const std::vector<int64_t>& dims, const std::vector<int64_t>& indices) {
  size_t linear_index = 0;
  for (unsigned int i = 0; i < dims.size(); i++) {
    ORT_ENFORCE(indices[i] >= 0);
    ORT_ENFORCE(indices[i] < dims[i]);
    ORT_ENFORCE(dims[i] > 1);
    linear_index = indices[i] + dims[i] * linear_index;
  }
  return linear_index;
}

std::vector<int64_t> GetSliceShape(
    const std::vector<int64_t>& shape,
    const size_t& slice_axis,
    const size_t& num_slices,
    size_t& contiguous_slice_size,
    size_t& total_num_elements) {
  ORT_ENFORCE(shape.size() > 0);
  ORT_ENFORCE(slice_axis < shape.size());
  ORT_ENFORCE(num_slices > 0);

  // Shape of slice along slice_axis.
  std::vector<int64_t> slice_shape;
  contiguous_slice_size = 1;
  total_num_elements = 1;

  for (size_t i_shape = 0; i_shape < shape.size(); ++i_shape) {
    const auto d = shape[i_shape];

    // Compute slice's shape.
    if (i_shape == slice_axis) {
      slice_shape.push_back(d / num_slices);
    } else {
      slice_shape.push_back(d);
    }

    // Compute the largest contigiguous memory block
    // copied from the original tensor.
    if (i_shape > slice_axis) {
      contiguous_slice_size *= d;
    }

    // Also record the number of total elements in the original tensor.
    // It will be used as boundary when copying slice content from the
    // original tensor.
    total_num_elements *= d;
  }

  return slice_shape;
}

OrtValue CreateCpuTensorValue(const MLDataType elem_type, std::vector<int64_t> shape, TrainingSession& session_state) {
  ORT_ENFORCE(elem_type->AsPrimitiveDataType(), "Tensor's element type must be a scalar type.");
  ORT_ENFORCE(shape.size() > 0, "Shape vector must be non-empty.");

  // Get CPU allocator from the session.
  OrtMemoryInfo cpu_location(onnxruntime::CPU, OrtDeviceAllocator);
  AllocatorPtr cpu_allocator = session_state.GetAllocator(cpu_location);

  // Given a shape, allocate a tensor using CPU allocator.
  auto cpu_tensor = onnxruntime::make_unique<Tensor>(elem_type, shape, cpu_allocator);

  auto tensor_type = DataTypeImpl::GetType<Tensor>();
  // Create OrtValue to wrap the allocated tensor.
  OrtValue cpu_value{cpu_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()};

  return cpu_value;
}

void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t& size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) {
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

  // Current CUDA device.
  int device;
  CUDA_CALL(cudaGetDevice(&device));

  if (device != src_location.id) {
    // Need to switch to the allocating device.
    CUDA_CALL(cudaSetDevice(src_location.id));
    // Copy from GPU to CPU.
    CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    // Switch back to current device.
    CUDA_CALL(cudaSetDevice(device));
  } else {
    // Copy from GPU to CPU.
    CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
  }
}

void CopyCpuToCpu(void* dst_ptr, const void* src_ptr, const size_t& size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) {
  ORT_ENFORCE(src_location.device.Type() == OrtDevice::CPU);
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);
  memcpy(dst_ptr, src_ptr, size);
}

// Copy a tensor allocated on GPU/CPU to CPU buffer.
void CopyToCpuTensor(Tensor& dst, const Tensor& src) {
  // Get the tensor shape.
  const TensorShape& dst_shape = dst.Shape();
  const TensorShape& src_shape = src.Shape();

  ORT_ENFORCE(dst_shape == src_shape);

  // Get tensor's element type.
  const MLDataType src_type = src.DataType();
  const MLDataType dst_type = dst.DataType();

  ORT_ENFORCE(dst_type == src_type);

  // Get tensor's memory location.
  const OrtMemoryInfo& dst_location = dst.Location();
  const OrtMemoryInfo& src_location = src.Location();

  ORT_ENFORCE(src_location.device.Type() == OrtDevice::CPU || src_location.device.Type() == OrtDevice::GPU,
              "The copy function can only copy source tensor from CPU/GPU to CPU.");
  ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU, "Destination tensor must be a CPU tensor.");

  auto src_ptr = src.DataRaw();
  auto dst_ptr = dst.MutableDataRaw();

  if (src_location.device.Type() == OrtDevice::GPU) {
    CopyGpuToCpu(dst_ptr, src_ptr, src.SizeInBytes(), dst_location, src_location);
  } else {
    CopyCpuToCpu(dst_ptr, src_ptr, src.SizeInBytes(), dst_location, src_location);
  }
}

// Copy a slice from source tensor to destination tensor.
// Assume that input shape is [10, 8, 2], slice_axis=1, num_slices=4.
// The destination's tensor is computed using
//  dst = src[:, lower:upper, :],
// where 
//  slice_stride = 8 / num_slices,
//  lower = slice_id * slice_stride,
//  upper = (slice_id + 1) * slice_stride.
void CopySlice(Tensor& dst, const Tensor& src, const size_t slice_id, const size_t slice_axis, const size_t num_slices) {
  ORT_ENFORCE(dst.Location().device.Type() == OrtDevice::CPU, "Source must be a CPU tensor.");
  ORT_ENFORCE(src.Location().device.Type() == OrtDevice::CPU, "Destination must be a CPU tensor.");

  const TensorShape& src_shape = src.Shape();

  ORT_ENFORCE(src_shape[slice_axis] % num_slices == 0,
              "The dimension along the sliced axis must be divisible by the number of slices.",
              " We have sliced dimension ", src_shape[slice_axis], " and number slices ", num_slices, ".");

  // Cache sliced dimension.
  const size_t slice_dim = src_shape[slice_axis];
  // The number of slices along the sliced axis.
  const size_t slice_size = src_shape[slice_axis] / num_slices;

  auto src_ptr = src.DataRaw();
  auto dst_ptr = dst.MutableDataRaw();

  // If we slice tensor with shape [D1, D2, ..., Dj, sliced_dim, Dk, ..., Dn], then segment_size is slice_size * Dk * ... * Dn.
  size_t segment_size = 1;
  // The total number of combinations of (D1, D2, ..., Dj). It's used as the total count of segments.
  size_t num_segments = 1;

  for (size_t i = 0; i < static_cast<size_t>(src_shape.NumDimensions()); ++i) {
    if (i > slice_axis) {
      segment_size *= src_shape[i];
    } else if (i == slice_axis) {
      segment_size *= slice_size;
    }
    if (i < slice_axis) {
      num_segments *= src_shape[i];
    }
  }

  // Update the linear index as if one extra axis is appended to the original shape.
  // For a tensor with shape [D1, D2, D3], the linear index of element at (x, y, z) is
  // i = x * (D2 * D3) + y * D3 + z. If we append one dimension to form a new shape [D1, D2, D3, D4],
  // the new linear index at (x, y, z, u) can be computed using i * D4 + u.
  auto update_linear_index = [](size_t linear_index, size_t new_axis_index, size_t new_axis_dim) {
    return linear_index * new_axis_dim + new_axis_index;
  };

  // For each segment, we have several consecutive memory blocks to copy. For example, the first segment is
  // input[0, ..., 0, slice_id*slice_size : (slice_id + 1) * slice_size, :, ..., :], where its memory blocks
  // are
  //   input[0, ..., 0, 0, :, ..., :],
  //   input[0, ..., 0, 1, :, ..., :],
  //   ...
  //   input[0, ..., slize_size - 1, 1, :, ..., :].
  for (size_t i = 0; i < num_segments; ++i) {
    // Do pointer arithmetic operations using "char*" because things are stored in terms of bytes.
    // Copy input[i, slice_id*slice_size : (slice_id + 1) * slice_size, :, ..., :] to buffer.
    const void* src_addr = reinterpret_cast<const char*>(src_ptr) + update_linear_index(i, slice_id * slice_size, slice_dim) * segment_size * src.DataType()->Size();
    void* dst_addr = reinterpret_cast<char*>(dst_ptr) + update_linear_index(i, 0 * slice_size, slice_dim) * segment_size * dst.DataType()->Size();
    memcpy(dst_addr, src_addr, segment_size * slice_size * src.DataType()->Size());
  }
}

OrtValue SliceTensor1(const OrtValue& value, const size_t slice_id,
                      const size_t slice_axis, const size_t num_slices, TrainingSession& session_state) {
  ORT_ENFORCE(value.IsTensor(), "Sliced value must be a tensor.");
  auto& src = value.Get<Tensor>();
  auto src_shape = src.Shape().GetDims();

  auto buf_value = CreateCpuTensorValue(src.DataType(), src_shape, session_state);
  ORT_ENFORCE(buf_value.IsTensor(), "Buffer value must be a tensor.");
  auto& buf = *buf_value.GetMutable<Tensor>();
  CopyToCpuTensor(buf, src);

  // TODO: remove.
  size_t contiguous_slice_size, total_num_elements;

  // Create the output, a slice of the original tensor.
  auto slice_shape = GetSliceShape(src_shape, slice_axis, num_slices, contiguous_slice_size, total_num_elements);
  auto dst_value = CreateCpuTensorValue(src.DataType(), slice_shape, session_state);
  ORT_ENFORCE(dst_value.IsTensor(), "Buffer value must be a tensor.");
  auto& dst = *dst_value.GetMutable<Tensor>();

  // Copy the content of slice from the original tensor.
  CopySlice(dst, buf, slice_id, slice_axis, num_slices);

  return dst_value;
}

OrtValue SliceTensor(const OrtValue& orig_value, const size_t slice_id,
                     const size_t slice_axis, const size_t num_slices, TrainingSession& session_state) {
  // Get tensor from OrtValue
  const Tensor& orig_tensor = orig_value.Get<Tensor>();

  // Get the tensor shape
  const TensorShape& orig_tensor_shape = orig_tensor.Shape();
  const MLDataType orig_tensor_type = orig_tensor.DataType();
  const OrtMemoryInfo orig_tensor_location = orig_tensor.Location();

  // Get the shape of the tensor to slice.
  const std::vector<int64_t>& orig_dims = orig_tensor_shape.GetDims();
  ORT_ENFORCE(orig_dims[slice_axis] % num_slices == 0,
              "The number of slices, \"num_slices\", does not evenly divide the dimension along the sliced axis \"slice_axis\".",
              " num_slices=", num_slices, ", slice_axis=", orig_dims[slice_axis]);

  // Declare a vector to store the shape of the tensor slice.
  size_t contiguous_slice_size, total_num_elements;
  auto small_dims = GetSliceShape(orig_dims, slice_axis, num_slices, contiguous_slice_size, total_num_elements);
  TensorShape small_shape(small_dims);

  // Use the calculated slice shape to allocate a tensor.
  // Content will be copied from original tensor later.
  OrtValue small_cpu_value = CreateCpuTensorValue(orig_tensor_type, small_dims, session_state);

  auto cpu_ptr = orig_value.Get<Tensor>().DataRaw();
  auto small_cpu_ptr = small_cpu_value.GetMutable<Tensor>()->MutableDataRaw();

  int device;
  cudaGetDevice(&device);
  size_t elements_read_so_far = 0;
  size_t bias = 0;
  size_t num_strides = 0;
  size_t slice_size = orig_dims[slice_axis] / num_slices;  //assuming that dims[slice_axis] is perfectly divisible else throw error
  size_t copied_size = orig_tensor_type->Size() * contiguous_slice_size * slice_size;
  while (elements_read_so_far <= total_num_elements) {
    bias = (elements_read_so_far + slice_id * slice_size * contiguous_slice_size) * orig_tensor_type->Size();
    if (std::string(orig_tensor_location.name) == std::string("Cuda")) {
      if (device != orig_tensor_location.id) {
        cudaSetDevice(orig_tensor_location.id);
      }

      cudaMemcpy(static_cast<char*>((void*)small_cpu_ptr) + num_strides * copied_size, static_cast<const char*>(cpu_ptr) + bias, copied_size, cudaMemcpyDeviceToHost);

      if (device != orig_tensor_location.id) {
        cudaSetDevice(device);
      }
    } else {
      memcpy(static_cast<char*>((void*)small_cpu_ptr) + num_strides * copied_size, static_cast<const char*>(cpu_ptr) + bias, copied_size);
    }
    elements_read_so_far += orig_dims[slice_axis] * contiguous_slice_size;

    num_strides += 1;
  }

  copied_size = num_strides * copied_size;
  if (std::string(orig_tensor_location.name) == std::string("Cuda")) {
    // Get CPU tensor to be copied.
    const Tensor& copied_tensor = small_cpu_value.Get<Tensor>();

    // Create GPU tensor to capture CPU data.
    AllocatorPtr allocator = session_state.GetAllocator(orig_tensor_location);
    auto small_tensor = onnxruntime::make_unique<Tensor>(orig_tensor_type, small_shape, allocator);
    auto tensor_type = DataTypeImpl::GetType<Tensor>();
    OrtValue small_value{small_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()};
    Tensor* capturing_tensor = small_value.GetMutable<Tensor>();

    if (device != orig_tensor_location.id) {
      cudaSetDevice(orig_tensor_location.id);
    }

    cudaMemcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size, cudaMemcpyHostToDevice);

    if (device != orig_tensor_location.id) {
      cudaSetDevice(device);
    }
    return small_value;
  } else if (std::string(orig_tensor_location.name) == std::string("Cpu")) {
    const Tensor& copied_tensor = small_cpu_value.Get<Tensor>();

    AllocatorPtr allocator = session_state.GetAllocator(orig_tensor_location);
    auto small_tensor = onnxruntime::make_unique<Tensor>(orig_tensor_type, small_shape, allocator);
    auto tensor_type = DataTypeImpl::GetType<Tensor>();
    OrtValue small_value{small_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()};
    Tensor* capturing_tensor = small_value.GetMutable<Tensor>();

    memcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size);
    return small_value;
  } else {
    ORT_ENFORCE(false, "This shouldn't happen.");
  }

}  // slice func ends
}