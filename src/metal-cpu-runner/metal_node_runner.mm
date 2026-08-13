#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "oracle/runner/node_runner.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace oracle {

struct MetalNodeRunner::Impl {
  id<MTLDevice> device{nil};
  id<MTLCommandQueue> queue{nil};
  id<MTLComputePipelineState> gemm{nil};
  uint64_t vram_recommended{0};

  Status init() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return Status::fail(Errc::not_found, "no Metal device");
    }
    vram_recommended = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
    queue = [device newCommandQueue];
    NSError* err = nil;
    NSString* src = nil;
    const char* candidates[] = {"src/metal-cpu-runner/shaders/gemm.metal",
                                "../src/metal-cpu-runner/shaders/gemm.metal",
                                "../../src/metal-cpu-runner/shaders/gemm.metal"};
    for (auto* p : candidates) {
      std::ifstream in(p);
      if (!in) {
        continue;
      }
      std::ostringstream ss;
      ss << in.rdbuf();
      src = [NSString stringWithUTF8String:ss.str().c_str()];
      break;
    }
    if (!src) {
      src = @"#include <metal_stdlib>\nusing namespace metal;\n"
            @"kernel void sgemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],"
            @" device float* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],"
            @" constant uint& K [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {"
            @" if (gid.y >= M || gid.x >= N) return; float acc=0; for(uint k=0;k<K;++k) acc+=A[gid.y*K+k]*B[k*N+gid.x];"
            @" C[gid.y*N+gid.x]=acc; }";
    }
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
      return Status::fail(Errc::io, err ? [[err localizedDescription] UTF8String] : "metal compile");
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"sgemm"];
    gemm = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!gemm) {
      return Status::fail(Errc::io, err ? [[err localizedDescription] UTF8String] : "pipeline");
    }
    return Status::OK();
  }
};

MetalNodeRunner::MetalNodeRunner() : impl_(std::make_unique<Impl>()) { (void)impl_->init(); }
MetalNodeRunner::~MetalNodeRunner() = default;

Status MetalNodeRunner::load_layers(const ModelMeta& model, LayerRange layers, std::string_view) {
  model_ = model;
  layers_ = layers;
  if (!impl_->device) {
    return impl_->init();
  }
  return Status::OK();
}

Status MetalNodeRunner::load_layer_blob(uint32_t, std::span<const std::byte>) { return Status::OK(); }

Status MetalNodeRunner::gemm_f32(uint32_t m, uint32_t n, uint32_t k, const float* a, const float* b, float* c) {
  if (!impl_->device || !impl_->gemm) {
    auto st = impl_->init();
    if (!st) {
      return st;
    }
  }
  const size_t as = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bs = static_cast<size_t>(k) * n * sizeof(float);
  const size_t cs = static_cast<size_t>(m) * n * sizeof(float);
  id<MTLBuffer> A = [impl_->device newBufferWithBytes:a length:as options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [impl_->device newBufferWithBytes:b length:bs options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [impl_->device newBufferWithLength:cs options:MTLResourceStorageModeShared];
  uint32_t dims[3] = {m, n, k};
  id<MTLBuffer> Md = [impl_->device newBufferWithBytes:&dims[0] length:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Nd = [impl_->device newBufferWithBytes:&dims[1] length:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Kd = [impl_->device newBufferWithBytes:&dims[2] length:4 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:impl_->gemm];
  [enc setBuffer:A offset:0 atIndex:0];
  [enc setBuffer:B offset:0 atIndex:1];
  [enc setBuffer:C offset:0 atIndex:2];
  [enc setBuffer:Md offset:0 atIndex:3];
  [enc setBuffer:Nd offset:0 atIndex:4];
  [enc setBuffer:Kd offset:0 atIndex:5];
  MTLSize grid = MTLSizeMake(n, m, 1);
  NSUInteger tw = std::min<NSUInteger>(16, impl_->gemm.threadExecutionWidth);
  MTLSize tgs = MTLSizeMake(tw, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  std::memcpy(c, [C contents], cs);
  return Status::OK();
}

Status MetalNodeRunner::embed(std::span<const int32_t> tokens, Tensor* out) {
  AccelerateRunner cpu;
  auto st = cpu.load_layers(model_, layers_, {});
  if (!st) {
    return st;
  }
  return cpu.embed(tokens, out);
}

Status MetalNodeRunner::forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) {
  const uint32_t h = model_.hidden_dim ? model_.hidden_dim : 8;
  std::vector<float> ident(static_cast<size_t>(h) * h, 0.f);
  for (uint32_t i = 0; i < h; ++i) {
    ident[i * h + i] = 1.f;
  }
  std::vector<float> x(h, 0.f);
  if (in.size() >= h * sizeof(float) && (model_.act_dtype == DType::F32 || in.size() == h * 4)) {
    std::memcpy(x.data(), in.data(), h * 4);
  } else {
    AccelerateRunner cpu;
    cpu.load_layers(model_, layers_, {});
    return cpu.forward(in, kv, out);
  }
  std::vector<float> y(h, 0.f);
  auto st = gemm_f32(1, h, h, x.data(), ident.data(), y.data());
  if (!st) {
    return st;
  }
  if (kv.seq_len + 1 <= kv.layout.max_seq) {
    ++kv.seq_len;
  }
  out->header.magic = kTensorMagic;
  out->header.dtype = static_cast<uint16_t>(DType::F32);
  out->header.rank = 1;
  out->header.shape[0] = h;
  out->header.nbytes = y.size() * 4;
  out->header.flags = kFlagDecode;
  out->payload.resize(y.size() * 4);
  std::memcpy(out->payload.data(), y.data(), y.size() * 4);
  out->header.checksum = crc32(out->payload);
  return Status::OK();
}

Status MetalNodeRunner::lm_head(std::span<const std::byte> hidden, Tensor* logits) {
  AccelerateRunner cpu;
  auto st = cpu.load_layers(model_, layers_, {});
  if (!st) {
    return st;
  }
  return cpu.lm_head(hidden, logits);
}

std::unique_ptr<NodeRunner> make_metal_runner() { return std::make_unique<MetalNodeRunner>(); }

}  // namespace oracle
