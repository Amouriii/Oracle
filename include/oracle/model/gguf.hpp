#pragma once

// GGUF container reader: metadata, tensor directory, and the derived model facts
// Oracle needs in order to recognise a model and plan a shard for it.
//
// The file is mapped read-only and never copied.  A worker that only owns
// layers [a, b) touches only those tensors, so its resident set stays close to
// its own shard even though the whole file is mapped.

#include "oracle/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle::model {

// ggml_type, matching the on-disk numbering in ggml.h.
enum GgmlType : uint32_t {
  GGML_F32 = 0,
  GGML_F16 = 1,
  GGML_Q4_0 = 2,
  GGML_Q4_1 = 3,
  GGML_Q5_0 = 6,
  GGML_Q5_1 = 7,
  GGML_Q8_0 = 8,
  GGML_Q8_1 = 9,
  GGML_Q2_K = 10,
  GGML_Q3_K = 11,
  GGML_Q4_K = 12,
  GGML_Q5_K = 13,
  GGML_Q6_K = 14,
  GGML_Q8_K = 15,
  GGML_IQ2_XXS = 16,
  GGML_IQ2_XS = 17,
  GGML_IQ3_XXS = 18,
  GGML_IQ1_S = 19,
  GGML_IQ4_NL = 20,
  GGML_IQ3_S = 21,
  GGML_IQ2_S = 22,
  GGML_IQ4_XS = 23,
  GGML_I8 = 24,
  GGML_I16 = 25,
  GGML_I32 = 26,
  GGML_I64 = 27,
  GGML_F64 = 28,
  GGML_IQ1_M = 29,
  GGML_BF16 = 30,
  GGML_TYPE_COUNT = 39,
};

struct GgmlTypeInfo {
  const char* name{"unknown"};
  uint32_t block_size{0};  // elements per block
  uint32_t type_size{0};   // bytes per block
  bool dequantizable{false};
};

const GgmlTypeInfo& ggml_type_info(uint32_t type);
// Bytes occupied by `n_elements` contiguous elements of `type` (must be a
// multiple of the block size).
uint64_t ggml_row_bytes(uint32_t type, uint64_t n_elements);
double ggml_bits_per_weight(uint32_t type);

enum class GgufValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

struct GgufValue {
  GgufValueType type{GgufValueType::UINT32};
  // Scalars are widened; the reader keeps whichever field the type implies.
  int64_t i{0};
  double f{0.0};
  std::string s;

  // Arrays.  String arrays (vocab, merges) keep their own storage; numeric
  // arrays are widened to double, which is lossless for every GGUF numeric
  // type that appears in practice except full-range int64.
  GgufValueType array_type{GgufValueType::UINT32};
  std::vector<std::string> array_str;
  std::vector<double> array_num;

  [[nodiscard]] bool is_array() const noexcept { return type == GgufValueType::ARRAY; }
  [[nodiscard]] uint64_t as_u64() const noexcept { return static_cast<uint64_t>(i); }
  [[nodiscard]] double as_f64() const noexcept;
};

struct GgufTensorInfo {
  std::string name;
  uint32_t type{GGML_F32};
  std::vector<uint64_t> ne;  // ggml dim order: ne[0] is the fastest-moving axis
  uint64_t offset{0};        // relative to the start of the tensor data section
  uint64_t nbytes{0};
  uint64_t n_elements{0};

  [[nodiscard]] uint64_t rows() const noexcept;  // product of ne[1..]
  [[nodiscard]] uint64_t cols() const noexcept { return ne.empty() ? 0 : ne[0]; }
};

// A read-only mapping of one .gguf file.
class GgufFile {
 public:
  GgufFile() = default;
  ~GgufFile();
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  Status open(const std::string& path);
  void close();

  [[nodiscard]] bool is_open() const noexcept { return base_ != nullptr; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] uint32_t version() const noexcept { return version_; }
  [[nodiscard]] uint64_t file_size() const noexcept { return size_; }
  [[nodiscard]] uint64_t data_offset() const noexcept { return data_offset_; }
  [[nodiscard]] uint32_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] const std::unordered_map<std::string, GgufValue>& kv() const noexcept { return kv_; }
  [[nodiscard]] const std::vector<GgufTensorInfo>& tensors() const noexcept { return tensors_; }

  [[nodiscard]] const GgufValue* find_kv(const std::string& key) const;
  [[nodiscard]] const GgufTensorInfo* find_tensor(const std::string& name) const;
  // Base pointer of a tensor's payload inside the mapping, or nullptr if the
  // tensor's extent does not lie entirely within the file.
  [[nodiscard]] const uint8_t* tensor_data(const GgufTensorInfo& t) const;

  [[nodiscard]] std::string str(const std::string& key, const std::string& def = {}) const;
  [[nodiscard]] uint64_t u64(const std::string& key, uint64_t def = 0) const;
  [[nodiscard]] double f64(const std::string& key, double def = 0.0) const;
  [[nodiscard]] bool boolean(const std::string& key, bool def = false) const;
  // Looks up "<architecture>.<suffix>" and falls back to a bare "<suffix>".
  [[nodiscard]] std::string arch_key(const std::string& suffix) const;

 private:
  std::string path_;
  int fd_{-1};
  const uint8_t* base_{nullptr};
  uint64_t size_{0};
  uint32_t version_{0};
  uint32_t alignment_{32};
  uint64_t data_offset_{0};
  std::unordered_map<std::string, GgufValue> kv_;
  std::vector<GgufTensorInfo> tensors_;
  std::unordered_map<std::string, size_t> tensor_index_;
  std::string arch_;
};

// Everything Oracle reports about a model after recognising it.
struct ModelInfo {
  std::string path;
  std::string architecture;   // llama, qwen2, gemma, phi3, ...
  std::string name;           // general.name, else the architecture
  std::string organization;   // general.organization / basename
  std::string quantization;   // Q4_K_M, Q8_0, F16, "mixed", ...
  std::string dominant_type;  // ggml type carrying most of the weights
  std::string tokenizer_model;
  bool supported_for_inference{false};
  std::string unsupported_reason;

  uint64_t file_size_bytes{0};
  uint64_t tensor_count{0};
  uint64_t param_count{0};
  double bits_per_weight{0.0};

  uint32_t n_layers{0};
  uint32_t n_embd{0};
  uint32_t n_ff{0};
  uint32_t n_heads{0};
  uint32_t n_kv_heads{0};
  uint32_t head_dim{0};
  uint32_t n_vocab{0};
  uint32_t context_length{0};
  uint32_t rope_dim{0};
  float rope_freq_base{10000.0f};
  float rope_freq_scale{1.0f};
  float rms_eps{1e-5f};

  // Memory requirements, in bytes.
  uint64_t weight_bytes_total{0};
  uint64_t bytes_per_layer{0};      // mean over the block tensors
  uint64_t non_layer_bytes{0};      // embeddings, output norm, lm_head
  uint64_t kv_bytes_per_token{0};   // whole model, f16 K+V
  uint64_t activation_bytes{0};     // scratch for one token
  uint64_t recommended_ram_bytes{0};

  std::map<std::string, uint64_t> type_histogram;  // ggml type name -> tensor count
  std::map<std::string, uint64_t> type_bytes;      // ggml type name -> bytes

  [[nodiscard]] std::string param_count_human() const;
  [[nodiscard]] std::string to_json() const;
};

// Parse `path` and fill in `out`.  Works for any GGUF file; `supported_for_inference`
// says whether Oracle's own runner can execute it.
Status inspect_gguf(const std::string& path, ModelInfo* out);
Status inspect_gguf(const GgufFile& f, ModelInfo* out);

// Project a ModelInfo onto the cluster-level ModelMeta used for sharding.
void apply_to_model_meta(const ModelInfo& info, ModelMeta* meta);

// KV-cache bytes for a layer range, at the cluster's activation dtype.
uint64_t kv_bytes_for(const ModelInfo& info, uint32_t n_layers, uint32_t max_seq, DType dtype);

}  // namespace oracle::model
