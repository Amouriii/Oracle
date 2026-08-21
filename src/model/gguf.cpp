#include "oracle/model/gguf.hpp"

#include "oracle/model/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace oracle::model {
namespace {

constexpr uint32_t kGgufMagic = 0x46554747u;  // "GGUF"
constexpr int QK_K = 256;

struct TypeRow {
  uint32_t type;
  GgmlTypeInfo info;
};

// block_size / type_size mirror ggml's type traits table.  Types Oracle cannot
// expand yet are still listed so that metadata, sizing and the shard planner
// work for any GGUF file, even one it declines to execute.
const TypeRow kTypes[] = {
    {GGML_F32, {"F32", 1, 4, true}},
    {GGML_F16, {"F16", 1, 2, true}},
    {GGML_Q4_0, {"Q4_0", 32, 18, true}},
    {GGML_Q4_1, {"Q4_1", 32, 20, true}},
    {GGML_Q5_0, {"Q5_0", 32, 22, true}},
    {GGML_Q5_1, {"Q5_1", 32, 24, true}},
    {GGML_Q8_0, {"Q8_0", 32, 34, true}},
    {GGML_Q8_1, {"Q8_1", 32, 40, false}},
    {GGML_Q2_K, {"Q2_K", QK_K, 84, true}},
    {GGML_Q3_K, {"Q3_K", QK_K, 110, true}},
    {GGML_Q4_K, {"Q4_K", QK_K, 144, true}},
    {GGML_Q5_K, {"Q5_K", QK_K, 176, true}},
    {GGML_Q6_K, {"Q6_K", QK_K, 210, true}},
    {GGML_Q8_K, {"Q8_K", QK_K, 4 + QK_K + QK_K / 16 * 2, true}},
    {GGML_IQ2_XXS, {"IQ2_XXS", QK_K, 2 + QK_K / 4, false}},
    {GGML_IQ2_XS, {"IQ2_XS", QK_K, 2 + QK_K / 4 + QK_K / 32, false}},
    {GGML_IQ3_XXS, {"IQ3_XXS", QK_K, 2 + QK_K / 4 + QK_K / 8, false}},
    {GGML_IQ1_S, {"IQ1_S", QK_K, 2 + QK_K / 8 + QK_K / 16, false}},
    {GGML_IQ4_NL, {"IQ4_NL", 32, 2 + 16, false}},
    {GGML_IQ3_S, {"IQ3_S", QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4, false}},
    {GGML_IQ2_S, {"IQ2_S", QK_K, 2 + QK_K / 4 + QK_K / 16, false}},
    {GGML_IQ4_XS, {"IQ4_XS", QK_K, 2 + 2 + QK_K / 64 + QK_K / 2, false}},
    {GGML_I8, {"I8", 1, 1, false}},
    {GGML_I16, {"I16", 1, 2, false}},
    {GGML_I32, {"I32", 1, 4, false}},
    {GGML_I64, {"I64", 1, 8, false}},
    {GGML_F64, {"F64", 1, 8, false}},
    {GGML_IQ1_M, {"IQ1_M", QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32, false}},
    {GGML_BF16, {"BF16", 1, 2, true}},
};

const GgmlTypeInfo kUnknownType{"UNKNOWN", 0, 0, false};

// A bounds-checked cursor over the mapped file.  Every read validates against
// the remaining length so a truncated or hostile GGUF cannot walk off the map.
class Cursor {
 public:
  Cursor(const uint8_t* base, uint64_t size) : base_(base), size_(size) {}

  [[nodiscard]] uint64_t pos() const { return pos_; }
  [[nodiscard]] bool ok() const { return ok_; }

  bool bytes(void* dst, uint64_t n) {
    if (!ok_ || n > size_ - pos_) {
      ok_ = false;
      return false;
    }
    std::memcpy(dst, base_ + pos_, static_cast<size_t>(n));
    pos_ += n;
    return true;
  }

  bool skip(uint64_t n) {
    if (!ok_ || n > size_ - pos_) {
      ok_ = false;
      return false;
    }
    pos_ += n;
    return true;
  }

  template <typename T>
  bool read(T* v) {
    return bytes(v, sizeof(T));
  }

  bool read_string(std::string* out) {
    uint64_t n = 0;
    if (!read(&n)) {
      return false;
    }
    if (n > size_ - pos_) {
      ok_ = false;
      return false;
    }
    out->assign(reinterpret_cast<const char*>(base_ + pos_), static_cast<size_t>(n));
    pos_ += n;
    return true;
  }

 private:
  const uint8_t* base_;
  uint64_t size_;
  uint64_t pos_{0};
  bool ok_{true};
};

bool read_scalar(Cursor& c, GgufValueType t, GgufValue* v) {
  switch (t) {
    case GgufValueType::UINT8: {
      uint8_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::INT8: {
      int8_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::UINT16: {
      uint16_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::INT16: {
      int16_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::UINT32: {
      uint32_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::INT32: {
      int32_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = x;
      return true;
    }
    case GgufValueType::FLOAT32: {
      float x = 0;
      if (!c.read(&x)) return false;
      v->f = x;
      v->i = static_cast<int64_t>(x);
      return true;
    }
    case GgufValueType::BOOL: {
      uint8_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x ? 1 : 0;
      v->f = v->i;
      return true;
    }
    case GgufValueType::STRING:
      return c.read_string(&v->s);
    case GgufValueType::UINT64: {
      uint64_t x = 0;
      if (!c.read(&x)) return false;
      v->i = static_cast<int64_t>(x);
      v->f = static_cast<double>(x);
      return true;
    }
    case GgufValueType::INT64: {
      int64_t x = 0;
      if (!c.read(&x)) return false;
      v->i = x;
      v->f = static_cast<double>(x);
      return true;
    }
    case GgufValueType::FLOAT64: {
      double x = 0;
      if (!c.read(&x)) return false;
      v->f = x;
      v->i = static_cast<int64_t>(x);
      return true;
    }
    case GgufValueType::ARRAY:
      return false;  // nested arrays are not part of the format
  }
  return false;
}

bool valid_value_type(uint32_t t) { return t <= static_cast<uint32_t>(GgufValueType::FLOAT64); }

std::string basename_of(const std::string& p) {
  const auto slash = p.find_last_of('/');
  return slash == std::string::npos ? p : p.substr(slash + 1);
}

// llama_ftype -> the label people actually recognise on a download page.
const char* ftype_name(uint64_t ft) {
  switch (ft) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 7: return "Q8_0";
    case 8: return "Q5_0";
    case 9: return "Q5_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K_S";
    case 12: return "Q3_K_M";
    case 13: return "Q3_K_L";
    case 14: return "Q4_K_S";
    case 15: return "Q4_K_M";
    case 16: return "Q5_K_S";
    case 17: return "Q5_K_M";
    case 18: return "Q6_K";
    case 19: return "IQ2_XXS";
    case 20: return "IQ2_XS";
    case 21: return "Q2_K_S";
    case 22: return "IQ3_XS";
    case 23: return "IQ3_XXS";
    case 24: return "IQ1_S";
    case 25: return "IQ4_NL";
    case 26: return "IQ3_S";
    case 27: return "IQ3_M";
    case 28: return "IQ2_S";
    case 29: return "IQ2_M";
    case 30: return "IQ4_XS";
    case 31: return "IQ1_M";
    case 32: return "BF16";
    default: return "";
  }
}

std::string json_escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

}  // namespace

double GgufValue::as_f64() const noexcept { return f; }

const GgmlTypeInfo& ggml_type_info(uint32_t type) {
  for (const auto& row : kTypes) {
    if (row.type == type) {
      return row.info;
    }
  }
  return kUnknownType;
}

uint64_t ggml_row_bytes(uint32_t type, uint64_t n_elements) {
  const auto& info = ggml_type_info(type);
  if (info.block_size == 0) {
    return 0;
  }
  return (n_elements / info.block_size) * info.type_size +
         ((n_elements % info.block_size) ? info.type_size : 0);
}

double ggml_bits_per_weight(uint32_t type) {
  const auto& info = ggml_type_info(type);
  if (info.block_size == 0) {
    return 0.0;
  }
  return 8.0 * static_cast<double>(info.type_size) / static_cast<double>(info.block_size);
}

uint64_t GgufTensorInfo::rows() const noexcept {
  uint64_t r = 1;
  for (size_t i = 1; i < ne.size(); ++i) {
    r *= ne[i];
  }
  return r;
}

GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
  if (base_ && size_) {
    munmap(const_cast<uint8_t*>(base_), static_cast<size_t>(size_));
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
  base_ = nullptr;
  fd_ = -1;
  size_ = 0;
  kv_.clear();
  tensors_.clear();
  tensor_index_.clear();
}

Status GgufFile::open(const std::string& path) {
  close();
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return Status::fail(Errc::not_found, "cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return Status::fail(Errc::io, "cannot stat " + path);
  }
  void* map = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    ::close(fd);
    return Status::fail(Errc::oom, "mmap failed for " + path + ": " + std::strerror(errno));
  }
  fd_ = fd;
  base_ = static_cast<const uint8_t*>(map);
  size_ = static_cast<uint64_t>(st.st_size);
  path_ = path;

  Cursor c(base_, size_);
  uint32_t magic = 0;
  if (!c.read(&magic) || magic != kGgufMagic) {
    close();
    return Status::fail(Errc::protocol, path + " is not a GGUF file (bad magic)");
  }
  if (!c.read(&version_)) {
    close();
    return Status::fail(Errc::protocol, "truncated GGUF header");
  }
  if (version_ < 2 || version_ > 3) {
    close();
    return Status::fail(Errc::protocol,
                        "unsupported GGUF version " + std::to_string(version_) + " (expected 2 or 3)");
  }
  uint64_t n_tensors = 0, n_kv = 0;
  if (!c.read(&n_tensors) || !c.read(&n_kv)) {
    close();
    return Status::fail(Errc::protocol, "truncated GGUF counts");
  }
  if (n_tensors > (1u << 22) || n_kv > (1u << 22)) {
    close();
    return Status::fail(Errc::protocol, "implausible GGUF tensor/kv count");
  }

  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    uint32_t raw_type = 0;
    if (!c.read_string(&key) || !c.read(&raw_type) || !valid_value_type(raw_type)) {
      close();
      return Status::fail(Errc::protocol, "malformed GGUF metadata entry #" + std::to_string(i));
    }
    GgufValue v;
    v.type = static_cast<GgufValueType>(raw_type);
    if (v.type == GgufValueType::ARRAY) {
      uint32_t at = 0;
      uint64_t n = 0;
      if (!c.read(&at) || !valid_value_type(at) || !c.read(&n)) {
        close();
        return Status::fail(Errc::protocol, "malformed GGUF array for key " + key);
      }
      v.array_type = static_cast<GgufValueType>(at);
      if (v.array_type == GgufValueType::STRING) {
        v.array_str.reserve(static_cast<size_t>(std::min<uint64_t>(n, 1u << 20)));
        for (uint64_t j = 0; j < n; ++j) {
          std::string s;
          if (!c.read_string(&s)) {
            close();
            return Status::fail(Errc::protocol, "truncated string array for key " + key);
          }
          v.array_str.push_back(std::move(s));
        }
      } else {
        v.array_num.reserve(static_cast<size_t>(std::min<uint64_t>(n, 1u << 22)));
        for (uint64_t j = 0; j < n; ++j) {
          GgufValue e;
          if (!read_scalar(c, v.array_type, &e)) {
            close();
            return Status::fail(Errc::protocol, "truncated numeric array for key " + key);
          }
          v.array_num.push_back(v.array_type == GgufValueType::FLOAT32 ||
                                        v.array_type == GgufValueType::FLOAT64
                                    ? e.f
                                    : static_cast<double>(e.i));
        }
      }
    } else if (!read_scalar(c, v.type, &v)) {
      close();
      return Status::fail(Errc::protocol, "truncated GGUF value for key " + key);
    }
    kv_.emplace(std::move(key), std::move(v));
  }

  arch_ = str("general.architecture", "");
  const auto align_it = kv_.find("general.alignment");
  if (align_it != kv_.end() && align_it->second.i > 0) {
    alignment_ = static_cast<uint32_t>(align_it->second.i);
  }
  if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
    alignment_ = 32;
  }

  tensors_.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    uint32_t n_dims = 0;
    if (!c.read_string(&t.name) || !c.read(&n_dims) || n_dims == 0 || n_dims > 4) {
      close();
      return Status::fail(Errc::protocol, "malformed GGUF tensor entry #" + std::to_string(i));
    }
    t.ne.resize(n_dims);
    t.n_elements = 1;
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!c.read(&t.ne[d]) || t.ne[d] == 0) {
        close();
        return Status::fail(Errc::protocol, "bad dimension in tensor " + t.name);
      }
      t.n_elements *= t.ne[d];
    }
    if (!c.read(&t.type) || !c.read(&t.offset)) {
      close();
      return Status::fail(Errc::protocol, "truncated tensor entry " + t.name);
    }
    const auto& info = ggml_type_info(t.type);
    if (info.block_size == 0) {
      close();
      return Status::fail(Errc::protocol,
                          "tensor " + t.name + " uses unknown ggml type " + std::to_string(t.type));
    }
    t.nbytes = ggml_row_bytes(t.type, t.ne[0]) * t.rows();
    tensor_index_.emplace(t.name, tensors_.size());
    tensors_.push_back(std::move(t));
  }
  if (!c.ok()) {
    close();
    return Status::fail(Errc::protocol, "GGUF directory extends past end of file");
  }

  data_offset_ = (c.pos() + alignment_ - 1) & ~static_cast<uint64_t>(alignment_ - 1);
  if (data_offset_ > size_) {
    close();
    return Status::fail(Errc::protocol, "GGUF tensor data starts past end of file");
  }
  return Status::OK();
}

const GgufValue* GgufFile::find_kv(const std::string& key) const {
  const auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}

const GgufTensorInfo* GgufFile::find_tensor(const std::string& name) const {
  const auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const uint8_t* GgufFile::tensor_data(const GgufTensorInfo& t) const {
  if (!base_) {
    return nullptr;
  }
  const uint64_t start = data_offset_ + t.offset;
  if (start > size_ || t.nbytes > size_ - start) {
    return nullptr;
  }
  return base_ + start;
}

std::string GgufFile::str(const std::string& key, const std::string& def) const {
  const auto* v = find_kv(key);
  return (v && v->type == GgufValueType::STRING) ? v->s : def;
}

uint64_t GgufFile::u64(const std::string& key, uint64_t def) const {
  const auto* v = find_kv(key);
  return (v && !v->is_array()) ? static_cast<uint64_t>(v->i) : def;
}

double GgufFile::f64(const std::string& key, double def) const {
  const auto* v = find_kv(key);
  return (v && !v->is_array()) ? v->f : def;
}

bool GgufFile::boolean(const std::string& key, bool def) const {
  const auto* v = find_kv(key);
  return (v && !v->is_array()) ? v->i != 0 : def;
}

std::string GgufFile::arch_key(const std::string& suffix) const {
  if (!arch_.empty()) {
    const std::string full = arch_ + "." + suffix;
    if (kv_.count(full)) {
      return full;
    }
  }
  // Some converters emit a bare key, and a few use a different arch prefix than
  // general.architecture; fall back to a suffix match before giving up.
  for (const auto& [k, unused] : kv_) {
    (void)unused;
    if (k.size() > suffix.size() && k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0 &&
        k[k.size() - suffix.size() - 1] == '.') {
      return k;
    }
  }
  return suffix;
}

std::string ModelInfo::param_count_human() const {
  const double p = static_cast<double>(param_count);
  char buf[64];
  if (p >= 1e12) {
    std::snprintf(buf, sizeof(buf), "%.2fT", p / 1e12);
  } else if (p >= 1e9) {
    std::snprintf(buf, sizeof(buf), "%.2fB", p / 1e9);
  } else if (p >= 1e6) {
    std::snprintf(buf, sizeof(buf), "%.1fM", p / 1e6);
  } else {
    std::snprintf(buf, sizeof(buf), "%.0f", p);
  }
  return buf;
}

std::string ModelInfo::to_json() const {
  std::ostringstream os;
  os << "{";
  os << "\"path\":\"" << json_escape(path) << "\"";
  os << ",\"name\":\"" << json_escape(name) << "\"";
  os << ",\"architecture\":\"" << json_escape(architecture) << "\"";
  os << ",\"quantization\":\"" << json_escape(quantization) << "\"";
  os << ",\"dominant_type\":\"" << json_escape(dominant_type) << "\"";
  os << ",\"tokenizer\":\"" << json_escape(tokenizer_model) << "\"";
  os << ",\"supported\":" << (supported_for_inference ? "true" : "false");
  if (!supported_for_inference) {
    os << ",\"unsupported_reason\":\"" << json_escape(unsupported_reason) << "\"";
  }
  os << ",\"file_size_bytes\":" << file_size_bytes;
  os << ",\"tensor_count\":" << tensor_count;
  os << ",\"param_count\":" << param_count;
  os << ",\"param_count_human\":\"" << param_count_human() << "\"";
  os << ",\"bits_per_weight\":" << bits_per_weight;
  os << ",\"n_layers\":" << n_layers;
  os << ",\"n_embd\":" << n_embd;
  os << ",\"n_ff\":" << n_ff;
  os << ",\"n_heads\":" << n_heads;
  os << ",\"n_kv_heads\":" << n_kv_heads;
  os << ",\"head_dim\":" << head_dim;
  os << ",\"n_vocab\":" << n_vocab;
  os << ",\"context_length\":" << context_length;
  os << ",\"rope_freq_base\":" << rope_freq_base;
  os << ",\"rms_eps\":" << rms_eps;
  os << ",\"memory\":{";
  os << "\"weight_bytes\":" << weight_bytes_total;
  os << ",\"bytes_per_layer\":" << bytes_per_layer;
  os << ",\"non_layer_bytes\":" << non_layer_bytes;
  os << ",\"kv_bytes_per_token\":" << kv_bytes_per_token;
  os << ",\"activation_bytes\":" << activation_bytes;
  os << ",\"recommended_ram_bytes\":" << recommended_ram_bytes;
  os << "}";
  os << ",\"types\":{";
  bool first = true;
  for (const auto& [k, v] : type_bytes) {
    if (!first) {
      os << ",";
    }
    first = false;
    const auto cit = type_histogram.find(k);
    os << "\"" << json_escape(k) << "\":{\"tensors\":" << (cit == type_histogram.end() ? 0 : cit->second)
       << ",\"bytes\":" << v << "}";
  }
  os << "}}";
  return os.str();
}

Status inspect_gguf(const std::string& path, ModelInfo* out) {
  GgufFile f;
  auto st = f.open(path);
  if (!st) {
    return st;
  }
  return inspect_gguf(f, out);
}

Status inspect_gguf(const GgufFile& f, ModelInfo* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null ModelInfo");
  }
  if (!f.is_open()) {
    return Status::fail(Errc::invalid_argument, "gguf file is not open");
  }
  ModelInfo m;
  m.path = f.path();
  m.architecture = f.str("general.architecture", "unknown");
  m.name = f.str("general.name", "");
  if (m.name.empty()) {
    m.name = basename_of(f.path());
  }
  m.organization = f.str("general.organization", "");
  m.tokenizer_model = f.str("tokenizer.ggml.model", "");
  m.file_size_bytes = f.file_size();
  m.tensor_count = f.tensors().size();

  m.n_layers = static_cast<uint32_t>(f.u64(f.arch_key("block_count"), 0));
  m.n_embd = static_cast<uint32_t>(f.u64(f.arch_key("embedding_length"), 0));
  m.n_ff = static_cast<uint32_t>(f.u64(f.arch_key("feed_forward_length"), 0));
  m.n_heads = static_cast<uint32_t>(f.u64(f.arch_key("attention.head_count"), 0));
  m.n_kv_heads = static_cast<uint32_t>(f.u64(f.arch_key("attention.head_count_kv"), m.n_heads));
  m.context_length = static_cast<uint32_t>(f.u64(f.arch_key("context_length"), 4096));
  m.rope_dim = static_cast<uint32_t>(f.u64(f.arch_key("rope.dimension_count"), 0));
  m.rope_freq_base = static_cast<float>(f.f64(f.arch_key("rope.freq_base"), 10000.0));
  m.rms_eps = static_cast<float>(f.f64(f.arch_key("attention.layer_norm_rms_epsilon"), 1e-5));
  const double rope_scale = f.f64(f.arch_key("rope.scaling.factor"), 0.0);
  m.rope_freq_scale = rope_scale > 0.0 ? static_cast<float>(1.0 / rope_scale) : 1.0f;

  const auto* vocab = f.find_kv("tokenizer.ggml.tokens");
  if (vocab && vocab->is_array()) {
    m.n_vocab = static_cast<uint32_t>(vocab->array_str.size());
  }

  // Tensor-derived facts.  The directory is authoritative: metadata keys are
  // occasionally missing or stale, but the shapes never lie.
  uint64_t layer_bytes = 0;
  for (const auto& t : f.tensors()) {
    const auto& info = ggml_type_info(t.type);
    m.param_count += t.n_elements;
    m.weight_bytes_total += t.nbytes;
    m.type_histogram[info.name] += 1;
    m.type_bytes[info.name] += t.nbytes;
    if (t.name.rfind("blk.", 0) == 0) {
      layer_bytes += t.nbytes;
    }
    if (t.name == "token_embd.weight" || t.name == "output.weight") {
      if (m.n_vocab == 0 && t.ne.size() >= 2) {
        m.n_vocab = static_cast<uint32_t>(t.ne[1]);
      }
      if (m.n_embd == 0 && !t.ne.empty()) {
        m.n_embd = static_cast<uint32_t>(t.ne[0]);
      }
    }
    if (t.name == "blk.0.ffn_gate.weight" && m.n_ff == 0 && t.ne.size() >= 2) {
      m.n_ff = static_cast<uint32_t>(t.ne[1]);
    }
  }
  m.non_layer_bytes = m.weight_bytes_total > layer_bytes ? m.weight_bytes_total - layer_bytes : 0;
  m.bytes_per_layer = m.n_layers ? layer_bytes / m.n_layers : 0;

  if (m.n_heads && m.n_embd) {
    m.head_dim = static_cast<uint32_t>(f.u64(f.arch_key("attention.key_length"), 0));
    if (m.head_dim == 0) {
      m.head_dim = m.n_embd / m.n_heads;
    }
  }
  if (m.rope_dim == 0) {
    m.rope_dim = m.head_dim;
  }
  if (m.n_kv_heads == 0) {
    m.n_kv_heads = m.n_heads;
  }

  // Quantisation label: prefer general.file_type, fall back to whichever type
  // holds most of the bytes.
  uint64_t best_bytes = 0;
  for (const auto& [name, bytes] : m.type_bytes) {
    if (bytes > best_bytes) {
      best_bytes = bytes;
      m.dominant_type = name;
    }
  }
  const auto* ft = f.find_kv("general.file_type");
  const char* ft_name = ft && !ft->is_array() ? ftype_name(static_cast<uint64_t>(ft->i)) : "";
  m.quantization = (ft_name && *ft_name) ? ft_name
                                         : (m.type_bytes.size() > 1 ? m.dominant_type + " (mixed)"
                                                                    : m.dominant_type);
  m.bits_per_weight =
      m.param_count ? 8.0 * static_cast<double>(m.weight_bytes_total) / static_cast<double>(m.param_count)
                    : 0.0;

  // Memory requirements.
  m.kv_bytes_per_token =
      2ull * m.n_layers * m.n_kv_heads * m.head_dim * dtype_size(DType::F16);
  m.activation_bytes = static_cast<uint64_t>(m.n_embd) * 4ull * 12ull +
                       static_cast<uint64_t>(m.n_ff) * 4ull * 3ull +
                       static_cast<uint64_t>(m.n_vocab) * 4ull;
  const uint64_t kv_full = m.kv_bytes_per_token * std::min<uint64_t>(m.context_length, 8192);
  m.recommended_ram_bytes = m.weight_bytes_total + kv_full + m.activation_bytes + (256ull << 20);

  // Can Oracle's own runner execute it?
  m.supported_for_inference = true;
  if (m.architecture != "llama" && m.architecture != "qwen2" && m.architecture != "mistral" &&
      m.architecture != "minicpm" && m.architecture != "deepseek" && m.architecture != "olmo") {
    m.supported_for_inference = false;
    m.unsupported_reason = "architecture '" + m.architecture +
                           "' is recognised but Oracle's runner only implements the llama-family graph";
  } else if (!f.find_tensor("token_embd.weight")) {
    m.supported_for_inference = false;
    m.unsupported_reason = "token_embd.weight is missing";
  } else if (m.n_layers == 0 || m.n_embd == 0 || m.n_heads == 0) {
    m.supported_for_inference = false;
    m.unsupported_reason = "incomplete metadata (block_count / embedding_length / head_count)";
  } else {
    for (const auto& [name, count] : m.type_histogram) {
      (void)count;
      bool ok = false;
      for (const auto& row : kTypes) {
        if (name == row.info.name) {
          ok = row.info.dequantizable;
          break;
        }
      }
      if (!ok) {
        m.supported_for_inference = false;
        m.unsupported_reason = "quantisation " + name + " has no dequantiser in Oracle";
        break;
      }
    }
  }

  *out = std::move(m);
  return Status::OK();
}

void apply_to_model_meta(const ModelInfo& info, ModelMeta* meta) {
  if (!meta) {
    return;
  }
  meta->name = info.name.empty() ? info.architecture : info.name;
  meta->path = info.path;
  if (info.n_layers) {
    meta->n_layers = info.n_layers;
  }
  if (info.n_embd) {
    meta->hidden_dim = info.n_embd;
  }
  if (info.n_heads) {
    meta->n_heads = info.n_heads;
  }
  if (info.n_kv_heads) {
    meta->n_kv_heads = info.n_kv_heads;
  }
  if (info.head_dim) {
    meta->head_dim = info.head_dim;
  }
  if (info.n_vocab) {
    meta->n_vocab = info.n_vocab;
  }
  if (info.context_length) {
    meta->max_seq = info.context_length;
  }
  meta->bytes_per_layer = info.bytes_per_layer;
  meta->total_weight_bytes = info.weight_bytes_total;
  meta->weight_dtype = info.bits_per_weight <= 9.0 ? DType::I8 : DType::F16;
}

uint64_t kv_bytes_for(const ModelInfo& info, uint32_t n_layers, uint32_t max_seq, DType dtype) {
  return 2ull * n_layers * info.n_kv_heads * info.head_dim * max_seq * dtype_size(dtype);
}

}  // namespace oracle::model
