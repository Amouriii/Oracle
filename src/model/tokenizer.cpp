#include "oracle/model/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <queue>

namespace oracle::model {
namespace {

// GGUF token_type values (llama_token_type).
constexpr int32_t kTypeNormal = 1;
constexpr int32_t kTypeUnknown = 2;
constexpr int32_t kTypeControl = 3;
constexpr int32_t kTypeUserDefined = 4;
constexpr int32_t kTypeByte = 6;

const std::string kSpmSpace = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK

size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;  // stray continuation byte: treat as its own symbol
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return s;
  }
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, from.size(), from) == 0) {
      out += to;
      i += from.size();
    } else {
      out += s[i++];
    }
  }
  return out;
}

// ---- GPT-2 byte <-> unicode alphabet -------------------------------------
// Byte-level BPE stores bytes as printable codepoints; these two tables are the
// standard mapping used by every GPT-2 derived tokeniser.
struct ByteAlphabet {
  std::array<std::string, 256> to_unicode;
  std::unordered_map<std::string, unsigned char> from_unicode;

  ByteAlphabet() {
    auto encode_cp = [](uint32_t cp) {
      std::string s;
      if (cp < 0x80) {
        s += static_cast<char>(cp);
      } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
      } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
      }
      return s;
    };
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      const uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
      to_unicode[static_cast<size_t>(b)] = encode_cp(cp);
      from_unicode[to_unicode[static_cast<size_t>(b)]] = static_cast<unsigned char>(b);
    }
  }
};

const ByteAlphabet& byte_alphabet() {
  static const ByteAlphabet a;
  return a;
}

uint32_t decode_cp(std::string_view s, size_t i, size_t* len) {
  const auto c = static_cast<unsigned char>(s[i]);
  const size_t n = utf8_len(c);
  *len = std::min(n, s.size() - i);
  if (*len == 1) {
    return c;
  }
  uint32_t cp = c & (0xFF >> (n + 1));
  for (size_t k = 1; k < *len; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  }
  return cp;
}

enum class CharClass { Space, Letter, Number, Other };

CharClass classify(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C) {
    return CharClass::Space;
  }
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
    return CharClass::Letter;
  }
  if (cp >= '0' && cp <= '9') {
    return CharClass::Number;
  }
  if (cp < 0x80) {
    return CharClass::Other;
  }
  // Outside ASCII, treat CJK/Latin/Cyrillic/etc. as letters and the common
  // punctuation blocks as "other".  This mirrors the GPT-2 regex closely enough
  // that merges reproduce reference ids on ordinary prose.
  if ((cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) ||
      (cp >= 0xFF00 && cp <= 0xFF0F) || (cp >= 0x2190 && cp <= 0x2BFF)) {
    return CharClass::Other;
  }
  return CharClass::Letter;
}

// Approximation of the GPT-2 pre-tokenizer regex, hand-rolled so we do not need
// a Unicode-property regex engine.
std::vector<std::string> gpt2_pretokenize(std::string_view text) {
  static const char* kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  std::vector<std::string> out;
  size_t i = 0;
  while (i < text.size()) {
    bool matched = false;
    for (const char* c : kContractions) {
      const size_t n = std::strlen(c);
      if (text.compare(i, n, c) == 0) {
        out.emplace_back(text.substr(i, n));
        i += n;
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    size_t start = i;
    size_t len = 0;
    uint32_t cp = decode_cp(text, i, &len);
    // An optional single leading space joins the following run.
    size_t j = i;
    if (cp == ' ' && i + len < text.size()) {
      size_t len2 = 0;
      const uint32_t next = decode_cp(text, i + len, &len2);
      const CharClass nc = classify(next);
      if (nc == CharClass::Letter || nc == CharClass::Number || nc == CharClass::Other) {
        j = i + len;
        cp = next;
        len = len2;
      }
    }
    const CharClass cls = classify(cp);
    if (cls == CharClass::Space) {
      // Runs of whitespace: keep all but the last space with this piece if more
      // text follows (the `\s+(?!\S)` branch).
      size_t k = i;
      while (k < text.size()) {
        size_t l = 0;
        if (classify(decode_cp(text, k, &l)) != CharClass::Space) {
          break;
        }
        k += l;
      }
      size_t end = k;
      if (k < text.size() && k - i > 1) {
        --end;
      }
      out.emplace_back(text.substr(i, end - i));
      i = end;
      continue;
    }
    size_t k = j;
    while (k < text.size()) {
      size_t l = 0;
      const uint32_t c2 = decode_cp(text, k, &l);
      if (classify(c2) != cls) {
        break;
      }
      k += l;
    }
    out.emplace_back(text.substr(start, k - start));
    i = k;
  }
  return out;
}

struct SpmSymbol {
  int prev{-1};
  int next{-1};
  const char* text{nullptr};
  size_t n{0};
};

struct SpmBigram {
  int left{-1};
  int right{-1};
  float score{0.f};
  size_t size{0};
  bool operator<(const SpmBigram& o) const {
    return score < o.score || (score == o.score && left > o.left);
  }
};

std::string detect_chat_family(const std::string& tmpl, const Tokenizer& tok) {
  if (tmpl.find("<|im_start|>") != std::string::npos) {
    return "chatml";
  }
  if (tmpl.find("<|start_header_id|>") != std::string::npos) {
    return "llama3";
  }
  if (tmpl.find("[INST]") != std::string::npos) {
    return tmpl.find("<<SYS>>") != std::string::npos ? "llama2" : "mistral";
  }
  if (tmpl.find("<start_of_turn>") != std::string::npos) {
    return "gemma";
  }
  if (!tmpl.empty()) {
    return "unknown";
  }
  // No template in the file: infer from the special tokens that exist.
  if (tok.n_vocab()) {
    return "plain";
  }
  return "plain";
}

}  // namespace

const char* Tokenizer::kind_name() const noexcept {
  switch (kind_) {
    case TokenizerKind::Spm: return "spm";
    case TokenizerKind::Bpe: return "bpe";
    case TokenizerKind::Bytes: return "bytes";
    case TokenizerKind::None: break;
  }
  return "none";
}

Status Tokenizer::load(const GgufFile& f) {
  tokens_.clear();
  scores_.clear();
  types_.clear();
  index_.clear();
  merge_rank_.clear();
  specials_.clear();

  const auto* toks = f.find_kv("tokenizer.ggml.tokens");
  if (!toks || !toks->is_array() || toks->array_str.empty()) {
    kind_ = TokenizerKind::Bytes;
    return Status::fail(Errc::not_found, "GGUF has no tokenizer.ggml.tokens; falling back to bytes");
  }
  tokens_ = toks->array_str;
  index_.reserve(tokens_.size() * 2);
  for (size_t i = 0; i < tokens_.size(); ++i) {
    index_.emplace(tokens_[i], static_cast<int32_t>(i));
  }

  if (const auto* sc = f.find_kv("tokenizer.ggml.scores"); sc && sc->is_array()) {
    scores_.reserve(sc->array_num.size());
    for (double d : sc->array_num) {
      scores_.push_back(static_cast<float>(d));
    }
  }
  scores_.resize(tokens_.size(), 0.f);

  if (const auto* ty = f.find_kv("tokenizer.ggml.token_type"); ty && ty->is_array()) {
    types_.reserve(ty->array_num.size());
    for (double d : ty->array_num) {
      types_.push_back(static_cast<int32_t>(d));
    }
  }
  types_.resize(tokens_.size(), kTypeNormal);

  const std::string model = f.str("tokenizer.ggml.model", "");
  if (model == "gpt2" || model == "bpe") {
    kind_ = TokenizerKind::Bpe;
  } else if (model == "llama" || model == "spm" || model.empty()) {
    kind_ = TokenizerKind::Spm;
  } else {
    kind_ = TokenizerKind::Bytes;
  }

  if (const auto* mg = f.find_kv("tokenizer.ggml.merges"); mg && mg->is_array()) {
    for (size_t i = 0; i < mg->array_str.size(); ++i) {
      merge_rank_.emplace(mg->array_str[i], static_cast<int32_t>(i));
    }
    if (!merge_rank_.empty() && kind_ == TokenizerKind::Spm && model.empty()) {
      kind_ = TokenizerKind::Bpe;
    }
  }
  if (kind_ == TokenizerKind::Bpe && merge_rank_.empty()) {
    // A byte-level vocab with no merge table can still round-trip single bytes.
    kind_ = TokenizerKind::Bytes;
  }

  auto tok_id = [&](const char* key, int32_t def) -> int32_t {
    const auto* v = f.find_kv(key);
    if (!v || v->is_array()) {
      return def;
    }
    const auto id = static_cast<int32_t>(v->i);
    return (id >= 0 && static_cast<size_t>(id) < tokens_.size()) ? id : def;
  };
  bos_ = tok_id("tokenizer.ggml.bos_token_id", -1);
  eos_ = tok_id("tokenizer.ggml.eos_token_id", -1);
  unk_ = tok_id("tokenizer.ggml.unknown_token_id", -1);
  pad_ = tok_id("tokenizer.ggml.padding_token_id", -1);
  eot_ = tok_id("tokenizer.ggml.eot_token_id", -1);
  add_bos_ = f.boolean("tokenizer.ggml.add_bos_token", kind_ == TokenizerKind::Spm);
  add_space_prefix_ = f.boolean("tokenizer.ggml.add_space_prefix", kind_ == TokenizerKind::Spm);
  chat_template_ = f.str("tokenizer.chat_template", "");
  chat_family_ = detect_chat_family(chat_template_, *this);

  // Control and user-defined tokens must be matched literally before the normal
  // tokeniser runs, otherwise "<|im_start|>" is shredded into fragments.
  for (size_t i = 0; i < tokens_.size(); ++i) {
    const int32_t t = types_[i];
    if ((t == kTypeControl || t == kTypeUserDefined) && !tokens_[i].empty()) {
      specials_.emplace_back(tokens_[i], static_cast<int32_t>(i));
    }
  }
  std::sort(specials_.begin(), specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  return Status::OK();
}

bool Tokenizer::is_control(int32_t token) const noexcept {
  if (token < 0 || static_cast<size_t>(token) >= types_.size()) {
    return false;
  }
  return types_[static_cast<size_t>(token)] == kTypeControl;
}

bool Tokenizer::is_eog(int32_t token) const noexcept {
  if (token < 0) {
    return true;
  }
  if (token == eos_ || (eot_ >= 0 && token == eot_)) {
    return true;
  }
  if (static_cast<size_t>(token) < tokens_.size()) {
    const std::string& s = tokens_[static_cast<size_t>(token)];
    if (s == "<|im_end|>" || s == "<|endoftext|>" || s == "<|eot_id|>" || s == "<end_of_turn>") {
      return true;
    }
  }
  return false;
}

int32_t Tokenizer::find(std::string_view piece) const {
  const auto it = index_.find(std::string(piece));
  return it == index_.end() ? -1 : it->second;
}

const std::string& Tokenizer::token_text(int32_t token) const {
  static const std::string kEmpty;
  if (token < 0 || static_cast<size_t>(token) >= tokens_.size()) {
    return kEmpty;
  }
  return tokens_[static_cast<size_t>(token)];
}

void Tokenizer::emit_byte_fallback(unsigned char b, std::vector<int32_t>* out) const {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
  const int32_t id = find(buf);
  if (id >= 0) {
    out->push_back(id);
    return;
  }
  const auto& alpha = byte_alphabet();
  const int32_t bid = find(alpha.to_unicode[b]);
  if (bid >= 0) {
    out->push_back(bid);
    return;
  }
  if (unk_ >= 0) {
    out->push_back(unk_);
  }
}

void Tokenizer::tokenize_spm(std::string_view text, std::vector<int32_t>* out) const {
  if (text.empty()) {
    return;
  }
  std::string norm = replace_all(std::string(text), " ", kSpmSpace);

  std::vector<SpmSymbol> syms;
  syms.reserve(norm.size());
  for (size_t i = 0; i < norm.size();) {
    const size_t n = std::min(utf8_len(static_cast<unsigned char>(norm[i])), norm.size() - i);
    SpmSymbol s;
    s.text = norm.data() + i;
    s.n = n;
    s.prev = static_cast<int>(syms.size()) - 1;
    s.next = (i + n >= norm.size()) ? -1 : static_cast<int>(syms.size()) + 1;
    syms.push_back(s);
    i += n;
  }
  if (syms.empty()) {
    return;
  }

  std::priority_queue<SpmBigram> work;
  auto try_add = [&](int left, int right) {
    if (left == -1 || right == -1) {
      return;
    }
    const std::string_view merged(syms[left].text, syms[left].n + syms[right].n);
    const int32_t id = find(merged);
    if (id < 0) {
      return;
    }
    SpmBigram b;
    b.left = left;
    b.right = right;
    b.score = scores_[static_cast<size_t>(id)];
    b.size = merged.size();
    work.push(b);
  };
  for (size_t i = 1; i < syms.size(); ++i) {
    try_add(static_cast<int>(i) - 1, static_cast<int>(i));
  }

  while (!work.empty()) {
    const SpmBigram b = work.top();
    work.pop();
    SpmSymbol& l = syms[static_cast<size_t>(b.left)];
    SpmSymbol& r = syms[static_cast<size_t>(b.right)];
    if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) {
      continue;  // one side was already merged into something else
    }
    l.n += r.n;
    r.n = 0;
    l.next = r.next;
    if (r.next >= 0) {
      syms[static_cast<size_t>(r.next)].prev = b.left;
    }
    try_add(l.prev, b.left);
    try_add(b.left, l.next);
  }

  for (int i = 0; i >= 0; i = syms[static_cast<size_t>(i)].next) {
    const SpmSymbol& s = syms[static_cast<size_t>(i)];
    if (s.n == 0) {
      continue;
    }
    const std::string_view piece(s.text, s.n);
    const int32_t id = find(piece);
    if (id >= 0) {
      out->push_back(id);
    } else {
      for (char c : piece) {
        emit_byte_fallback(static_cast<unsigned char>(c), out);
      }
    }
  }
}

void Tokenizer::tokenize_bpe(std::string_view text, std::vector<int32_t>* out) const {
  const auto& alpha = byte_alphabet();
  for (const auto& piece : gpt2_pretokenize(text)) {
    // Byte-level encode, one symbol per input byte.
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (unsigned char c : piece) {
      sym.push_back(alpha.to_unicode[c]);
    }
    // Repeatedly apply the lowest-rank adjacent merge.
    for (;;) {
      int best_rank = -1;
      size_t best_i = 0;
      for (size_t i = 0; i + 1 < sym.size(); ++i) {
        const auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
        if (it != merge_rank_.end() && (best_rank < 0 || it->second < best_rank)) {
          best_rank = it->second;
          best_i = i;
        }
      }
      if (best_rank < 0) {
        break;
      }
      sym[best_i] += sym[best_i + 1];
      sym.erase(sym.begin() + static_cast<long>(best_i) + 1);
    }
    for (const auto& s : sym) {
      const int32_t id = find(s);
      if (id >= 0) {
        out->push_back(id);
        continue;
      }
      for (char c : s) {
        emit_byte_fallback(static_cast<unsigned char>(c), out);
      }
    }
  }
}

void Tokenizer::tokenize_bytes(std::string_view text, std::vector<int32_t>* out) const {
  for (char c : text) {
    emit_byte_fallback(static_cast<unsigned char>(c), out);
  }
}

void Tokenizer::tokenize_fragment(std::string_view text, std::vector<int32_t>* out) const {
  switch (kind_) {
    case TokenizerKind::Spm:
      tokenize_spm(text, out);
      return;
    case TokenizerKind::Bpe:
      tokenize_bpe(text, out);
      return;
    default:
      tokenize_bytes(text, out);
      return;
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos) const {
  std::vector<int32_t> out;
  if (tokens_.empty()) {
    return out;
  }
  if (add_bos && bos_ >= 0) {
    out.push_back(bos_);
  }
  std::string work(text);
  if (kind_ == TokenizerKind::Spm && add_space_prefix_ && !work.empty() && work[0] != ' ') {
    work.insert(work.begin(), ' ');
  }

  // Split around literal special tokens, tokenising the gaps normally.
  size_t i = 0;
  size_t frag_start = 0;
  while (i < work.size()) {
    const std::pair<std::string, int32_t>* hit = nullptr;
    for (const auto& sp : specials_) {
      if (sp.first.size() <= work.size() - i && work.compare(i, sp.first.size(), sp.first) == 0) {
        hit = &sp;
        break;  // specials_ is sorted longest-first
      }
    }
    if (!hit) {
      ++i;
      continue;
    }
    if (i > frag_start) {
      tokenize_fragment(std::string_view(work).substr(frag_start, i - frag_start), &out);
    }
    out.push_back(hit->second);
    i += hit->first.size();
    frag_start = i;
  }
  if (frag_start < work.size()) {
    tokenize_fragment(std::string_view(work).substr(frag_start), &out);
  }
  return out;
}

std::string Tokenizer::decode(int32_t token) const {
  if (token < 0 || static_cast<size_t>(token) >= tokens_.size()) {
    return {};
  }
  const size_t idx = static_cast<size_t>(token);
  const int32_t type = types_[idx];
  if (type == kTypeControl) {
    return {};
  }
  const std::string& s = tokens_[idx];
  if (type == kTypeByte || (s.size() == 6 && s.compare(0, 3, "<0x") == 0 && s.back() == '>')) {
    const int v = static_cast<int>(std::strtol(s.substr(3, 2).c_str(), nullptr, 16));
    return std::string(1, static_cast<char>(v));
  }
  if (kind_ == TokenizerKind::Bpe) {
    const auto& alpha = byte_alphabet();
    std::string out;
    for (size_t i = 0; i < s.size();) {
      const size_t n = std::min(utf8_len(static_cast<unsigned char>(s[i])), s.size() - i);
      const auto it = alpha.from_unicode.find(s.substr(i, n));
      if (it != alpha.from_unicode.end()) {
        out += static_cast<char>(it->second);
      } else {
        out += s.substr(i, n);
      }
      i += n;
    }
    return out;
  }
  return replace_all(s, kSpmSpace, " ");
}

std::string Tokenizer::decode(std::span<const int32_t> tokens) const {
  std::string out;
  for (int32_t t : tokens) {
    out += decode(t);
  }
  return out;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& msgs,
                                           bool add_generation_prompt) const {
  std::string out;
  const std::string& fam = chat_family_;

  if (fam == "chatml" || fam == "unknown") {
    for (const auto& m : msgs) {
      out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    }
    if (add_generation_prompt) {
      out += "<|im_start|>assistant\n";
    }
    return out;
  }
  if (fam == "llama3") {
    out += "<|begin_of_text|>";
    for (const auto& m : msgs) {
      out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
    }
    if (add_generation_prompt) {
      out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return out;
  }
  if (fam == "llama2" || fam == "mistral") {
    std::string system;
    for (const auto& m : msgs) {
      if (m.role == "system") {
        system = m.content;
      }
    }
    bool first = true;
    for (const auto& m : msgs) {
      if (m.role == "system") {
        continue;
      }
      if (m.role == "user") {
        out += "[INST] ";
        if (first && !system.empty() && fam == "llama2") {
          out += "<<SYS>>\n" + system + "\n<</SYS>>\n\n";
        } else if (first && !system.empty()) {
          out += system + "\n\n";
        }
        out += m.content + " [/INST]";
        first = false;
      } else {
        out += " " + m.content + "</s>";
      }
    }
    return out;
  }
  if (fam == "gemma") {
    for (const auto& m : msgs) {
      const std::string role = m.role == "assistant" ? "model" : "user";
      out += "<start_of_turn>" + role + "\n" + m.content + "<end_of_turn>\n";
    }
    if (add_generation_prompt) {
      out += "<start_of_turn>model\n";
    }
    return out;
  }

  // No usable template: a plain transcript, which is what a base model expects.
  for (const auto& m : msgs) {
    if (m.role == "system") {
      out += m.content + "\n\n";
    } else if (m.role == "user") {
      out += "User: " + m.content + "\n";
    } else {
      out += "Assistant: " + m.content + "\n";
    }
  }
  if (add_generation_prompt) {
    out += "Assistant:";
  }
  return out;
}

}  // namespace oracle::model
