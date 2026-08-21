#pragma once

// A small, dependency-free JSON reader/writer.
//
// Oracle's HTTP surface has to parse real OpenAI request bodies -- nested
// message arrays, numbers, booleans, escaped strings -- so scanning for
// substrings is not good enough.  This is a strict recursive-descent parser
// with depth and size limits, which matters because the input is untrusted.

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
 public:
  Value() = default;
  explicit Value(bool b) : type_(Type::Bool), bool_(b) {}
  explicit Value(double d) : type_(Type::Number), num_(d) {}
  explicit Value(std::string s) : type_(Type::String), str_(std::move(s)) {}
  explicit Value(Array a) : type_(Type::Array), arr_(std::move(a)) {}
  explicit Value(Object o) : type_(Type::Object), obj_(std::move(o)) {}

  [[nodiscard]] Type type() const noexcept { return type_; }
  [[nodiscard]] bool is_null() const noexcept { return type_ == Type::Null; }
  [[nodiscard]] bool is_object() const noexcept { return type_ == Type::Object; }
  [[nodiscard]] bool is_array() const noexcept { return type_ == Type::Array; }
  [[nodiscard]] bool is_string() const noexcept { return type_ == Type::String; }
  [[nodiscard]] bool is_number() const noexcept { return type_ == Type::Number; }
  [[nodiscard]] bool is_bool() const noexcept { return type_ == Type::Bool; }

  [[nodiscard]] const Object& as_object() const {
    static const Object kEmpty;
    return type_ == Type::Object ? obj_ : kEmpty;
  }
  [[nodiscard]] const Array& as_array() const {
    static const Array kEmpty;
    return type_ == Type::Array ? arr_ : kEmpty;
  }

  // Missing or wrong-typed members fall back to `def` rather than throwing:
  // the HTTP layer validates semantics, not shapes.
  [[nodiscard]] const Value* find(std::string_view key) const {
    if (type_ != Type::Object) {
      return nullptr;
    }
    const auto it = obj_.find(std::string(key));
    return it == obj_.end() ? nullptr : &it->second;
  }
  [[nodiscard]] std::string str(std::string_view key, std::string def = {}) const {
    const auto* v = find(key);
    return (v && v->type_ == Type::String) ? v->str_ : std::move(def);
  }
  [[nodiscard]] double num(std::string_view key, double def = 0.0) const {
    const auto* v = find(key);
    return (v && v->type_ == Type::Number) ? v->num_ : def;
  }
  [[nodiscard]] bool boolean(std::string_view key, bool def = false) const {
    const auto* v = find(key);
    return (v && v->type_ == Type::Bool) ? v->bool_ : def;
  }
  [[nodiscard]] const std::string& string_value() const { return str_; }
  [[nodiscard]] double number_value() const { return num_; }
  [[nodiscard]] bool bool_value() const { return bool_; }

 private:
  Type type_{Type::Null};
  bool bool_{false};
  double num_{0.0};
  std::string str_;
  Array arr_;
  Object obj_;
};

struct ParseResult {
  bool ok{false};
  std::string error;
  Value value;
};

namespace detail {

struct Parser {
  std::string_view s;
  size_t i{0};
  int depth{0};
  std::string error;

  void skip_ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
      ++i;
    }
  }

  bool fail(const char* what) {
    if (error.empty()) {
      error = std::string(what) + " at offset " + std::to_string(i);
    }
    return false;
  }

  bool parse_string(std::string* out) {
    if (i >= s.size() || s[i] != '"') {
      return fail("expected a string");
    }
    ++i;
    out->clear();
    while (i < s.size()) {
      const char c = s[i++];
      if (c == '"') {
        return true;
      }
      if (c != '\\') {
        *out += c;
        continue;
      }
      if (i >= s.size()) {
        return fail("truncated escape");
      }
      const char e = s[i++];
      switch (e) {
        case '"': *out += '"'; break;
        case '\\': *out += '\\'; break;
        case '/': *out += '/'; break;
        case 'b': *out += '\b'; break;
        case 'f': *out += '\f'; break;
        case 'n': *out += '\n'; break;
        case 'r': *out += '\r'; break;
        case 't': *out += '\t'; break;
        case 'u': {
          if (i + 4 > s.size()) {
            return fail("truncated \\u escape");
          }
          uint32_t cp = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = s[i + k];
            cp <<= 4;
            if (h >= '0' && h <= '9') {
              cp |= static_cast<uint32_t>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              cp |= static_cast<uint32_t>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              cp |= static_cast<uint32_t>(h - 'A' + 10);
            } else {
              return fail("bad \\u escape");
            }
          }
          i += 4;
          // Surrogate pair, as emitted by most JSON writers for astral planes.
          if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            uint32_t lo = 0;
            bool good = true;
            for (int k = 0; k < 4; ++k) {
              const char h = s[i + 2 + k];
              lo <<= 4;
              if (h >= '0' && h <= '9') {
                lo |= static_cast<uint32_t>(h - '0');
              } else if (h >= 'a' && h <= 'f') {
                lo |= static_cast<uint32_t>(h - 'a' + 10);
              } else if (h >= 'A' && h <= 'F') {
                lo |= static_cast<uint32_t>(h - 'A' + 10);
              } else {
                good = false;
                break;
              }
            }
            if (good && lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              i += 6;
            }
          }
          if (cp < 0x80) {
            *out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            *out += static_cast<char>(0xC0 | (cp >> 6));
            *out += static_cast<char>(0x80 | (cp & 0x3F));
          } else if (cp < 0x10000) {
            *out += static_cast<char>(0xE0 | (cp >> 12));
            *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            *out += static_cast<char>(0xF0 | (cp >> 18));
            *out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          return fail("unknown escape");
      }
    }
    return fail("unterminated string");
  }

  bool parse_value(Value* out) {
    if (++depth > 32) {
      --depth;
      return fail("nesting is too deep");
    }
    struct Pop {
      int* d;
      ~Pop() { --*d; }
    } pop{&depth};

    skip_ws();
    if (i >= s.size()) {
      return fail("unexpected end of input");
    }
    const char c = s[i];
    if (c == '{') {
      ++i;
      Object obj;
      skip_ws();
      if (i < s.size() && s[i] == '}') {
        ++i;
        *out = Value(std::move(obj));
        return true;
      }
      for (;;) {
        skip_ws();
        std::string key;
        if (!parse_string(&key)) {
          return false;
        }
        skip_ws();
        if (i >= s.size() || s[i] != ':') {
          return fail("expected ':'");
        }
        ++i;
        Value v;
        if (!parse_value(&v)) {
          return false;
        }
        obj.emplace(std::move(key), std::move(v));
        skip_ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') {
          ++i;
          *out = Value(std::move(obj));
          return true;
        }
        return fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++i;
      Array arr;
      skip_ws();
      if (i < s.size() && s[i] == ']') {
        ++i;
        *out = Value(std::move(arr));
        return true;
      }
      for (;;) {
        Value v;
        if (!parse_value(&v)) {
          return false;
        }
        arr.push_back(std::move(v));
        skip_ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') {
          ++i;
          *out = Value(std::move(arr));
          return true;
        }
        return fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      std::string str;
      if (!parse_string(&str)) {
        return false;
      }
      *out = Value(std::move(str));
      return true;
    }
    if (s.compare(i, 4, "true") == 0) {
      i += 4;
      *out = Value(true);
      return true;
    }
    if (s.compare(i, 5, "false") == 0) {
      i += 5;
      *out = Value(false);
      return true;
    }
    if (s.compare(i, 4, "null") == 0) {
      i += 4;
      *out = Value();
      return true;
    }
    // Number
    const size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
      ++i;
    }
    while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                            s[i] == '-' || s[i] == '+')) {
      ++i;
    }
    if (i == start) {
      return fail("unexpected token");
    }
    try {
      *out = Value(std::stod(std::string(s.substr(start, i - start))));
    } catch (...) {
      return fail("bad number");
    }
    return true;
  }
};

}  // namespace detail

inline ParseResult parse(std::string_view text) {
  ParseResult r;
  if (text.size() > (16u << 20)) {
    r.error = "document is too large";
    return r;
  }
  detail::Parser p{text, 0, 0, {}};
  Value v;
  if (!p.parse_value(&v)) {
    r.error = p.error.empty() ? "invalid JSON" : p.error;
    return r;
  }
  p.skip_ws();
  if (p.i != text.size()) {
    r.error = "trailing content after the JSON value";
    return r;
  }
  r.ok = true;
  r.value = std::move(v);
  return r;
}

// Escapes a string for a JSON document *and* guarantees the result is valid
// UTF-8.
//
// A model's byte-fallback tokens are individual bytes, and a generation cut off
// at max_tokens can end mid-character, so the text handed to us is not always
// well-formed.  JSON strings must be valid UTF-8, and a client that stops
// parsing at the first bad byte silently loses the rest of the response, so any
// byte that is not part of a well-formed sequence becomes U+FFFD.
inline std::string escape(std::string_view s) {
  static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";  // U+FFFD
  std::string o;
  o.reserve(s.size() + 8);
  size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        case '\b': o += "\\b"; break;
        case '\f': o += "\\f"; break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            o += buf;
          } else {
            o += static_cast<char>(c);
          }
      }
      ++i;
      continue;
    }

    size_t need = 0;
    uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) {
      need = 2;
      cp = c & 0x1Fu;
    } else if ((c & 0xF0) == 0xE0) {
      need = 3;
      cp = c & 0x0Fu;
    } else if ((c & 0xF8) == 0xF0) {
      need = 4;
      cp = c & 0x07u;
    }
    bool valid = need != 0 && i + need <= s.size();
    for (size_t k = 1; valid && k < need; ++k) {
      const auto cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) {
        valid = false;
        break;
      }
      cp = (cp << 6) | (cc & 0x3Fu);
    }
    if (valid) {
      // Reject overlong encodings, surrogates and anything past U+10FFFF.
      const bool overlong = (need == 2 && cp < 0x80) || (need == 3 && cp < 0x800) ||
                            (need == 4 && cp < 0x10000);
      if (overlong || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        valid = false;
      }
    }
    if (valid) {
      o.append(s.substr(i, need));
      i += need;
    } else {
      o.append(kReplacement);
      ++i;
    }
  }
  return o;
}

inline std::string quote(std::string_view s) { return "\"" + escape(s) + "\""; }

}  // namespace oracle::json
