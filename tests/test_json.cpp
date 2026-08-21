// The HTTP surface parses untrusted request bodies and emits text produced by a
// model, so both directions need to be well-defined: the parser must reject
// malformed input rather than guess, and the writer must never emit a JSON
// string that is not valid UTF-8.
#include "check.hpp"

#include "oracle/util/json.hpp"

#include <iostream>
#include <string>

using namespace oracle;

namespace {

void check_parsing() {
  const auto r = json::parse(
      R"({"model":"m","messages":[{"role":"user","content":"hi \u00e9\n"}],)"
      R"("max_tokens":42,"stream":true,"temperature":0.7,"stop":["\n\n"],"n":null})");
  CHECK(r.ok);
  CHECK(r.value.is_object());
  CHECK(r.value.str("model") == "m");
  CHECK(r.value.num("max_tokens") == 42);
  CHECK(r.value.boolean("stream"));
  CHECK(r.value.num("temperature") > 0.69 && r.value.num("temperature") < 0.71);

  const auto* msgs = r.value.find("messages");
  CHECK(msgs && msgs->is_array() && msgs->as_array().size() == 1);
  CHECK(msgs->as_array()[0].str("role") == "user");
  // \u00e9 must decode to the two-byte UTF-8 encoding of e-acute.
  CHECK(msgs->as_array()[0].str("content") == "hi \xC3\xA9\n");

  const auto* stop = r.value.find("stop");
  CHECK(stop && stop->is_array() && stop->as_array()[0].string_value() == "\n\n");
  const auto* n = r.value.find("n");
  CHECK(n && n->is_null());

  // Missing and wrong-typed members fall back rather than throwing.
  CHECK(r.value.str("nope", "fallback") == "fallback");
  CHECK(r.value.num("model", -1) == -1);
  CHECK(r.value.find("nope") == nullptr);

  // A surrogate pair becomes one four-byte character.
  const auto emoji = json::parse(R"({"c":"\ud83d\ude00"})");
  CHECK(emoji.ok);
  CHECK(emoji.value.str("c") == "\xF0\x9F\x98\x80");
}

void check_rejection() {
  const char* bad[] = {
      "{bad}", "{\"a\":1}trailing", "{\"a\":}", "{\"a\" 1}", "[1,]", "{\"a\":1",
      "\"unterminated", "{\"a\":\"\\u00\"}", "", "   ", "{\"a\":tru}",
  };
  for (const char* text : bad) {
    const auto r = json::parse(text);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
  }
  // Deeply nested input is refused instead of blowing the stack.
  std::string deep(200, '[');
  CHECK(!json::parse(deep).ok);
}

void check_escaping() {
  CHECK(json::escape("plain") == "plain");
  CHECK(json::escape("a\"b") == "a\\\"b");
  CHECK(json::escape("a\\b") == "a\\\\b");
  CHECK(json::escape("a\nb") == "a\\nb");
  CHECK(json::escape(std::string("a\x01""b")) == "a\\u0001b");
  // Well-formed UTF-8 passes through untouched.
  CHECK(json::escape("caf\xC3\xA9") == "caf\xC3\xA9");
  CHECK(json::escape("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80");

  // A model's byte-fallback tokens, and a generation truncated mid-character,
  // both hand us bytes that are not valid UTF-8.  Every one of them must become
  // U+FFFD so the response stays parseable.
  const std::string replacement = "\xEF\xBF\xBD";
  CHECK(json::escape(std::string("\x9F")) == replacement);          // lone continuation
  CHECK(json::escape(std::string("\xC3")) == replacement);          // truncated 2-byte
  CHECK(json::escape(std::string("\xF0\x9F\x98")) == replacement + replacement + replacement);
  CHECK(json::escape(std::string("\xC0\xAF")) == replacement + replacement);  // overlong '/'
  CHECK(json::escape(std::string("\xED\xA0\x80")) ==
        replacement + replacement + replacement);  // UTF-16 surrogate
  CHECK(json::escape(std::string("ok\x9F""ok")) == "ok" + replacement + "ok");

  // Whatever comes out of escape() must round-trip through the parser: that is
  // the property the API actually depends on.
  const std::string ugly = "iWr\x9F)\xC3yy\xF0\x9F";
  const auto doc = "{\"content\":\"" + json::escape(ugly) + "\"}";
  const auto parsed = json::parse(doc);
  CHECK(parsed.ok);
  CHECK(parsed.value.str("content").find("iWr") == 0);
  CHECK(parsed.value.str("content").find("yy") != std::string::npos);
}

}  // namespace

int main() {
  check_parsing();
  check_rejection();
  check_escaping();
  std::cout << "test_json ok\n";
  return 0;
}
