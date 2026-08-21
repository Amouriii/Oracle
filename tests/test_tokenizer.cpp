// Exercises the SentencePiece path against the fixture vocabulary: merges,
// byte fallback, special-token splitting, round-tripping and chat templating.
#include "gguf_fixture.hpp"

#include "oracle/model/gguf.hpp"
#include "oracle/model/tokenizer.hpp"

#include <algorithm>
#include "check.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <iostream>

int main() {
  const auto path = (std::filesystem::temp_directory_path() / "oracle-tok.gguf").string();
  const auto m = oracle_test::build_tiny_gguf(path);
  CHECK(m.n_vocab > 0);

  oracle::model::GgufFile f;
  auto st = f.open(path);
  if (!st) {
    std::cerr << "open: " << st.message << "\n";
    return 1;
  }
  oracle::model::Tokenizer tok;
  st = tok.load(f);
  if (!st) {
    std::cerr << "tokenizer: " << st.message << "\n";
    return 1;
  }
  CHECK(tok.kind() == oracle::model::TokenizerKind::Spm);
  CHECK(tok.n_vocab() == m.n_vocab);
  CHECK(tok.bos() == 1);
  CHECK(tok.eos() == 2);
  CHECK(tok.is_eog(tok.eos()));
  CHECK(!tok.is_eog(300));
  CHECK(tok.is_control(1));
  CHECK(!tok.is_control(300));

  // Whole words in the vocabulary must win over byte fallback.
  const auto ids = tok.encode("hello world", true);
  CHECK(!ids.empty());
  CHECK(ids.front() == tok.bos());
  bool saw_hello = false, saw_world = false;
  for (int32_t id : ids) {
    saw_hello |= (tok.token_text(id) == "\xE2\x96\x81" "hello");
    saw_world |= (tok.token_text(id) == "\xE2\x96\x81" "world");
  }
  CHECK(saw_hello && saw_world);
  // The greedy merge must consume the whole words, not leave byte crumbs.
  CHECK(ids.size() == 3);
  CHECK(tok.decode(ids) == " hello world");

  // A character with no vocabulary entry falls back to <0xNN> byte tokens and
  // still round-trips exactly.
  const auto bytes = tok.encode("\xC3\xA9", false);
  CHECK(!bytes.empty());
  CHECK(tok.decode(bytes).find("\xC3\xA9") != std::string::npos);

  // Control tokens are matched literally (never shredded into bytes) and render
  // as nothing when decoded.
  const auto with_special = tok.encode("<s>hello", false);
  CHECK(!with_special.empty());
  CHECK(std::find(with_special.begin(), with_special.end(), 1) != with_special.end());
  CHECK(tok.decode(1).empty());
  CHECK(tok.decode(with_special).find("hello") != std::string::npos);

  // Empty input still yields just the BOS when one is requested.
  const auto only_bos = tok.encode("", true);
  CHECK(only_bos.size() == 1 && only_bos[0] == tok.bos());

  // No chat template in the file: the plain transcript form is used.
  const std::vector<oracle::model::ChatMessage> msgs{{"system", "be brief"}, {"user", "hi"}};
  const auto prompt = tok.apply_chat_template(msgs, true);
  CHECK(prompt.find("be brief") != std::string::npos);
  CHECK(prompt.find("hi") != std::string::npos);
  CHECK(prompt.find("Assistant:") != std::string::npos);

  std::remove(path.c_str());
  std::cout << "test_tokenizer ok tokens=" << ids.size() << " kind=" << tok.kind_name() << "\n";
  return 0;
}
