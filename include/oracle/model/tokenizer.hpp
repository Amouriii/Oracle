#pragma once

// Tokeniser driven entirely by the vocabulary embedded in the GGUF file.
//
// Two families cover essentially every llama-family GGUF in circulation:
//   * "llama" / SPM  - SentencePiece unigram merges scored by the vocab, with
//     <0xNN> byte fallback.
//   * "gpt2" / BPE   - byte-level BPE using the merge table from the file.
// Anything else falls back to a pure byte tokeniser so the engine still runs.

#include "oracle/model/gguf.hpp"
#include "oracle/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace oracle::model {

enum class TokenizerKind { None, Spm, Bpe, Bytes };

struct ChatMessage {
  std::string role;
  std::string content;
};

class Tokenizer {
 public:
  Status load(const GgufFile& f);

  [[nodiscard]] TokenizerKind kind() const noexcept { return kind_; }
  [[nodiscard]] const char* kind_name() const noexcept;
  [[nodiscard]] uint32_t n_vocab() const noexcept { return static_cast<uint32_t>(tokens_.size()); }
  [[nodiscard]] int32_t bos() const noexcept { return bos_; }
  [[nodiscard]] int32_t eos() const noexcept { return eos_; }
  [[nodiscard]] int32_t unk() const noexcept { return unk_; }
  [[nodiscard]] bool add_bos_default() const noexcept { return add_bos_; }
  [[nodiscard]] bool is_eog(int32_t token) const noexcept;
  [[nodiscard]] bool is_control(int32_t token) const noexcept;

  [[nodiscard]] std::vector<int32_t> encode(std::string_view text, bool add_bos) const;
  // Detokenise a single id.  Control tokens render as the empty string so they
  // never leak into a chat completion's content.
  [[nodiscard]] std::string decode(int32_t token) const;
  [[nodiscard]] std::string decode(std::span<const int32_t> tokens) const;
  [[nodiscard]] const std::string& token_text(int32_t token) const;

  // Render an OpenAI-style message list into a prompt using the model's own
  // chat template family (chatml / llama2 / llama3 / mistral / gemma).
  [[nodiscard]] std::string apply_chat_template(const std::vector<ChatMessage>& msgs,
                                                bool add_generation_prompt) const;
  [[nodiscard]] const std::string& chat_template() const noexcept { return chat_template_; }

 private:
  [[nodiscard]] int32_t find(std::string_view piece) const;
  void tokenize_spm(std::string_view text, std::vector<int32_t>* out) const;
  void tokenize_bpe(std::string_view text, std::vector<int32_t>* out) const;
  void tokenize_bytes(std::string_view text, std::vector<int32_t>* out) const;
  void tokenize_fragment(std::string_view text, std::vector<int32_t>* out) const;
  void emit_byte_fallback(unsigned char b, std::vector<int32_t>* out) const;

  TokenizerKind kind_{TokenizerKind::None};
  std::vector<std::string> tokens_;
  std::vector<float> scores_;
  std::vector<int32_t> types_;
  std::unordered_map<std::string, int32_t> index_;
  std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
  std::vector<std::pair<std::string, int32_t>> specials_;  // text -> id, longest first
  int32_t bos_{-1};
  int32_t eos_{-1};
  int32_t unk_{-1};
  int32_t pad_{-1};
  int32_t eot_{-1};
  bool add_bos_{true};
  bool add_space_prefix_{true};
  std::string chat_template_;
  std::string chat_family_;
};

}  // namespace oracle::model
