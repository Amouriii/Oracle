#include "oracle/orch/pipeline_orchestrator.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"

#include <chrono>
#include <sstream>

namespace oracle {
namespace {

std::string json_escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        o += "\\\"";
        break;
      case '\\':
        o += "\\\\";
        break;
      case '\n':
        o += "\\n";
        break;
      case '\r':
        o += "\\r";
        break;
      default:
        o += c;
        break;
    }
  }
  return o;
}

std::string extract_json_string(const std::string& body, const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  auto p = body.find(pat);
  if (p == std::string::npos) {
    return {};
  }
  p = body.find(':', p);
  if (p == std::string::npos) {
    return {};
  }
  p = body.find('"', p);
  if (p == std::string::npos) {
    return {};
  }
  ++p;
  std::string out;
  for (; p < body.size(); ++p) {
    if (body[p] == '\\' && p + 1 < body.size()) {
      out += body[p + 1];
      ++p;
      continue;
    }
    if (body[p] == '"') {
      break;
    }
    out += body[p];
  }
  return out;
}

uint32_t extract_json_uint(const std::string& body, const std::string& key, uint32_t def) {
  const std::string pat = "\"" + key + "\"";
  auto p = body.find(pat);
  if (p == std::string::npos) {
    return def;
  }
  p = body.find(':', p);
  if (p == std::string::npos) {
    return def;
  }
  return static_cast<uint32_t>(std::strtoul(body.c_str() + p + 1, nullptr, 10));
}

bool extract_json_bool(const std::string& body, const std::string& key, bool def) {
  const std::string pat = "\"" + key + "\"";
  auto p = body.find(pat);
  if (p == std::string::npos) {
    return def;
  }
  p = body.find(':', p);
  if (p == std::string::npos) {
    return def;
  }
  auto s = body.substr(p + 1, 8);
  if (s.find("true") != std::string::npos) {
    return true;
  }
  if (s.find("false") != std::string::npos) {
    return false;
  }
  return def;
}

}  // namespace

Status run_openai_server(PipelineOrchestrator& orch, uint16_t port) {
  httplib::Server svr;
  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
  });
  svr.Get("/v1/models", [&orch](const httplib::Request&, httplib::Response& res) {
    const auto& m = orch.config().model.name.empty() ? std::string("Oracle") : orch.config().model.name;
    std::ostringstream os;
    os << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(m)
       << "\",\"object\":\"model\",\"owned_by\":\"Oracle\"}]}";
    res.set_content(os.str(), "application/json");
  });
  auto handle_chat = [&orch](const httplib::Request& req, httplib::Response& res) {
    GenerateRequest gr;
    gr.prompt = extract_json_string(req.body, "content");
    if (gr.prompt.empty()) {
      gr.prompt = extract_json_string(req.body, "prompt");
    }
    if (gr.prompt.empty()) {
      gr.prompt = "hello";
    }
    gr.model = extract_json_string(req.body, "model");
    gr.max_tokens = extract_json_uint(req.body, "max_tokens", 32);
    gr.stream = extract_json_bool(req.body, "stream", false);
    if (gr.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      res.set_content_provider("text/event-stream", [&orch, gr](size_t, httplib::DataSink& sink) {
        std::string acc;
        auto st = const_cast<PipelineOrchestrator&>(orch).generate(gr, [&](const GenerateToken& t) {
          std::ostringstream os;
          os << "data: {\"id\":\"cmpl-oracle\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
                "\"delta\":{\"content\":\""
             << json_escape(t.text) << "\"},\"finish_reason\":"
             << (t.stop ? "\"stop\"" : "null") << "}]}\n\n";
          const auto chunk = os.str();
          sink.write(chunk.data(), chunk.size());
        });
        (void)st;
        sink.write("data: [DONE]\n\n", 15);
        sink.done();
        return true;
      });
      return;
    }
    std::string acc;
    auto st = orch.generate(gr, [&](const GenerateToken& t) { acc += t.text; });
    std::ostringstream os;
    os << "{\"id\":\"cmpl-oracle\",\"object\":\"chat.completion\",\"model\":\""
       << json_escape(gr.model.empty() ? "Oracle" : gr.model)
       << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(acc)
       << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":"
       << acc.size() << "}}";
    if (!st) {
      res.status = 503;
      res.set_content("{\"error\":{\"message\":\"" + json_escape(st.message) + "\"}}", "application/json");
      return;
    }
    res.set_content(os.str(), "application/json");
  };
  svr.Post("/v1/chat/completions", handle_chat);
  svr.Post("/v1/completions", handle_chat);
  if (!svr.listen("0.0.0.0", static_cast<int>(port))) {
    return Status::fail(Errc::io, "http listen failed on " + std::to_string(port));
  }
  return Status::OK();
}

}  // namespace oracle
