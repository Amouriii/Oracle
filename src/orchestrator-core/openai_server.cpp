// OpenAI-compatible HTTP surface plus Oracle's own operational endpoints.
//
//   POST /v1/chat/completions   streaming (SSE) and buffered
//   POST /v1/completions
//   GET  /v1/models
//   GET  /health                liveness, no auth by default
//   GET  /cluster               nodes, pipeline, network, scheduler, security
//   GET  /metrics               Prometheus text format
//   GET  /                      the dashboard
//
// Every request passes through the security gate (auth -> size/rate -> slot)
// and then the scheduler (priority queue -> concurrency -> stage health) before
// it is allowed to touch the model.

#include "oracle/orch/pipeline_orchestrator.hpp"
#include "oracle/util/json.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <sstream>

namespace oracle {
namespace {

using json::escape;

std::string client_ip(const httplib::Request& req) {
  // Trust a proxy header only for the left-most hop, and only as a label: it
  // affects rate-limit bucketing, never authorisation.
  const auto fwd = req.get_header_value("X-Forwarded-For");
  if (!fwd.empty()) {
    const auto comma = fwd.find(',');
    return comma == std::string::npos ? fwd : fwd.substr(0, comma);
  }
  return req.remote_addr.empty() ? "unknown" : req.remote_addr;
}

std::string error_body(const std::string& message, const std::string& type, const std::string& code) {
  std::ostringstream os;
  os << "{\"error\":{\"message\":\"" << escape(message) << "\",\"type\":\"" << escape(type)
     << "\",\"param\":null,\"code\":" << (code.empty() ? "null" : "\"" + escape(code) + "\"") << "}}";
  return os.str();
}

void send_error(httplib::Response& res, int status, const std::string& message,
                const std::string& type, const std::string& code, double retry_after = 0) {
  res.status = status;
  if (retry_after > 0) {
    res.set_header("Retry-After", std::to_string(static_cast<int>(retry_after + 0.999)));
  }
  res.set_content(error_body(message, type, code), "application/json");
}

bool apply_decision(const security::Decision& d, httplib::Response& res) {
  if (d.allowed) {
    return true;
  }
  std::string type = "invalid_request_error";
  if (d.http_status == 401 || d.http_status == 403) {
    type = "authentication_error";
  } else if (d.http_status == 429) {
    type = "rate_limit_error";
  } else if (d.http_status >= 500) {
    type = "server_error";
  }
  send_error(res, d.http_status, d.message, type, d.error_code, d.retry_after_seconds);
  return false;
}

std::string now_unix() {
  return std::to_string(
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

struct ParsedChat {
  GenerateRequest gen;
  bool is_chat{true};
  size_t prompt_chars{0};
  size_t message_count{0};
};

// Accepts both the chat and legacy completion shapes.
Status parse_request(const std::string& body, bool chat, ParsedChat* out) {
  auto parsed = json::parse(body);
  if (!parsed.ok) {
    return Status::fail(Errc::invalid_argument, "request body is not valid JSON: " + parsed.error);
  }
  const auto& v = parsed.value;
  if (!v.is_object()) {
    return Status::fail(Errc::invalid_argument, "request body must be a JSON object");
  }

  GenerateRequest g;
  g.model = v.str("model");
  g.max_tokens = static_cast<uint32_t>(v.num("max_tokens", v.num("max_completion_tokens", 256)));
  g.temperature = static_cast<float>(v.num("temperature", 0.0));
  g.top_p = static_cast<float>(v.num("top_p", 1.0));
  g.top_k = static_cast<uint32_t>(v.num("top_k", 0));
  g.repeat_penalty = static_cast<float>(v.num("repetition_penalty", v.num("frequency_penalty", 0.0)) + 1.0);
  g.seed = static_cast<uint64_t>(v.num("seed", 0));
  g.stream = v.boolean("stream", false);
  g.priority = static_cast<int>(v.num("priority", 0));

  if (const auto* stop = v.find("stop")) {
    if (stop->is_string()) {
      g.stop.push_back(stop->string_value());
    } else if (stop->is_array()) {
      for (const auto& s : stop->as_array()) {
        if (s.is_string()) {
          g.stop.push_back(s.string_value());
        }
      }
    }
  }

  size_t chars = 0;
  if (chat) {
    const auto* msgs = v.find("messages");
    if (!msgs || !msgs->is_array() || msgs->as_array().empty()) {
      return Status::fail(Errc::invalid_argument, "'messages' must be a non-empty array");
    }
    for (const auto& m : msgs->as_array()) {
      if (!m.is_object()) {
        return Status::fail(Errc::invalid_argument, "each message must be an object");
      }
      model::ChatMessage cm;
      cm.role = m.str("role", "user");
      const auto* content = m.find("content");
      if (content && content->is_string()) {
        cm.content = content->string_value();
      } else if (content && content->is_array()) {
        // OpenAI's multi-part content: keep the text parts, ignore the rest.
        for (const auto& part : content->as_array()) {
          if (part.str("type", "text") == "text") {
            cm.content += part.str("text");
          }
        }
      } else if (content && !content->is_null()) {
        return Status::fail(Errc::invalid_argument, "message content must be a string or an array");
      }
      if (cm.role != "system" && cm.role != "user" && cm.role != "assistant" && cm.role != "tool" &&
          cm.role != "developer") {
        return Status::fail(Errc::invalid_argument, "unsupported message role '" + cm.role + "'");
      }
      chars += cm.content.size();
      g.messages.push_back(std::move(cm));
    }
    out->message_count = g.messages.size();
  } else {
    const auto* prompt = v.find("prompt");
    if (prompt && prompt->is_string()) {
      g.prompt = prompt->string_value();
    } else if (prompt && prompt->is_array() && !prompt->as_array().empty() &&
               prompt->as_array()[0].is_string()) {
      g.prompt = prompt->as_array()[0].string_value();
    } else {
      return Status::fail(Errc::invalid_argument, "'prompt' must be a string");
    }
    chars = g.prompt.size();
    out->message_count = 1;
  }

  out->gen = std::move(g);
  out->is_chat = chat;
  out->prompt_chars = chars;
  return Status::OK();
}

std::string chunk_json(const std::string& id, const std::string& model, const std::string& delta,
                       const char* finish_reason, bool first) {
  std::ostringstream os;
  os << "{\"id\":\"" << escape(id) << "\",\"object\":\"chat.completion.chunk\",\"created\":" << now_unix()
     << ",\"model\":\"" << escape(model) << "\",\"choices\":[{\"index\":0,\"delta\":{";
  if (first) {
    os << "\"role\":\"assistant\"";
    if (!delta.empty()) {
      os << ",";
    }
  }
  if (!delta.empty()) {
    os << "\"content\":\"" << escape(delta) << "\"";
  }
  os << "},\"finish_reason\":" << (finish_reason ? "\"" + std::string(finish_reason) + "\"" : "null")
     << "}]}";
  return os.str();
}

}  // namespace

Status run_openai_server(PipelineOrchestrator& orch, uint16_t port) {
  httplib::Server svr;
  auto& gate = orch.security();

  svr.set_payload_max_length(static_cast<size_t>(
      std::max<uint64_t>(gate.config().max_request_bytes, 4096)));
  svr.set_read_timeout(60, 0);
  svr.set_write_timeout(600, 0);
  svr.set_keep_alive_max_count(64);

  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string what = "internal error";
    try {
      if (ep) {
        std::rethrow_exception(ep);
      }
    } catch (const std::exception& e) {
      what = e.what();
    } catch (...) {
    }
    res.status = 500;
    res.set_content(error_body(what, "server_error", "internal_error"), "application/json");
  });

  // ---- open endpoints ----------------------------------------------------
  svr.Get("/health", [&orch, &gate](const httplib::Request& req, httplib::Response& res) {
    if (!gate.config().allow_anonymous_health) {
      security::RequestIdentity id;
      if (!apply_decision(gate.authenticate(req.get_header_value("Authorization"), client_ip(req), false,
                                            &id),
                          res)) {
        return;
      }
    }
    res.set_content(orch.health_json(), "application/json");
  });

  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(dashboard_html(), "text/html; charset=utf-8");
  });

  svr.Get("/v1/models", [&orch, &gate](const httplib::Request& req, httplib::Response& res) {
    if (gate.config().require_api_key && !gate.config().allow_anonymous_health) {
      security::RequestIdentity id;
      if (!apply_decision(gate.authenticate(req.get_header_value("Authorization"), client_ip(req), false,
                                            &id),
                          res)) {
        return;
      }
    }
    res.set_content(orch.models_json(), "application/json");
  });

  svr.Get("/cluster", [&orch, &gate](const httplib::Request& req, httplib::Response& res) {
    security::RequestIdentity id;
    // The cluster view is readable by any valid key; only an admin key sees the
    // security detail (recent audit events, limits, key count).
    if (gate.config().require_api_key) {
      if (!apply_decision(gate.authenticate(req.get_header_value("Authorization"), client_ip(req), false,
                                            &id),
                          res)) {
        return;
      }
    }
    res.set_content(orch.cluster_json(id.admin), "application/json");
  });

  svr.Get("/metrics", [&orch](const httplib::Request&, httplib::Response& res) {
    res.set_content(orch.metrics_text(), "text/plain; version=0.0.4; charset=utf-8");
  });

  svr.Get("/v1/security", [&gate](const httplib::Request& req, httplib::Response& res) {
    security::RequestIdentity id;
    if (!apply_decision(gate.authenticate(req.get_header_value("Authorization"), client_ip(req), true,
                                          &id),
                        res)) {
      return;
    }
    res.set_content(gate.status_json(), "application/json");
  });

  // ---- generation --------------------------------------------------------
  auto handle = [&orch, &gate](const httplib::Request& req, httplib::Response& res, bool chat) {
    const std::string ip = client_ip(req);
    security::RequestIdentity id;
    if (!apply_decision(gate.authenticate(req.get_header_value("Authorization"), ip, false, &id), res)) {
      return;
    }
    if (!apply_decision(gate.check_request(id, req.path, req.body.size()), res)) {
      return;
    }

    ParsedChat parsed;
    auto st = parse_request(req.body, chat, &parsed);
    if (!st) {
      gate.audit().warn("request", id.key_id, st.message);
      send_error(res, 400, st.message, "invalid_request_error", "invalid_request");
      return;
    }
    st = gate.validate_generation(id, parsed.prompt_chars, parsed.gen.max_tokens,
                                  parsed.gen.temperature, parsed.message_count);
    if (!st) {
      send_error(res, 400, st.message, "invalid_request_error", "invalid_request");
      return;
    }
    if (!parsed.gen.model.empty() && parsed.gen.model != orch.model_name()) {
      // Be permissive: clients often hard-code a model name.  Say which model
      // actually served the request in the response rather than refusing.
      gate.audit().info("request", id.key_id,
                        "asked for model '" + parsed.gen.model + "'; serving " + orch.model_name());
    }

    security::ConcurrencyLimiter::Lease lease;
    if (!apply_decision(gate.acquire_slot(id, &lease), res)) {
      return;
    }

    Status why;
    auto admission = orch.scheduler().admit(id.key_id, orch.model_name(),
                                            static_cast<uint32_t>(parsed.prompt_chars / 4),
                                            parsed.gen.max_tokens, parsed.gen.priority, &why);
    if (!admission.admitted()) {
      const int status = why.code == Errc::timeout ? 504 : 503;
      const char* code = why.code == Errc::timeout      ? "queue_timeout"
                         : why.code == Errc::worker_dead ? "cluster_degraded"
                                                         : "server_busy";
      gate.audit().warn("capacity", id.key_id, why.message);
      send_error(res, status, why.message, "server_error", code, 2.0);
      return;
    }

    parsed.gen.seq_id = admission.ticket().seq_id;
    parsed.gen.api_key_id = id.key_id;
    parsed.gen.request_id = admission.ticket().id;
    const std::string completion_id = (chat ? "chatcmpl-" : "cmpl-") + admission.ticket().id;
    const std::string model_name = orch.model_name();

    if (parsed.gen.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      res.set_header("X-Oracle-Request-Id", admission.ticket().id);
      // The provider runs on the connection's own thread, so tokens are flushed
      // as they are produced and a disconnect stops generation promptly.
      auto shared_admission = std::make_shared<Admission>(std::move(admission));
      auto shared_lease = std::make_shared<security::ConcurrencyLimiter::Lease>(std::move(lease));
      auto gen = parsed.gen;
      res.set_content_provider(
          "text/event-stream",
          [&orch, gen, completion_id, model_name, shared_admission, shared_lease, chat](
              size_t, httplib::DataSink& sink) {
            bool first = true;
            uint32_t emitted = 0;
            bool aborted = false;
            auto write = [&](const std::string& payload) {
              const std::string frame = "data: " + payload + "\n\n";
              if (!sink.write(frame.data(), frame.size())) {
                aborted = true;
              }
            };
            GenerateResult result;
            auto st = orch.generate(
                gen,
                [&](const GenerateToken& t) {
                  if (aborted) {
                    return;
                  }
                  if (!t.text.empty() || first) {
                    write(chunk_json(completion_id, model_name, t.text, nullptr, first));
                    first = false;
                  }
                  ++emitted;
                },
                &result);
            if (!st) {
              std::ostringstream os;
              os << "{\"error\":{\"message\":\"" << escape(st.message)
                 << "\",\"type\":\"server_error\",\"code\":\"" << errc_name(st.code) << "\"}}";
              write(os.str());
              shared_admission->fail(st.message);
            } else {
              write(chunk_json(completion_id, model_name, {},
                               result.finish_reason.empty() ? "stop" : result.finish_reason.c_str(),
                               first));
              shared_admission->complete(result.completion_tokens);
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            shared_lease->release();
            (void)emitted;
            return true;
          });
      return;
    }

    GenerateResult result;
    std::string text;
    st = orch.generate(parsed.gen, [&](const GenerateToken& t) { text += t.text; }, &result);
    if (!st) {
      admission.fail(st.message);
      gate.audit().warn("request", id.key_id, "generation failed: " + st.message);
      const int status = st.code == Errc::worker_dead ? 503 : (st.code == Errc::timeout ? 504 : 500);
      send_error(res, status, st.message, "server_error", std::string(errc_name(st.code)));
      return;
    }
    admission.complete(result.completion_tokens);

    std::ostringstream os;
    os << "{\"id\":\"" << escape(completion_id) << "\",\"object\":\""
       << (chat ? "chat.completion" : "text_completion") << "\",\"created\":" << now_unix()
       << ",\"model\":\"" << escape(model_name) << "\",\"choices\":[{\"index\":0,";
    if (chat) {
      os << "\"message\":{\"role\":\"assistant\",\"content\":\"" << escape(text) << "\"}";
    } else {
      os << "\"text\":\"" << escape(text) << "\"";
    }
    os << ",\"finish_reason\":\"" << escape(result.finish_reason) << "\"}]"
       << ",\"usage\":{\"prompt_tokens\":" << result.prompt_tokens
       << ",\"completion_tokens\":" << result.completion_tokens
       << ",\"total_tokens\":" << (result.prompt_tokens + result.completion_tokens) << "}"
       << ",\"oracle\":{\"request_id\":\"" << escape(parsed.gen.request_id)
       << "\",\"prefill_ms\":" << result.prefill_ms << ",\"decode_ms\":" << result.decode_ms
       << ",\"tokens_per_second\":" << result.tokens_per_second << "}}";
    res.set_header("X-Oracle-Request-Id", parsed.gen.request_id);
    res.set_content(os.str(), "application/json");
  };

  svr.Post("/v1/chat/completions",
           [&handle](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });
  svr.Post("/v1/completions",
           [&handle](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });

  svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.status = 204;
  });
  svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
  });

  if (!svr.listen("0.0.0.0", static_cast<int>(port))) {
    return Status::fail(Errc::io, "cannot bind the HTTP port " + std::to_string(port));
  }
  return Status::OK();
}

}  // namespace oracle
