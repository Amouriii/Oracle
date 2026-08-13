#include "oracle/orch/pipeline_orchestrator.hpp"

namespace oracle {

std::vector<DagNode> build_linear_dag(const ClusterConfig& cfg) {
  std::vector<DagNode> dag;
  dag.reserve(cfg.nodes.size());
  for (size_t i = 0; i < cfg.nodes.size(); ++i) {
    DagNode n;
    n.id = cfg.nodes[i].id;
    n.layers = cfg.nodes[i].layers;
    n.is_embed = (i == 0);
    n.is_lm_head = (i + 1 == cfg.nodes.size());
    dag.push_back(n);
  }
  return dag;
}

}  // namespace oracle
