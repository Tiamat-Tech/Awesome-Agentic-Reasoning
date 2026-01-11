# Awesome Agentic Reasoning Papers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)
[![GitHub stars](https://img.shields.io/github/stars/weitianxin/Awesome-Agentic-Reasoning?style=social)](https://github.com/weitianxin/Awesome-Agentic-Reasoning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/weitianxin/Awesome-Agentic-Reasoning?style=social)](https://github.com/weitianxin/Awesome-Agentic-Reasoning/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/weitianxin/Awesome-Agentic-Reasoning/blob/main/CONTRIBUTING.md)
![Last Commit](https://img.shields.io/github/last-commit/weitianxin/Awesome-Agentic-Reasoning)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=weitianxin.Awesome-Agentic-Reasoning)

This repository organizes research by thematic areas that integrate reasoning with action, including planning, tool use, search, self-evolution through memory and feedback, multi-agent systems, and real-world applications and benchmarks.

> 📄 **Based on the survey**: *[Agentic Reasoning for Large Language Models: A Survey](TBD)*

![Framework overview](figs/overview.png)


## 📋 Table of Contents

- [Awesome Agentic Reasoning Papers](#awesome-agentic-reasoning-papers)
  - [📋 Table of Contents](#-table-of-contents)
  - [1. 🌟 Introduction](#1--introduction)
  - [2. ⚡ From LLM Reasoning to Agentic Reasoning](#2--from-llm-reasoning-to-agentic-reasoning)
  - [3. 🏗️ Foundational Agentic Reasoning](#3-️-foundational-agentic-reasoning)
    - [🗺️ 3.1 Planning Reasoning](#️-31-planning-reasoning)
      - [3.1.1 In-context Planning](#311-in-context-planning)
      - [3.1.2 Post-training Planning](#312-post-training-planning)
    - [🛠️ 3.2 Tool-Use Optimization](#️-32-tool-use-optimization)
      - [3.2.1 In-Context Tool-Integration](#321-in-context-tool-integration)
      - [3.2.2 Post-training Tool-Integration](#322-post-training-tool-integration)
      - [3.2.3 Orchestration-based Tool-Integration](#323-orchestration-based-tool-integration)
    - [🔍 3.3 Agentic Search](#-33-agentic-search)
      - [3.3.1 In-Context Search](#331-in-context-search)
      - [3.3.2 Post-Training Search](#332-post-training-search)
  - [4. 🧬 Self-evolving Agentic Reasoning](#4--self-evolving-agentic-reasoning)
    - [🔄 4.1 Agentic Feedback Mechanisms](#-41-agentic-feedback-mechanisms)
      - [4.1.1 Reflective Feedback](#411-reflective-feedback)
      - [4.1.2 Parametric Adaptation](#412-parametric-adaptation)
      - [4.1.3 Validator-Driven Feedback](#413-validator-driven-feedback)
    - [🧠 4.2 Agentic Memory](#-42-agentic-memory)
      - [4.2.1 Agentic Use of Memory](#421-agentic-use-of-memory)
      - [4.2.2 Structured Memory Representations](#422-structured-memory-representations)
      - [4.2.3 Post-training Memory Control](#423-post-training-memory-control)
    - [🚀 4.3 Evolving Foundational Agentic Capabilities](#-43-evolving-foundational-agentic-capabilities)
      - [4.3.1 Self-evolving Planning](#431-self-evolving-planning)
      - [4.3.2 Self-evolving Tool-use](#432-self-evolving-tool-use)
      - [4.3.3 Self-evolving Search](#433-self-evolving-search)
  - [5. 👥 Collective Multi-agent Reasoning](#5--collective-multi-agent-reasoning)
    - [🎭 5.1 Role Taxonomy of Multi-Agent Systems (MAS)](#-51-role-taxonomy-of-multi-agent-systems-mas)
      - [5.1.1 Generic Roles](#511-generic-roles)
      - [5.1.2 Domain-Specific Roles](#512-domain-specific-roles)
    - [🤝 5.2 Collaboration and Division of Labor](#-52-collaboration-and-division-of-labor)
      - [5.2.1 In-context Collaboration](#521-in-context-collaboration)
      - [5.2.2 Post-training Collaboration](#522-post-training-collaboration)
    - [🌱 5.3 Multi-Agent Evolution](#-53-multi-agent-evolution)
      - [5.3.1 From Single-Agent Evolution to Multi-Agent Evolution](#531-from-single-agent-evolution-to-multi-agent-evolution)
      - [5.3.2 Multi-agent Memory Management for Evolution](#532-multi-agent-memory-management-for-evolution)
      - [5.3.3 Training Multi-agent to Evolve](#533-training-multi-agent-to-evolve)
  - [6. 🎨 Applications](#6--applications)
    - [💻 6.1 Math Exploration \& Vibe Coding Agents](#-61-math-exploration--vibe-coding-agents)
    - [🔬 6.2 Scientific Discovery Agents](#-62-scientific-discovery-agents)
    - [🤖 6.3 Embodied Agents](#-63-embodied-agents)
    - [⚕️ 6.4 Healthcare \& Medicine Agents](#️-64-healthcare--medicine-agents)
    - [🌐 6.5 Autonomous Web Exploration \& Research Agents](#-65-autonomous-web-exploration--research-agents)
  - [7. 📊 Benchmarks](#7--benchmarks)
    - [⚙️ 7.1 Core Mechanisms of Agentic Reasoning](#️-71-core-mechanisms-of-agentic-reasoning)
      - [7.1.1 Tool Use](#711-tool-use)
      - [7.1.2 Memory](#712-memory)
      - [7.1.3 Multi-Agent System](#713-multi-agent-system)
    - [🎯 7.2 Applications of Agentic Reasoning](#-72-applications-of-agentic-reasoning)
      - [7.2.1 Embodied Agents](#721-embodied-agents)
      - [7.2.2 Scientific Discovery Agents](#722-scientific-discovery-agents)
      - [7.2.3 Autonomous Research Agents](#723-autonomous-research-agents)
      - [7.2.4 Medical and Clinical Agents](#724-medical-and-clinical-agents)
      - [7.2.5 Web Agents](#725-web-agents)
      - [7.2.6 General Tool-Use Agents](#726-general-tool-use-agents)
  - [🤝 Contributing](#-contributing)
  - [Citation](#citation)
  - [License](#license)
  - [Star History](#star-history)

---

## 1. 🌟 Introduction

Bridging thought and action through autonomous agents that reason, act, and learn via continual interaction with their environments. The goal is to enhance agent capabilities by grounding reasoning in action.

We organize agentic reasoning into three layers, each corresponding to a distinct reasoning paradigm under different *environmental dynamics*:

🔹 **Foundational Reasoning.** Core single-agent abilities (planning, tool-use, search) in stable environments

🔹 **Self-Evolving Reasoning.** Adaptation through feedback, memory, and learning in dynamic settings

🔹 **Collective Reasoning.** Multi-agent coordination, role specialization, and collaborative intelligence

Across these layers, we further identify complementary reasoning paradigms defined by their *optimization settings*.

🔸 **In-Context Reasoning.** Test-time scaling through structured orchestration and adaptive workflows

🔸 **Post-Training Reasoning.** Behavior optimization via RL and supervised fine-tuning

---

## 2. ⚡ From LLM Reasoning to Agentic Reasoning


| Dimension | LLM Reasoning |  | Agentic Reasoning |
|----------|---------------|--|-------------------|
| **Paradigm** | passive | ↔ | interactive |
|          | static input | ↔ | dynamic context |
| **Computation** | single pass | ↔ | multi-step |
|          | internal compute | ↔ | with feedback |
| **Statefulness** | context window | ↔ | external memory |
|          | no persistence | ↔ | state tracking |
| **Learning** | offline pretraining | ↔ | continual improvement |
|          | fixed knowledge | ↔ | self-evolving |
| **Goal Orientation** | prompt-based | ↔ | explicit goals |
|          | reactive | ↔ | planning |


---

## 3. 🏗️ Foundational Agentic Reasoning

### 🗺️ 3.1 Planning Reasoning

![plan](figs/planning.png)


#### 3.1.1 In-context Planning


| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 3.1.2 Post-training Planning

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🛠️ 3.2 Tool-Use Optimization

![tool](figs/tool_use.png)


#### 3.2.1 In-Context Tool-Integration

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 3.2.2 Post-training Tool-Integration

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 3.2.3 Orchestration-based Tool-Integration

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🔍 3.3 Agentic Search

![search](figs/search.png)


#### 3.3.1 In-Context Search

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 3.3.2 Post-Training Search

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

---

## 4. 🧬 Self-evolving Agentic Reasoning

### 🔄 4.1 Agentic Feedback Mechanisms

![feed](figs/feedback.png)


#### 4.1.1 Reflective Feedback

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.1.2 Parametric Adaptation

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.1.3 Validator-Driven Feedback

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🧠 4.2 Agentic Memory

![mem](figs/memory.png)


#### 4.2.1 Agentic Use of Memory

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.2.2 Structured Memory Representations

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.2.3 Post-training Memory Control

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🚀 4.3 Evolving Foundational Agentic Capabilities

![mem](figs/evolve.png)

#### 4.3.1 Self-evolving Planning

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.3.2 Self-evolving Tool-use

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 4.3.3 Self-evolving Search

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

---

## 5. 👥 Collective Multi-agent Reasoning

### 🎭 5.1 Role Taxonomy of Multi-Agent Systems (MAS)

![mem](figs/mas.png)

#### 5.1.1 Generic Roles

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 5.1.2 Domain-Specific Roles

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🤝 5.2 Collaboration and Division of Labor

![collab](figs/multi-agent-collab.png)


#### 5.2.1 In-context Collaboration

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 5.2.2 Post-training Collaboration

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🌱 5.3 Multi-Agent Evolution

![mmem](figs/multi-agent memory.png)

#### 5.3.1 From Single-Agent Evolution to Multi-Agent Evolution

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 5.3.2 Multi-agent Memory Management for Evolution

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

#### 5.3.3 Training Multi-agent to Evolve

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

---

## 6. 🎨 Applications

![app](figs/application.png)


### 💻 6.1 Math Exploration & Vibe Coding Agents

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🔬 6.2 Scientific Discovery Agents

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🤖 6.3 Embodied Agents

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### ⚕️ 6.4 Healthcare & Medicine Agents

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

### 🌐 6.5 Autonomous Web Exploration & Research Agents

| ID | Paper | Venue | Link |
|-------|-------|-------|------|
| ID | Paper Title | Venue Year | [Paper]() |

---

## 7. 📊 Benchmarks

![bench](figs/benchmark.png)


### ⚙️ 7.1 Core Mechanisms of Agentic Reasoning

#### 7.1.1 Tool Use

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.1.2 Memory

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.1.3 Multi-Agent System

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

### 🎯 7.2 Applications of Agentic Reasoning

#### 7.2.1 Embodied Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.2.2 Scientific Discovery Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.2.3 Autonomous Research Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.2.4 Medical and Clinical Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.2.5 Web Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

#### 7.2.6 General Tool-Use Agents

| Benchmark | Paper | Link |
|-----------|-------|------|
| Benchmark Name | Venue Year | [Paper]() [Code]() |

---

## 🤝 Contributing
This collection is an ongoing effort. We are actively expanding and refining its coverage, and welcome contributions from the community. You can:

- Submit a pull request to add papers or resources
- Open an issue to suggest additional papers or resources
- Email us at twei10@illinois.edu

We regularly update the repository to include new research.
## Citation

If you find this repository or paper useful, please consider citing the survey paper:

```bibtex
@article{wei2026agent,
  title={Agentic Reasoning for Large Language Models},
  author={Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This repository is licensed under the MIT License.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weitianxin/Awesome-Agentic-Reasoning&type=Date)](https://star-history.com/#weitianxin/Awesome-Agentic-Reasoning&Date)