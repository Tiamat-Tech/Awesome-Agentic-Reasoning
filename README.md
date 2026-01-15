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
- [📋 Table of Contents](#-table-of-contents)
- [🌟 Introduction](#-introduction)
- [🤝 Contributing](#-contributing)
- [📝 Citation](#-citation)
- [🏗️ Foundational Agentic Reasoning](#-foundational-agentic-reasoning)
  - [🗺️ Planning Reasoning](#-planning-reasoning)
  - [🛠️ Tool-Use Optimization](#-tool-use-optimization)
  - [🔍 Agentic Search](#-agentic-search)
- [🧬 Self-evolving Agentic Reasoning](#-self-evolving-agentic-reasoning)
  - [🔄 Agentic Feedback Mechanisms](#-agentic-feedback-mechanisms)
  - [🧠 Agentic Memory](#-agentic-memory)
  - [🚀 Evolving Foundational Agentic Capabilities](#-evolving-foundational-agentic-capabilities)
- [👥 Collective Multi-agent Reasoning](#-collective-multi-agent-reasoning)
  - [🎭 Role Taxonomy of Multi-Agent Systems (MAS)](#-role-taxonomy-of-multi-agent-systems-mas)
  - [🤝 Collaboration and Division of Labor](#-collaboration-and-division-of-labor)
  - [🌱 Multi-Agent Evolution](#-multi-agent-evolution)
- [🎨 Applications](#-applications)
  - [💻 Math Exploration & Vibe Coding Agents](#-math-exploration--vibe-coding-agents)
  - [🔬 Scientific Discovery Agents](#-scientific-discovery-agents)
  - [🤖 Embodied Agents](#-embodied-agents)
  - [🏥 Healthcare & Medicine Agents](#-healthcare--medicine-agents)
  - [🌐 Autonomous Web Exploration & Research Agents](#-autonomous-web-exploration--research-agents)
- [📊 Benchmarks](#-benchmarks)
  - [⚙️ Core Mechanisms of Agentic Reasoning](#-core-mechanisms-of-agentic-reasoning)
    - [Tool Use](#tool-use)
    - [Memory](#memory-1)
    - [Multi-Agent System](#multi-agent-system)
  - [🎯 Applications of Agentic Reasoning](#-applications-of-agentic-reasoning)
    - [Embodied Agents](#embodied-agents-1)
    - [Scientific Discovery Agents](#scientific-discovery-agents-1)
    - [Autonomous Research Agents](#autonomous-research-agents)
    - [Medical and Clinical Agents](#medical-and-clinical-agents)
    - [Web Agents](#web-agents)
    - [General Tool-Use Agents](#general-tool-use-agents)

---

## 🌟 Introduction

Bridging thought and action through autonomous agents that reason, act, and learn via continual interaction with their environments. The goal is to enhance agent capabilities by grounding reasoning in action.

We organize agentic reasoning into three layers, each corresponding to a distinct reasoning paradigm under different *environmental dynamics*:

🔹 **Foundational Reasoning.** Core single-agent abilities (planning, tool-use, search) in stable environments

🔹 **Self-Evolving Reasoning.** Adaptation through feedback, memory, and learning in dynamic settings

🔹 **Collective Reasoning.** Multi-agent coordination, role specialization, and collaborative intelligence

Across these layers, we further identify complementary reasoning paradigms defined by their *optimization settings*.

🔸 **In-Context Reasoning.** Test-time scaling through structured orchestration and adaptive workflows

🔸 **Post-Training Reasoning.** Behavior optimization via RL and supervised fine-tuning


## 🤝 Contributing
This collection is an ongoing effort. We are actively expanding and refining its coverage, and welcome contributions from the community. You can:

- Submit a pull request to add papers or resources
- Open an issue to suggest additional papers or resources
- Email us at twei10@illinois.edu

We regularly update the repository to include new research.


## 📝 Citation

If you find this repository or paper useful, please consider citing the survey paper:

```bibtex
@article{wei2026agent,
  title={Agentic Reasoning for Large Language Models},
  author={Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He},
  journal={arXiv preprint},
  year={2026}
}
```


---

## 🏗️ Foundational Agentic Reasoning

### 🗺️ Planning Reasoning

![plan](figs/planning.png)


#### In-context Planning

##### Workflow Design

| Paper | Venue |
| --- | --- |
| [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477) | ArXiv 2023 |
| [PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change](https://arxiv.org/abs/2206.10498) | NeurIPS 2023 DB Track |
| [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323) | ArXiv 2023 |
| [LLM-Reasoners: New Evaluation Approaches for Large Language Models](https://arxiv.org/abs/2404.05221) | ArXiv 2024 |
| [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) | ICLR 2023 |
| [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091) | ACL 2023 |
| [Algorithm of Thoughts: Enhancing LLM Reasoning Capabilities via Algorithmic Reasoning](https://arxiv.org/abs/2308.10379) | ICML 2024 |
| [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/abs/2303.17580) | ArXiv 2023 |
| [Plan, Eliminate, and Track -- Language Models are Good Teachers for Embodied Agents](https://arxiv.org/abs/2305.02412) | ArXiv 2023 |
| [PERIA: A Unified Multimodal Workflow](https://arxiv.org/abs/2511.14210) | ArXiv 2024 |
| [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) | ArXiv 2025 |
| [CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499) | FSE 2024 |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070) | NeurIPS 2023 |
| [Wilbur: Adaptive In-Context Learning for Robust and Accurate Web Agents](https://arxiv.org/abs/2404.05902) | ArXiv 2024 |
| [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.10312) | ICML 2024 |
| [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) | ArXiv 2023 |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | ArXiv 2023 |
| [CodeNav: Beyond Tool-Use to Using Real-World Codebases with LLM Agents](https://arxiv.org/abs/2402.13463) | ACL 2024 |
| [MARCO: A Multi-Agent System for Optimizing HPC Code Generation](https://arxiv.org/abs/2505.03906) | ArXiv 2025 |
| [Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection Agents](https://arxiv.org/abs/2501.00430) | ArXiv 2025 |
| [Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents](https://arxiv.org/abs/2505.09970) | ArXiv 2025 |
| [REST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003) | ArXiv 2023 |
| [Self-Planning Code Generation with Large Language Models](https://arxiv.org/abs/2303.06689) | TOSEM 2024 |
| [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/abs/2207.04429) | CoRL 2022 |

##### Tree Search / Algorithm Simulation

| Paper | Venue |
| --- | --- |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | NeurIPS 2023 |
| [Tree Search for Language Model Agents](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Tree-Planner: Efficient Planning with Large Language Models](https://arxiv.org/abs/2310.08582) | ICLR 2024 |
| [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283) | ArXiv 2024 |
| [LLM-A*: Large Language Model Guided A* Search](https://arxiv.org/html/2407.02511v1) | ArXiv 2024 |
| [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) | ArXiv 2024 |
| [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992) | NeurIPS 2023 |
| [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199) | ArXiv 2024 |
| [Monte Carlo Tree Search with Large Language Models](https://arxiv.org/abs/2405.00451) | ArXiv 2023 |
| [Prompt-Based Monte-Carlo Tree Search for Goal-Oriented Dialogue](https://arxiv.org/abs/2305.13660) | ArXiv 2023 |
| [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126) | ICLR 2024 |
| [Everything of Thoughts: Defying the Laws of Pen and Paper](https://arxiv.org/abs/2311.04254) | ArXiv 2023 |
| [Tree-of-Thought Prompting](https://www.google.com/search?q=https://arxiv.org/abs/2305.00000) | ArXiv 2024 |
| [AlphaZero-Like Tree Search for LLM Reasoning](https://arxiv.org/abs/2309.17179) | ArXiv 2023 |
| [Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs](https://arxiv.org/abs/2503.11586) | ArXiv 2025 |
| [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633) | NeurIPS 2023 |
| [Pathfinder: Guided Search over Multi-Step Reasoning Paths](https://arxiv.org/abs/2312.05180) | ArXiv 2023 |
| [Discriminator-Guided Embodied Planning for LLM Agent](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D...) | ICLR 2025 |
| [Stream of Search (SoS): Learning to Search in Language](https://arxiv.org/abs/2404.03683) | ArXiv 2024 |
| [System-1.x: Learning to Balance Fast and Slow Planning](https://arxiv.org/abs/2407.14414) | ArXiv 2024 |
| [Agent-E: From Autonomous Web Navigation to Foundational Design](https://arxiv.org/abs/2407.13032) | ArXiv 2024 |
| [Intelligent Virtual Assistants with LLM-based Process Automation](https://arxiv.org/abs/2312.06677) | ArXiv 2023 |
| [Agent S: An Open Agentic Framework that Uses Computers Like a Human](https://arxiv.org/abs/2410.08164) | ArXiv 2024 |
| [HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](https://arxiv.org/abs/2505.02322) | ArXiv 2025 |
| [Tree-of-Code: A Self-Growing Tree Framework for Code Generation](https://arxiv.org/abs/2412.15305) | ACL 2025 |
| [Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution](https://arxiv.org/abs/2504.16563) | ArXiv 2025 |
| [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents](https://arxiv.org/abs/2505.19761) | ArXiv 2025 |
| [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search](https://arxiv.org/abs/2410.20285) | ICLR 2025 |
| [BTGenBot: Behavior Tree Generation for Robot Control](https://arxiv.org/abs/2403.12761) | ArXiv 2024 |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608) | CoRL 2022 |

##### Process Formalization

| Paper | Venue |
| --- | --- |
| [Leveraging Pre-trained Large Language Models to Construct World Models](https://arxiv.org/abs/2305.14909) | NeurIPS 2023 |
| [Leveraging Environment Interaction for Automated PDDL Translation](https://arxiv.org/abs/2407.12979) | NeurIPS 2024 |
| [Thought of Search: Planning with Language Models](https://arxiv.org/abs/2404.11833) | NeurIPS 2024 |
| [CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499) | FSE 2024 |
| [Planning Anything with Rigor: General-Purpose Zero-Shot Planning](https://arxiv.org/abs/2410.12112) | ArXiv 2024 |
| [From an LLM Swarm to a PDDL-empowered Hive](https://arxiv.org/abs/2412.12839) | ArXiv 2024 |

##### Decoupling / Decomposition

| Paper | Venue |
| --- | --- |
| [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323) | NeurIPS 2023 |
| [DiffuserLite: Towards Real-time Diffusion Planning](https://arxiv.org/abs/2401.15443) | ArXiv 2024 |
| [Goal-Space Planning with Subgoal Models](https://www.jmlr.org/papers/volume25/24-0040/24-0040.pdf) | JMLR 2024 |
| [Agent-Oriented Planning in Multi-Agent Systems](https://arxiv.org/abs/2410.02189) | ArXiv 2024 |
| [GoPlan: Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/abs/2310.20025) | ArXiv 2023 |
| [RetroInText: A Multimodal LLM Framework for Retrosynthetic Planning](https://openreview.net/forum?id=J6e4hurEKd) | ICLR 2025 |
| [HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](https://arxiv.org/abs/2505.02322) | ArXiv 2025 |
| [VisualPredicator: Learning Abstract World Models](https://arxiv.org/abs/2410.23156) | ArXiv 2024 |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/abs/2410.14157) | ArXiv 2024 |
| [PlanAgent: A Multi-modal Large Language Agent for Vehicle Motion Planning](https://arxiv.org/abs/2406.01587) | ArXiv 2024 |
| [Long-Horizon Planning for Multi-Agent Robots](https://arxiv.org/abs/2407.10031) | ArXiv 2024 |

##### External Aid / Tool Use

| Paper | Venue |
| --- | --- |
| [Plan-on-Graph: Self-Correcting Adaptive Planning on Knowledge Graphs](https://arxiv.org/abs/2410.23875) | NeurIPS 2024 |
| [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG](https://arxiv.org/abs/2504.04578) | ArXiv 2025 |
| [TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching](https://arxiv.org/abs/2505.00562) | ArXiv 2025 |
| [FlexPlanner: Flexible 3D Floorplanning via Deep Reinforcement Learning in Hybrid Action Space with Multi-Modality Representation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/585e9cf25585612ac27b535457116513-Abstract-Conference.html) | NeurIPS 2024 |
| [Exploratory Retrieval-Augmented Planning](https://neurips.cc/) | NeurIPS 2024 |
| [Benchmarking Multimodal RAG with Dynamic VQA](https://arxiv.org/abs/2411.02937) | ArXiv 2024 |
| [RAG over Tables: Hierarchical Memory Index](https://arxiv.org/abs/2504.01346) | 2025 |
| [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992) | NeurIPS 2023 |
| [Leveraging Pre-trained LLMs to Construct World Models](https://arxiv.org/abs/2305.14909) | NeurIPS 2023 |
| [Agent Planning with World Knowledge Model](https://arxiv.org/abs/2405.14205) | NeurIPS 2024 |
| [BehaviorGPT: Smart Agent Simulation for Autonomous Driving](https://neurips.cc/) | NeurIPS 2024 |
| [Dino-WM: World Models on Pre-trained Visual Features](https://arxiv.org/abs/2411.04983) | ArXiv 2024 |
| [FLIP: Flow-Centric Generative Planning](https://arxiv.org/abs/2412.08261) | ArXiv 2024 |
| [Continual Reinforcement Learning by Planning with Online World Models](https://arxiv.org/abs/2507.09177) | ArXiv 2025 |
| [AdaWM: Adaptive World Model Based Planning](https://arxiv.org/abs/2501.13072) | ArXiv 2025 |
| [HuggingGPT: Solving AI Tasks with ChatGPT](https://arxiv.org/abs/2303.17580) | ArXiv 2023 |
| [Tool-Planner: Task Planning with Clusters](https://arxiv.org/abs/2406.03807) | ArXiv 2024 |
| [RetroInText: A Multimodal LLM Framework for Retrosynthetic Planning](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3Db2fbf1c9bc) | ICLR 2025 |

#### Post-training Planning

| Paper | Venue |
| --- | --- |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Reflect-then-Plan: Offline Model-Based Planning](https://arxiv.org/abs/2506.06261) | ArXiv 2025 |
| [Rational Decision-Making Agent with Internalized Utility Judgment](https://arxiv.org/abs/2308.12519) | ArXiv 2023 |
| [Scaling Autonomous Agents via Automatic Reward Modeling](https://arxiv.org/abs/2502.12130) | ArXiv 2025 |
| [Strategic Planning: A Top-Down Approach to Option Generation](https://openreview.net/forum?id=xkgQWEj9F2&noteId=mt0BbGT077) | ICML 2025 |
| [Non-Myopic Generation of Language Models for Reasoning](https://arxiv.org/abs/2410.17195) | ArXiv 2024 |
| [Physics-Informed Temporal Difference Metric Learning](https://arxiv.org/abs/2505.05691) | ArXiv 2025 |
| [Generalizable Motion Planning via Operator Learning](https://arxiv.org/abs/2410.17547) | ArXiv 2024 |
| [ToolOrchestra: Elevating Intelligence via Efficient Model](https://arxiv.org/abs/2511.21689) | ArXiv 2025 |
| [Latent Diffusion Planning for Imitation Learning](https://arxiv.org/abs/2504.16925) | ArXiv 2025 |
| [SafeDiffuser: Safe Planning with Diffusion Probabilistic Models](https://openreview.net/forum?id=ig2wk7kK9J) | ICLR 2023 |
| [ContraDiff: Planning Towards High Return States](https://openreview.net/forum?id=XMOaOigOQo) | ICLR 2025 |
| [Amortized Planning with Large-Scale Transformers](https://arxiv.org/abs/2402.04494) | NeurIPS 2024 |
| [GoPlan: Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/abs/2310.20025) | ArXiv 2023 |


### 🛠️ Tool-Use Optimization

![tool](figs/tool_use.png)


#### In-Context Tool-Integration


##### Interleaving Reasoning and Tool Use

| Paper | Venue |
| --- | --- |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | NeurIPS 2022 |
| [ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models](https://aclanthology.org/2023.findings-emnlp.985/) | EMNLP 2023 |
| [MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting](https://aclanthology.org/2023.acl-short.130/) | ACL 2023 |
| [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [ART: Automatic Multi-step Reasoning and Tool-use for Large Language Models](https://arxiv.org/abs/2303.09014) | ArXiv 2023 |

##### Optimizing Context for Tool Interaction

| Paper | Venue |
| --- | --- |
| [Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models](https://arxiv.org/abs/2308.00675) | ArXiv 2023 |
| [EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction](https://arxiv.org/abs/2401.06201) | NAACL 2025 |
| [GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution](https://aclanthology.org/2024.eacl-long.7/) | EACL 2024 |
| [AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2db8ce969b000fe0b3fb172490c33ce8-Abstract-Conference.html) | NeurIPS 2024 |


#### Post-training Tool-Integration


##### Bootstrapping of Tool Use via SFT

| Paper | Venue |
| --- | --- |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | NeurIPS 2023 |
| [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) | ICLR 2024 |
| [ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301) | ArXiv 2023 |
| [Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) | NeurIPS 2023 |
| [RestGPT: Connecting Large Language Models with Real-World RESTful APIs](https://arxiv.org/abs/2306.06624) | ArXiv 2023 |
| [ADAPT: As-Needed Decomposition and Planning with Language Models](https://arxiv.org/abs/2311.05772) | ArXiv 2023 |
| [Agent Lumos: Unified and Modular Training for Open-Source Language Agents](https://arxiv.org/abs/2311.05657) | ArXiv 2023 |
| [Learning to Use Tools via Cooperative and Interactive Agents](https://arxiv.org/abs/2403.03031) | ArXiv 2024 |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452) | ArXiv 2023 |
| [Preserving Diversity in Supervised Fine-Tuning of Large Language Models](https://arxiv.org/abs/2408.16673) | ArXiv 2024 |
| [Attributing Mode Collapse in the Fine-Tuning of Large Language Models](https://arxiv.org/abs/2410.05559) | ICLR Workshop 2024 |
| [Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](https://arxiv.org/abs/2505.16270) | ArXiv 2025 |
| [Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning](https://arxiv.org/html/2501.09766v1) | ArXiv 2025 |
| [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) | ArXiv 2025 |
| [Demystifying Reinforcement Learning in Agentic Reasoning](https://arxiv.org/abs/2510.11701) | ArXiv 2025 |

##### Mastery of Tool Use via RL

| Paper | Venue |
| --- | --- |
| [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449) | ArXiv 2025 |
| [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search](https://arxiv.org/abs/2410.20285) | ArXiv 2024 |
| [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844) | ArXiv 2025 |
| [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) | ArXiv 2025 |
| [AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning](https://arxiv.org/abs/2512.13278) | ArXiv 2025 |
| [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007) | ArXiv 2025 |
| [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) | ArXiv 2025 |
| [ZeroSearch: Incentivize the Search Capability of LLMs Without Searching](https://arxiv.org/abs/2505.04588) | ArXiv 2025 |
| [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) | ArXiv 2025 |
| [Gemini 2.5: Pushing the Frontier with Advanced Reasoning and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261) | ArXiv 2025 |
| [Kimi k2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) | ArXiv 2025 |
| [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) | ArXiv 2025 |
| [TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning](https://arxiv.org/abs/2510.06217) | ArXiv 2025 |

#### Orchestration-based Tool-Integration

##### Agentic Pipelines for Tool Orchestration

| Paper | Venue |
| --- | --- |
| [ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback](https://arxiv.org/abs/2409.14826) | ArXiv 2025 |
| [Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning](https://arxiv.org/abs/2506.04625) | KDD 2025 |
| [OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](https://arxiv.org/abs/2502.11271) | ArXiv 2025 |
| [Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models](https://arxiv.org/abs/2503.16779) | ArXiv 2025 |
| [PyVision: Agentic Vision with Dynamic Tooling](https://arxiv.org/abs/2507.07998) | ArXiv 2025 |
| [Learning to Use Tools via Cooperative and Interactive Agents](https://arxiv.org/abs/2403.03031) | ArXiv 2024 |
| [El Agente: An Autonomous Agent for Quantum Chemistry](https://arxiv.org/abs/2505.02484) | ArXiv 2025 |

##### Tool Representations for Orchestration

| Paper | Venue |
| --- | --- |
| [ToolExpNet: Optimizing Multi-Tool Selection in LLMs with Similarity and Dependency-Aware Experience Networks](https://aclanthology.org/2025.findings-acl.811/) | ACL (Findings) 2025 |
| [T^2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](https://arxiv.org/abs/2505.19768) | ArXiv 2025 |
| [ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search](https://arxiv.org/abs/2310.13227) | ArXiv 2023 |
| [ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval](https://aclanthology.org/2024.lrec-main.1413/) | COLING 2024 |

### 🔍 Agentic Search

![search](figs/search.png)


#### In-Context Search

##### Interleaving Reasoning and Search

| Paper | Venue |
| --- | --- |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350) | ArXiv 2022 |
| [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) | ArXiv 2022 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | NeurIPS Workshop 2023 |
| [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-Adaptive Planning Agent](https://arxiv.org/abs/2411.02937) | ArXiv 2024 |
| [DeepRAG: Thinking to Retrieve Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142) | ArXiv 2025 |
| [MC-Search: Benchmarking Multimodal Agentic RAG with Structured Reasoning Chains](https://openreview.net/forum?id=S2zaYgT7Ic) | NeurIPS Workshop 2025 |

##### Structure-Enhanced Search

| Paper | Venue |
| --- | --- |
| [Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation](https://openreview.net/forum?id=g2C947jjjQ) | 2025 |
| [MC-Search: Benchmarking Multimodal Agentic RAG with Structured Reasoning Chains](https://openreview.net/forum?id=S2zaYgT7Ic) | NeurIPS Workshop 2025 |
| [GeAR: Graph-Enhanced Agent for Retrieval-Augmented Generation](https://arxiv.org/abs/2412.18431) | ArXiv 2024 |
| [Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection](https://arxiv.org/abs/2502.14932) | ArXiv 2025 |

#### Post-Training Search


##### SFT-Based Agentic Search

| Paper | Venue |
| --- | --- |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf) | NeurIPS 2023 |
| [INTERS: Unlocking the power of large language models in search with instruction tuning](https://arxiv.org/abs/2401.06532) | ArXiv 2024 |
| [RAG-Studio: Towards In-Domain Adaptation of Retrieval Augmented Generation through Self-Alignment](https://aclanthology.org/2024.findings-emnlp.41/) | EMNLP (Findings) 2024 |
| [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) | ArXiv 2024 |
| [Search-o1: Agentic search-enhanced large reasoning models](https://arxiv.org/abs/2501.05366) | ArXiv 2025 |
| [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352) | ICLR 2023 |
| [SFR-RAG: Towards Contextually Faithful LLMs](https://arxiv.org/abs/2409.09916) | ArXiv 2024 |

##### RL-Based Agentic Search

| Paper | Venue |
| --- | --- |
| [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) | ArXiv 2021 |
| [RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning](https://arxiv.org/abs/2503.12759) | ArXiv 2025 |
| [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-World Environments](https://arxiv.org/abs/2504.03160) | ArXiv 2025 |
| [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) | ArXiv 2025 |
| [ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding](https://arxiv.org/abs/2501.07861) | ArXiv 2025 |

---

## 🧬 Self-evolving Agentic Reasoning

### 🔄 Agentic Feedback Mechanisms

![feed](figs/feedback.png)


#### Reflective Feedback

| Paper | Venue |
| --- | --- |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [Enable Language Models to Implicitly Learn Self-Improvement From Data](https://arxiv.org/abs/2310.00898) | ICLR 2024 |
| [A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve](https://arxiv.org/abs/2507.21046) | TMLR 2025 |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | NeurIPS 2023 |
| [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) | AAAI 2024 |
| [Zero-Shot Verification-Guided Chain of Thoughts](https://arxiv.org/abs/2501.13122) | ArXiv 2025 |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [WebGPT: Browser-assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332) | ArXiv 2021 |
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | ArXiv 2023 |
| [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | ArXiv 2023 |

#### Parametric Adaptation


| Paper | Venue |
| --- | --- |
| [AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://arxiv.org/abs/2310.12823) | ArXiv 2023 |
| [ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003) | ArXiv 2023 |
| [Re-ReST: Reflection-Reinforced Self-Training for Language Agents](https://arxiv.org/abs/2406.01495) | ArXiv 2024 |
| [Distilling Step-by-Step: Outperforming Larger LMs with Less Data](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 |
| [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) | NeurIPS 2017 |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | NeurIPS 2023 |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | ArXiv 2022 |
| [ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection](https://aclanthology.org/2025.findings-acl.871/) | ACL (Findings) 2025 |

#### Validator-Driven Feedback

| Paper | Venue |
| --- | --- |
| [ReZero: Enhancing LLM search ability by trying one-more-time](https://arxiv.org/abs/2504.11001) | ArXiv 2025 |
| [Are Retrials All You Need? Enhancing Large Language Model Reasoning Without Verbalized Feedback](https://arxiv.org/abs/2504.12951) | ArXiv 2025 |
| [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780) | ArXiv 2022 |
| [LEVER: Learning to Verify Language-to-Code Generation with Execution](https://arxiv.org/abs/2302.08468) | ICML 2023 |
| [SWE-bench: Can Language Models Resolve Real-world Github Issues?](https://arxiv.org/abs/2310.06770) | ICLR 2024 |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | ICML 2023 |
| [Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.24726) | ArXiv 2025 |

### 🧠 Agentic Memory

![mem](figs/memory.png)


#### Agentic Use of Memory


##### Conversational Memory and Factual Memory

| Paper | Venue |
| --- | --- |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | NeurIPS 2020 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8) | ICLR 2024 |
| [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) | ArXiv 2023 |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Software 2022 |
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | ArXiv 2023 |
| [RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/abs/2305.14322) | ArXiv 2023 |
| [SCM: Enhancing Large Language Model with Self-Controlled Memory Framework](https://arxiv.org/abs/2304.13343) | ArXiv 2023 |
| [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) | ArXiv 2024 |
| [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) | ArXiv 2024 |
| [SELFGOAL: Your Language Agents Already Know How to Achieve High-level Goals](https://aclanthology.org/2025.naacl-long.36/) | NAACL 2025 |
| [PlanAgent: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning](https://arxiv.org/abs/2406.01587) | ArXiv 2024 |
| [Large Language Models Can Self-Improve At Web Agent Tasks](https://arxiv.org/abs/2405.20309) | ArXiv 2024 |
| [Reflective Multi-Agent Collaboration based on Large Language Models](https://api.semanticscholar.org/CorpusID:276318441) | NeurIPS 2024 |
| [FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design](https://doi.org/10.1609/aaaiss.v3i1.31290) | AAAI Spring Symposium 2024 |
| [A-mem: Agentic memory for llm agents](https://arxiv.org/abs/2502.12110) | ArXiv 2025 |

##### Reasoning Memory and Experience Reuse

| Paper | Venue |
| --- | --- |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |
| [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171) | ArXiv 2025 |
| [Dynamic Cheatsheet: Test-time learning with adaptive memory](https://arxiv.org/abs/2504.07952) | ArXiv 2025 |
| [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) | ArXiv 2025 |
| [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140) | ArXiv 2025 |
| [Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory](https://arxiv.org/abs/2511.20857) | ArXiv 2025 |

##### Multimodal Extensions

| Paper | Venue |
| --- | --- |
| [Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory](https://arxiv.org/abs/2508.09736) | ArXiv 2025 |
| [Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations](https://arxiv.org/abs/2510.00496) | ArXiv 2025 |

#### Structured Memory Representations

| Paper | Venue |
| --- | --- |
| [RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph](https://arxiv.org/abs/2410.14684) | ArXiv 2024 |
| [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) | ArXiv 2024 |
| [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) | ArXiv 2025 |
| [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) | ArXiv 2025 |
| [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) | ArXiv 2025 |
| [AutoFlow: Automated Workflow Generation for Large Language Model Agents](https://arxiv.org/abs/2407.12821) | ArXiv 2024 |
| [AFlow: Automating Agentic Workflow Generation](https://openreview.net/forum?id=z5uVAKwmjf) | ICLR 2025 |
| [FlowMind: Automatic Workflow Generation with LLMs](https://arxiv.org/abs/2304.14671) | ICAIF 2023 |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |
| [Dynamic Cheatsheet: Test-time learning with adaptive memory](https://arxiv.org/abs/2504.07952) | ArXiv 2025 |

#### Post-training Memory Control

| Paper | Venue |
| --- | --- |
| [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](https://arxiv.org/abs/2507.02259) | ArXiv 2025 |
| [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) | ArXiv 2025 |
| [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828) | ArXiv 2025 |
| [Mem-alpha: Learning Memory Construction via Reinforcement Learning](https://arxiv.org/abs/2509.25911) | ArXiv 2025 |
| [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635) | ArXiv 2025 |
| [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) | ArXiv 2025 |

### 🚀 Evolving Foundational Agentic Capabilities

![mem](figs/evolve.png)

#### Self-evolving Planning

| Paper | Venue |
| --- | --- |
| [Self-challenging language model agents](https://arxiv.org/abs/2506.01716) | ArXiv 2025 |
| [Self-rewarding language models](https://arxiv.org/abs/2401.10020) | ICML 2024 |
| [Self Rewarding Self Improving](https://arxiv.org/abs/2505.08827) | ArXiv 2025 |
| [Self: Self-evolution with language feedback](https://arxiv.org/abs/2310.00533) | ArXiv 2023 |
| [Training language models to self-correct via reinforcement learning](https://arxiv.org/abs/2409.12917) | ArXiv 2024 |
| [PAG: Policy Alignment with Feedback](https://arxiv.org/abs/2503.00005) | ArXiv 2025 |
| [TextGrad: Differentiable Text Feedback for Language Models](https://arxiv.org/abs/2406.07496) | ArXiv 2024 |
| [AutoRule: Converting Reasoning Traces to Reward Rules](https://arxiv.org/abs/2504.00007) | ArXiv 2025 |
| [AgentGen: Generating Interactive Environments for Agents](https://arxiv.org/abs/2402.11263) | ArXiv 2024 |
| [Reflexion: Language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Adaplanner: Adaptive planning from feedback with language models](https://openreview.net/forum?id=rnKgbKmelt) | NeurIPS 2023 |
| [Self-refine: Iterative refinement with self-feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [A self-improving coding agent](https://arxiv.org/abs/2504.15228) | ArXiv 2025 |
| [Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning](https://arxiv.org/abs/2504.20073) | ArXiv 2025 |
| [DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning](https://arxiv.org/abs/2505.03209) | ArXiv 2025 |

#### Self-evolving Tool-use

| Paper | Venue |
| --- | --- |
| [Large Language Models as Tool Makers](https://openreview.net/forum?id=qV83K9d5WB) | ICLR 2024 |
| [CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets](https://openreview.net/forum?id=G0vdDSt9XM) | ICLR 2024 |
| [CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models](https://aclanthology.org/2023.findings-emnlp.462/) | EMNLP 2023 |
| [LLM Agents Making Agent Tools](https://arxiv.org/abs/2502.11705) | ArXiv 2025 |

#### Self-evolving Search

| Paper | Venue |
| --- | --- |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | NeurIPS 2020 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8) | ICLR 2024 |
| [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) | ArXiv 2023 |
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | ArXiv 2023 |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |
| [Dynamic Cheatsheet: Test-time learning with adaptive memory](https://arxiv.org/abs/2504.07952) | ArXiv 2025 |
| [Reflexion: Language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140) | ArXiv 2025 |
| [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) | ArXiv 2025 |
| [AutoFlow: Automated Workflow Generation for Large Language Model Agents](https://arxiv.org/abs/2407.12821) | ArXiv 2024 |
| [AFlow: Automating Agentic Workflow Generation](https://openreview.net/forum?id=z5uVAKwmjf) | ICLR 2025 |
| [FlowMind: Automatic Workflow Generation with LLMs](https://arxiv.org/abs/2304.14671) | ICAIF 2023 |
| [RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph](https://arxiv.org/abs/2410.14684) | ArXiv 2024 |
| [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) | ArXiv 2024 |
| [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) | ArXiv 2025 |
| [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) | ArXiv 2025 |
| [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) | ArXiv 2025 |
| [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635) | ArXiv 2025 |

---

## 👥 Collective Multi-agent Reasoning

![mem](figs/mas.png)

### 🤝 Collaboration and Division of Labor

![collab](figs/multi-agent-collab.png)


#### In-context Collaboration


##### Manually Crafted Pipelines

| Paper | Venue |
| --- | --- |
| [AgentOrchestra: A Hierarchical Multi-Agent Framework for General-Purpose Task Solving](https://arxiv.org/abs/2506.12508) | ArXiv 2025 |
| [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://openreview.net/forum?id=VtmBAGCN7o) | ICLR 2024 |
| [SurgRAW: Multi-agent workflow with chain-of-thought reasoning for surgical intelligence](https://arxiv.org/abs/2503.10265) | ArXiv 2025 |
| [Collab-RAG: Boosting retrieval-augmented generation for complex question answering via white-box and black-box llm collaboration](https://arxiv.org/abs/2504.04915) | ArXiv 2025 |
| [MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.20096) | ArXiv 2025 |
| [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://openreview.net/forum?id=LuCLf4BJsr) | NeurIPS 2024 |
| [AutoAgents: a framework for automatic agent generation](https://doi.org/10.24963/ijcai.2024/3) | IJCAI 2024 |
| [RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning](https://arxiv.org/abs/2503.13514) | ArXiv 2025 |
| [SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents](https://arxiv.org/abs/2411.03284) | ArXiv 2024 |
| [MDocAgent: A multi-modal multi-agent framework for document understanding](https://arxiv.org/abs/2503.13964) | ArXiv 2025 |

##### LLM-Driven Pipelines

| Paper | Venue |
| --- | --- |
| [AutoML-Agent: A multi-agent llm framework for full-pipeline automl](https://arxiv.org/abs/2410.02958) | ArXiv 2024 |
| [Magentic-One: A generalist multi-agent system for solving complex tasks](https://arxiv.org/abs/2411.04468) | ArXiv 2024 |
| [MAS-GPT: Training LLMs to build LLM-based multi-agent systems](https://arxiv.org/abs/2503.03686) | ArXiv 2025 |
| [MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](https://arxiv.org/abs/2507.22606) | ArXiv 2025 |
| [Agent-oriented planning in multi-agent systems](https://arxiv.org/abs/2410.02189) | ArXiv 2024 |
| [AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering](https://arxiv.org/abs/2510.05445) | ArXiv 2025 |
| [Talk to Right Specialists: Routing and planning in multi-agent system for question answering](https://arxiv.org/abs/2501.07813) | ArXiv 2025 |

##### Theory-of-Mind-Augmented Collaboration

| Paper | Venue |
| --- | --- |
| [Theory of mind for multi-agent collaboration via large language models](https://arxiv.org/abs/2310.10701) | ArXiv 2023 |
| [Hypothetical Minds: Scaffolding theory of mind for multi-agent tasks with large language models](https://arxiv.org/abs/2407.07086) | ArXiv 2024 |
| [MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning](https://arxiv.org/abs/2411.12977) | ArXiv 2024 |
| [How large language models encode theory-of-mind: a study on sparse parameter patterns](https://www.google.com/search?q=https://www.nature.com/articles/s44247-025-00020-x) | npj Artificial Intelligence 2025 |
| [Large Language Models as Theory of Mind Aware Generative Agents with Counterfactual Reflection](https://arxiv.org/abs/2501.15355) | ArXiv 2025 |
| [BeliefNest: A Joint Action Simulator for Embodied Agents with Theory of Mind](https://arxiv.org/abs/2505.12321) | ArXiv 2025 |

#### Post-training Collaboration

##### Multi-agent Prompt Optimization

| Paper | Venue |
| --- | --- |
| [AutoAgents: A Framework for Automatic Agent Generation](https://arxiv.org/abs/2309.17288) | IJCAI 2024 |
| [Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration](https://aclanthology.org/2024.naacl-long.15/) | NAACL 2024 |
| [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382) | ArXiv 2023 |
| [Multi-agent Design: Optimizing Agents with Better Prompts and Topologies](https://arxiv.org/abs/2502.02533) | ArXiv 2025 |
| [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) | ArXiv 2023 |

##### Graph-based Topology Generation

| Paper | Venue |
| --- | --- |
| [Learning Multi-Agent Communication from Graph Modeling Perspective](https://arxiv.org/abs/2405.08550) | ArXiv 2024 |
| [G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/abs/2410.11782) | ArXiv 2024 |
| [Graph Diffusion for Robust Multi-Agent Coordination](https://openreview.net/forum?id=T5IZ32ImAB) | ICML 2025 |
| [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2410.02506) | ArXiv 2024 |
| [Adaptive Graph Pruning for Multi-Agent Communication](https://arxiv.org/abs/2506.02951) | ArXiv 2025 |
| [G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-Agent Systems](https://arxiv.org/abs/2502.11127) | ArXiv 2025 |
| [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762) | ICLR 2025 |
| [Multi-agent Design: Optimizing Agents with Better Prompts and Topologies](https://arxiv.org/abs/2502.02533) | ArXiv 2025 |
| [Multi-Agent Architecture Search via Agentic Supernet](https://arxiv.org/abs/2502.04180) | ArXiv 2025 |
| [DynaSwarm: Dynamically Graph Structure Selection for LLM-based Multi-Agent System](https://arxiv.org/abs/2507.23261) | ArXiv 2025 |
| [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | ICML 2024 |

##### Policy-based Topology Generation

| Paper | Venue |
| --- | --- |
| [MASRouter: Learning to Route LLMs for Multi-Agent Systems](https://arxiv.org/abs/2502.11133) | ArXiv 2025 |
| [RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory](https://arxiv.org/abs/2508.04903) | ArXiv 2025 |
| [xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning](https://arxiv.org/abs/2510.08439) | ArXiv 2025 |
| [Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration](https://arxiv.org/abs/2511.02200) | ArXiv 2025 |
| [LLM Collaboration with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652) | ArXiv 2025 |
| [Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2506.02718) | ArXiv 2025 |
| [Enhancing Multi-Agent Systems via Reinforcement Learning with LLM-based Planner and Graph-based Policy](https://arxiv.org/abs/2503.10049) | ArXiv 2025 |
| [LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation](https://arxiv.org/abs/2506.01538) | IEEE RA-L 2025 |
| [MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://arxiv.org/abs/2502.18439) | ArXiv 2025 |
| [Reflective Multi-Agent Collaboration Based on Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html) | NeurIPS 2024 |
| [Sirius: Self-Improving Multi-Agent Systems via Bootstrapped Reasoning](https://arxiv.org/abs/2502.04780) | ArXiv 2025 |
| [Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](https://arxiv.org/abs/2501.05707) | ArXiv 2025 |
| [M3HF: Multi-Agent Reinforcement Learning from Multi-Phase Human Feedback of Mixed Quality](https://arxiv.org/abs/2503.02077) | ArXiv 2025 |
| [O-MAPL: Offline Multi-Agent Preference Learning](https://arxiv.org/abs/2501.18944) | ArXiv 2025 |

### 🌱 Multi-Agent Evolution

![mem](figs/multi-agent-memory.png)

#### From Single-Agent Evolution to Multi-Agent Evolution

##### Intra-test-time Evolution

| Paper | Venue |
| --- | --- |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [AdaPlanner: Adaptive Planning from Feedback with Language Models](https://arxiv.org/abs/2305.16653) | NeurIPS 2023 |
| [TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution](https://arxiv.org/abs/2402.01586) | TiFA 2024 |
| [Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) | ArXiv 2025 |
| [TTRL: Test-Time Reinforcement Learning](https://arxiv.org/abs/2504.16084) | ArXiv 2025 |
| [Ladder: Self-Improving LLMs through Recursive Problem Decomposition](https://arxiv.org/abs/2503.00735) | ArXiv 2025 |

##### Inter-test-time Evolution

| Paper | Venue |
| --- | --- |
| [Self: Self-Evolution with Language Feedback](https://arxiv.org/abs/2310.00533) | ArXiv 2023 |
| [STaR: Bootstrapping Reasoning with Reasoning](https://arxiv.org/abs/2203.14465) | NeurIPS 2022 |
| [Reasoning Beyond Limits: Advances and Open Problems for LLMs](https://arxiv.org/abs/2503.22732) | ArXiv 2025 |
| [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073) | ArXiv 2025 |
| [DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning](https://arxiv.org/abs/2505.03209) | ArXiv 2025 |
| [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/abs/2411.02337) | ArXiv 2024 |
| [Why do animals need shaping? A theory of task composition and curriculum learning](https://arxiv.org/abs/2402.18361) | ArXiv 2024 |
| [SAGE: Self-evolving Agents with Reflective and Memory-augmented Abilities](https://dl.acm.org/doi/10.1016/j.neucom.2025.130470) | Neurocomputing 2025 |
| [MemInsight: Autonomous Memory Augmentation for LLM Agents](https://arxiv.org/abs/2503.21760) | ArXiv 2025 |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |

##### Multi-agent Evolution

| Paper | Venue |
| --- | --- |
| [Self: Self-Evolution with Language Feedback](https://arxiv.org/abs/2310.00533) | ArXiv 2023 |
| [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917) | ArXiv 2024 |
| [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) | ArXiv 2024 |
| [REMA: Learning to Meta-Think for LLMs with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2503.09501) | ArXiv 2025 |
| [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) | ArXiv 2025 |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |
| [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) | ArXiv 2025 |
| [Multi-agent Design: Optimizing Agents with Better Prompts and Topologies](https://arxiv.org/abs/2502.02533) | ArXiv 2025 |
| [AFlow: Automating Agentic Workflow Generation](https://openreview.net/forum?id=z5uVAKwmjf) | ICLR 2025 |
| [Testing Advanced Driver Assistance Systems Using Multi-Objective Search and Neural Networks](https://dl.acm.org/doi/10.1145/2970276.2970311) | ASE 2016 |
| [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) | ArXiv 2025 |

#### Multi-agent Memory Management for Evolution

| Paper | Venue |
| --- | --- |
| [G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://arxiv.org/abs/2506.07398) | ArXiv 2025 |
| [Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory](https://arxiv.org/abs/2508.08997) | ArXiv 2025 |
| [LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://arxiv.org/abs/2502.05453) | ArXiv 2025 |
| [SEDM: Scalable Self-Evolving Distributed Memory for Agents](https://arxiv.org/abs/2509.09498) | ArXiv 2025 |
| [Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control](https://arxiv.org/abs/2505.18279) | ArXiv 2025 |
| [Memory Sharing for Large Language Model based Agents](https://arxiv.org/abs/2404.09982) | ArXiv 2024 |
| [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957) | ArXiv 2025 |
| [LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation](https://arxiv.org/abs/2510.04851) | ArXiv 2025 |
| [MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning](https://aclanthology.org/2025.alta-main.10/) | ALTA 2025 |
| [Lyfe Agents: Generative agents for low-cost real-time social interactions](https://arxiv.org/abs/2310.02172) | ArXiv 2023 |
| [Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://arxiv.org/abs/2507.06229) | ArXiv 2025 |

#### Training Multi-agent to Evolve

| Paper | Venue |
| --- | --- |
| [Multi-Agent Evolve: LLM Self-Improve through Co-evolution](https://arxiv.org/abs/2510.23595) | ArXiv 2025 |
| [CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards](https://arxiv.org/abs/2510.08529) | ArXiv 2025 |
| [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129) | ArXiv 2025 |
| [Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs](https://arxiv.org/abs/2510.11062) | ArXiv 2025 |
| [MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://arxiv.org/abs/2502.18439) | ArXiv 2025 |
| [MALT: Multi-Agent Learning from Trajectories](https://arxiv.org/abs/2412.01928) | ArXiv 2025 |
| [MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2510.04935) | ArXiv 2025 |
| [Preference-Based Multi-Agent Reinforcement Learning: Data Coverage and Algorithmic Techniques](https://arxiv.org/abs/2409.00717) | ArXiv 2024 |
| [The Alignment Waltz: Jointly Training Agents to Collaborate for Safety](https://arxiv.org/abs/2510.08240) | ArXiv 2025 |

---

## 🎨 Applications

![app](figs/application.png)


### 💻 Math Exploration & Vibe Coding Agents

#### Foundational Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [Advancing mathematics by guiding human intuition with AI](https://www.nature.com/articles/s41586-021-04086-x) | Nature 2021 |
| [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5) | Nature 2024 |
| [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6) | Nature 2024 |
| [Mathematical Exploration and Discovery at Scale](https://arxiv.org/abs/2511.02864) | ArXiv 2025 |
| [Advancing geometry with AI: Multi-agent generation of polytopes](https://arxiv.org/abs/2502.05199) | ArXiv 2025 |
| [Towards Robust Mathematical Reasoning](https://aclanthology.org/2025.emnlp-main.1794/) | EMNLP 2025 |
| [CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules](https://arxiv.org/abs/2310.08992) | ICLR 2024 |
| [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030) | ICML 2024 |
| [Knowledge-Aware Code Generation with Large Language Models](https://arxiv.org/abs/2401.15940) | ICPC 2024 |
| [CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499) | FSE 2024 |
| [Multi-stage guided code generation for Large Language Models](https://doi.org/10.1016/j.engappai.2024.109491) | Eng. App. AI 2025 |
| [CodeTree: Agent-Guided Tree Search for Code Generation with Large Language Models](https://arxiv.org/abs/2411.04329) | ArXiv 2024 |
| [Tree-of-Code: A Self-Growing Tree Framework for End-to-End Code Generation and Execution in Complex Tasks](https://aclanthology.org/2025.findings-acl.509/) | ACL 2025 |
| [DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](https://arxiv.org/abs/2503.14269) | ArXiv 2025 |
| [Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html) | NeurIPS 2024 |
| [VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning](https://arxiv.org/abs/2412.07822) | AAAI 2025 |
| [Guided Search Strategies in Non-Serializable Environments with Applications to Software Engineering Agents](https://arxiv.org/abs/2505.13652) | ICML 2025 |
| [An In-Context Learning Agent for Formal Theorem-Proving](https://arxiv.org/abs/2310.04353) | COLM 2024 |
| [Formal Mathematical Reasoning: A New Frontier in AI](https://arxiv.org/abs/2412.16075) | ArXiv 2024 |
| [Generative Modelling for Mathematical Discovery](https://arxiv.org/abs/2503.11061) | ArXiv 2025 |
| [AI Achieves Silver-Medal Standard Solving International Mathematical Olympiad Problems](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level) | Google DeepMind 2024 |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | NeurIPS 2023 |
| [ToolCoder: Teach Code Generation Models to use API search tools](https://arxiv.org/abs/2305.04032) | ArXiv 2023 |
| [ToolGen: Unified Tool Retrieval and Calling via Generation](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D...) | ICLR 2025 |
| [CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges](https://aclanthology.org/2024.acl-long.739/) | ACL 2024 |
| [ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation](https://arxiv.org/abs/2411.07112) | ICSE 2025 |
| [CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision](https://arxiv.org/abs/2503.20840) | ArXiv 2025 |
| [RepoHyper: Better Context Retrieval is All You Need for Repository-Level Code Completion](https://arxiv.org/abs/2403.06095) | ArXiv 2024 |
| [CodeNav: Beyond Tool-Use to Using Real-World Codebases with LLM Agents](https://arxiv.org/abs/2402.13463) | ICLR 2024 |
| [Optimizing Code Runtime Performance Through Context-Aware Retrieval-Augmented Generation](https://ieeexplore.ieee.org/document/10638538) | ICPC 2025 |
| [Knowledge Graph Based Repository-Level Code Generation](https://conf.researchr.org/details/icse-2025/llm4code-2025-papers/26/Knowledge-Graph-Based-Repository-Level-Code-Generation-Virtual-Talk-) | LLM4Code 2025 |
| [cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree](https://arxiv.org/abs/2506.15655) | ArXiv 2025 |

#### Self-evolving Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [Evaluating Language Models for Mathematics through Interactions](https://www.pnas.org/doi/10.1073/pnas.2318124121) | PNAS 2024 |
| [Self-Edit: Fault-Aware Code Editor for Code Generation](https://aclanthology.org/2023.acl-long.43/) | ACL 2023 |
| [Is Self-Repair a Silver Bullet for Code Generation?](https://arxiv.org/abs/2306.09896) | ArXiv 2024 |
| [LeDeX: Learning to Debug with Execution Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ea832724870c700f0a03c665572e2a9-Paper-Conference.pdf) | NeurIPS 2024 |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [A Self-Iteration Code Generation Method Based on Large Language Models](https://ieeexplore.ieee.org/document/10476069) | ICPADS 2023 |
| [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128) | ICLR 2024 |
| [Self-Collaboration Code Generation via ChatGPT](https://arxiv.org/abs/2304.07590) | TOSEM 2024 |
| [L2MAC: Large Language Model Automatic Computer for Extensive Code Generation](https://arxiv.org/abs/2310.02003) | ArXiv 2023 |
| [Cogito, Ergo Sum: A Neurobiologically-Inspired Cognition-Memory-Growth System for Code Generation](https://arxiv.org/abs/2501.18653) | ArXiv 2025 |

#### Collective Multi-agent Reasoning

| Paper | Venue |
| --- | --- |
| [AgentCoder: Multi-Agent-Based Code Generation with Iterative Testing and Optimisation](https://arxiv.org/abs/2312.13010) | ArXiv 2023 |
| [A Pair Programming Framework for Code Generation via Multi-Plan Exploration and Feedback-Driven Refinement](https://arxiv.org/abs/2409.05001) | ASE 2024 |
| [SOEN-101: Code Generation by Emulating Software Process Models Using Large Language Model Agents](https://ieeexplore.ieee.org/abstract/document/11029771) | ICSE 2025 |
| [Self-Organized Agents: A LLM Multi-Agent Framework Toward Ultra Large-Scale Code Generation](https://arxiv.org/abs/2404.02183) | ArXiv 2024 |
| [MapCoder: Multi-Agent Code Generation for Competitive Problem Solving](https://arxiv.org/abs/2405.11403) | ArXiv 2024 |
| [AutoSafeCoder: A Multi-Agent Framework for Securing LLM Code Generation through Static Analysis and Fuzz Testing](https://arxiv.org/abs/2409.10737) | ArXiv 2024 |
| [QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks](https://arxiv.org/abs/2501.17167) | ArXiv 2025 |
| [SEW: Self-Evolving Agentic Workflows for Automated Code Generation](https://arxiv.org/abs/2505.18646) | ArXiv 2025 |
| [Self-Evolving Multi-Agent Collaboration Networks for Software Development (EvoMAC)](https://arxiv.org/abs/2410.16946) | ArXiv 2024 |
| [Lingma SWE-GPT: An Open Development-Process-Centric Language Model for Automated Software Improvement](https://arxiv.org/abs/2411.00622) | ArXiv 2024 |
| [CodeCoR: An LLM-based Self-Reflective Multi-Agent Framework for Code Generation](https://arxiv.org/abs/2501.07811) | ArXiv 2025 |
| [SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering](https://arxiv.org/abs/2502.06994) | ICML 2025 |
| [Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation (CANDOR)](https://arxiv.org/abs/2506.02943) | ArXiv 2025 |

### 🔬 Scientific Discovery Agents

Here are the extracted citation tables grouped by their respective sections.

#### Foundational Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [ProtAgents: protein discovery via large language model multi-agent collaborations combining physics and machine learning](https://www.google.com/search?q=https://doi.org/10.1039/d4dd00039k) | Digital Discovery 2024 |
| [Agent-based learning of materials datasets from the scientific literature](https://doi.org/10.1039/D4DD00252K) | Digital Discovery 2024 |
| [React: Synergizing reasoning and acting in language models](https://openreview.net/forum?id=WE_vluYUL-X) | ICLR 2023 |
| [Biomni: A General-Purpose Biomedical AI Agent](https://www.biorxiv.org/content/10.1101/2025.05.01.636412v1) | bioRxiv 2025 |
| [Sciagent: Tool-augmented language models for scientific reasoning](https://arxiv.org/abs/2402.11451) | ArXiv 2024 |
| [Chemcrow: Augmenting large-language models with chemistry tools](https://arxiv.org/abs/2304.05376) | ArXiv 2023 |
| [Cactus: Chemistry agent connecting tool usage to science](https://pubs.acs.org/doi/10.1021/acsomega.4c06277) | ACS Omega 2024 |
| [Chemtoolagent: The impact of tools on language agents for chemistry problem solving](https://arxiv.org/abs/2411.07228) | ArXiv 2024 |
| [ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning](https://arxiv.org/abs/2506.07551) | ArXiv 2025 |
| [TxAgent: An AI agent for therapeutic reasoning across a universe of tools](https://arxiv.org/abs/2503.10970) | ArXiv 2025 |
| [Agentmd: Empowering language agents for risk prediction with large-scale clinical tool learning](https://arxiv.org/abs/2402.13225) | Nature Communications 2025 |
| [Paperqa: Retrieval-augmented generative agent for scientific research](https://arxiv.org/abs/2312.07559) | ArXiv 2023 |
| [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740) | ArXiv 2024 |
| [LLaMP: Large language model made powerful for high-fidelity materials knowledge retrieval and distillation](https://arxiv.org/abs/2401.17244) | ArXiv 2024 |
| [Honeycomb: A flexible llm-based agent system for materials science](https://arxiv.org/abs/2409.00135) | ArXiv 2024 |
| [Crispr-gpt: An llm agent for automated design of gene-editing experiments](https://arxiv.org/abs/2404.18021) | ArXiv 2024 |
| [Pharmagents: Building a virtual pharma with large language model agents](https://arxiv.org/abs/2503.22164) | ArXiv 2025 |
| [ORGANA: A robotic assistant for automated chemistry experimentation and characterization](https://doi.org/10.1016/j.matt.2024.12.001) | Matter 2025 |
| [AtomAgents: Alloy design and discovery through physics-aware multi-modal multi-agent artificial intelligence](https://arxiv.org/abs/2407.10022) | ArXiv 2024 |
| [Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation](https://arxiv.org/abs/2311.10776) | ArXiv 2024 |
| [Llm and simulation as bilevel optimizers: A new paradigm to advance physical scientific discovery](https://arxiv.org/abs/2405.09783) | ArXiv 2024 |
| [CellAgent: LLM-Driven Multi-Agent Framework for Natural Language-Based Single-Cell Analysis](https://www.biorxiv.org/content/10.1101/2024.05.13.593861v4) | BioRxiv 2024 |
| [Biodiscoveryagent: An ai agent for designing genetic perturbation experiments](https://arxiv.org/abs/2405.17631) | ArXiv 2024 |
| [Drugagent: Explainable drug repurposing agent with large language model-based reasoning](https://arxiv.org/abs/2408.13378) | ArXiv 2024 |
| [Accelerating Scientific Research Through a Multi-LLM Framework](https://arxiv.org/abs/2502.07960) | ArXiv 2025 |
| [The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search](https://arxiv.org/abs/2504.08066) | ArXiv 2025 |
| [Large language models are zero shot hypothesis proposers](https://arxiv.org/abs/2311.05965) | ArXiv 2023 |
| [Paperqa: Retrieval-augmented generative agent for scientific research](https://arxiv.org/abs/2312.07559) | ArXiv 2023 |
| [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740) | ArXiv 2024 |
| [LLaMP: Large language model made powerful for high-fidelity materials knowledge retrieval and distillation](https://arxiv.org/abs/2401.17244) | ArXiv 2024 |
| [React: Synergizing reasoning and acting in language models](https://openreview.net/forum?id=WE_vluYUL-X) | ICLR 2023 |

#### Self-evolving Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [Chemagent: Self-updating library in large language models improves chemical reasoning](https://arxiv.org/abs/2501.06590) | ArXiv 2025 |
| [Accelerated Inorganic Materials Design with Generative AI Agents](https://arxiv.org/abs/2504.00741) | ArXiv 2025 |
| [Llm and simulation as bilevel optimizers: A new paradigm to advance physical scientific discovery](https://arxiv.org/abs/2405.09783) | ArXiv 2024 |
| [ChemReasoner: Heuristic search over a large language model's knowledge space using quantum-chemical feedback](https://arxiv.org/abs/2402.10980) | ArXiv 2024 |
| [Llmatdesign: Autonomous materials discovery with large language models](https://arxiv.org/abs/2406.13163) | ArXiv 2024 |
| [Hypothesis generation for materials discovery and design using goal-driven and constraint-guided llm agents](https://arxiv.org/abs/2501.13299) | ArXiv 2025 |

#### Collective multi-agent reasoning

| Paper | Venue |
| --- | --- |
| [ProtAgents: protein discovery via large language model multi-agent collaborations combining physics and machine learning](https://www.google.com/search?q=https://doi.org/10.1039/d4dd00039k) | Digital Discovery 2024 |
| [PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration](https://arxiv.org/abs/2505.15047) | ArXiv 2025 |
| [AtomAgents: Alloy design and discovery through physics-aware multi-modal multi-agent artificial intelligence](https://arxiv.org/abs/2407.10022) | ArXiv 2024 |
| [CellAgent: LLM-Driven Multi-Agent Framework for Natural Language-Based Single-Cell Analysis](https://www.biorxiv.org/content/10.1101/2024.05.13.593861v4) | BioRxiv 2024 |
| [Accelerating Scientific Research Through a Multi-LLM Framework](https://arxiv.org/abs/2502.07960) | ArXiv 2025 |
| [Toward a team of ai-made scientists for scientific discovery from gene expression data](https://arxiv.org/abs/2402.12391) | ArXiv 2024 |
| [The virtual lab: Ai agents design new sars-cov-2 nanobodies with experimental validation](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1) | bioRxiv 2024 |

### 🤖 Embodied Agents

#### Foundational Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://proceedings.mlr.press/v229/rana23a.html) | CoRL 2023 |
| [EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](https://arxiv.org/abs/2305.15021) | NeurIPS 2023 |
| [Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents](https://www.google.com/search?q=https://arxiv.org/abs/2408.06877) | ECCV 2024 |
| [Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents](https://arxiv.org/abs/2302.01560) | NeurIPS 2023 |
| [Robotic Control via Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693) | CoRL 2024 |
| [Fast ECoT: Fast Embodied Chain-of-Thought for Vision-Language-Action Models](https://arxiv.org/abs/2501.12745) | ArXiv 2025 |
| [Cosmos-Reason1: Physical Commonsense with Multimodal Chain of Thought Reasoning](https://arxiv.org/abs/2501.03062) | ArXiv 2025 |
| [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://arxiv.org/abs/2501.12574) | ArXiv 2025 |
| [Emma-X: An Embodied Multimodal Action Model with Chain of Thought Reasoning](https://arxiv.org/abs/2412.18525) | ArXiv 2024 |
| [Robot-R1: Reinforcement Learning Enhanced Large Vision-Language Models for Robotic Manipulation](https://arxiv.org/abs/2501.19245) | ArXiv 2025 |
| [ManipLVM-R1: Learning to Reason for Robotic Manipulation via Reinforcement Learning](https://arxiv.org/abs/2502.01235) | ArXiv 2025 |
| [Embodied-R: Emergent Spatial Reasoning in Robotics via Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2502.12641) | ArXiv 2025 |
| [VIKI-R: A VLM-Based Reinforcement Learning Approach for Heterogeneous Multi-Agent Cooperation](https://arxiv.org/abs/2502.06450) | ArXiv 2025 |
| [GSCE: A Prompt Framework for Enhanced Logical Reasoning in LLM-Based Drone Control](https://arxiv.org/abs/2501.00940) | ArXiv 2025 |
| [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](https://arxiv.org/abs/2206.08853) | NeurIPS 2022 |
| [Physical AI Agents: Integrating Generative AI, Symbolic AI and Robotics](https://arxiv.org/abs/2501.07720) | ArXiv 2025 |
| [Chat with the Environment: Interactive Multimodal Perception using Large Language Models](https://arxiv.org/abs/2303.08268) | IROS 2023 |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [Leo: An embodied generalist agent in 3d world](https://arxiv.org/abs/2311.12871) | ICML 2024 |
| [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](https://arxiv.org/abs/2501.12749) | ArXiv 2025 |
| [Gemini Robotics: Bringing AI to the Physical World](https://arxiv.org/abs/2502.02898) | ArXiv 2025 |
| [Octopus: Embodied Vision-Language Programmer from Environmental Feedback](https://arxiv.org/abs/2310.08588) | ECCV 2024 |
| [CaPo: Cooperative Plan Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2403.06093) | ArXiv 2024 |
| [Coherent: Collaboration of heterogeneous robots via nature-language-based team reasoning](https://arxiv.org/abs/2405.00693) | ArXiv 2024 |
| [MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception](https://arxiv.org/abs/2312.07472) | CVPR 2024 |
| [LLM-Planner: Few-Shot Grounded High-Level Planning for Embodied Agents with Large Language Models](https://arxiv.org/abs/2212.04088) | ICCV 2023 |
| [EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](https://arxiv.org/abs/2305.15021) | NeurIPS 2023 |
| [L3MVN: Leveraging Large Language Models for Visual Target Navigation](https://arxiv.org/abs/2304.05501) | ArXiv 2023 |
| [SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments](https://arxiv.org/abs/2309.04077) | ICAPS 2023 |
| [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://proceedings.mlr.press/v229/rana23a.html) | CoRL 2023 |
| [ReMEmbR: Building and Reasoning with Long-Horizon Spatio-Temporal Memory for Embodied Agents](https://arxiv.org/abs/2501.07421) | ArXiv 2025 |
| [Embodied-RAG: General Non-parametric Embodied Memory for Retrieval-Augmented Generation](https://arxiv.org/abs/2409.18244) | ArXiv 2024 |
| [EmbodiedRAG: Retrieval-Augmented Generation for Embodied Agents](https://arxiv.org/abs/2410.15082) | ArXiv 2024 |
| [Retrieval-Augmented Embodied Agents](https://arxiv.org/abs/2407.16733) | ArXiv 2024 |
| [MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents](https://arxiv.org/abs/2412.16660) | ArXiv 2024 |

#### Self-evolving Agentic Reasoning

| Paper | Venue |
| --- | --- |
| [LLM-Empowered Embodied Agent with Memory-Augmented Planning](https://arxiv.org/abs/2412.18260) | ArXiv 2024 |
| [Optimus-1: Hybrid Multimodal Memory Empowered Agents for Long-Horizon Tasks in Minecraft](https://www.google.com/search?q=https://arxiv.org/abs/2408.06371) | ArXiv 2024 |
| [Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models](https://arxiv.org/abs/2310.15127) | EMNLP 2023 |
| [Endowing Embodied Agents with Spatial Intelligence by Brain-Inspired Memory](https://arxiv.org/abs/2501.08253) | ArXiv 2025 |
| [Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents](https://arxiv.org/abs/2408.06877) | ArXiv 2024 |
| [ELLA: Embodied Social Agents with Long-Term Multimodal Memory](https://arxiv.org/abs/2502.13119) | ArXiv 2025 |
| [Chat with the Environment: Interactive Multimodal Perception using Large Language Models](https://arxiv.org/abs/2303.08268) | IROS 2023 |
| [Strangers to Assistants: Fast Desire Alignment for Embodied Agents via Minimal Feedback](https://arxiv.org/abs/2502.04690) | ArXiv 2025 |
| [Robots That Ask for Help: Uncertainty Alignment for Large Language Model Planners](https://arxiv.org/abs/2309.01582) | CoRL 2023 |
| [Octopus: Embodied Vision-Language Programmer from Environmental Feedback](https://arxiv.org/abs/2310.08588) | ECCV 2024 |
| [MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning](https://arxiv.org/abs/2411.12977) | ArXiv 2024 |
| [Efficient LLM Grounding for Embodied Multi-Agent Collaboration](https://arxiv.org/abs/2410.16548) | ArXiv 2024 |
| [Optimus-1: Hybrid Multimodal Memory Empowered Agents for Long-Horizon Tasks in Minecraft](https://www.google.com/search?q=https://arxiv.org/abs/2408.06371) | ArXiv 2024 |
| [Self-Correction for Humanoid Robots via LLM-Based Agentic Workflows](https://arxiv.org/abs/2502.04351) | ArXiv 2025 |
| [EMAC+: Enhanced Multimodal Agent for Continuous and Cooperative Tasks](https://arxiv.org/abs/2501.10705) | ArXiv 2025 |
| [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | ArXiv 2023 |

#### Collective multi-agent reasoning

| Paper | Venue |
| --- | --- |
| [Smart-LLM: Smart Multi-Agent Robot Task Planning with Large Language Models](https://arxiv.org/abs/2309.10062) | ArXiv 2024 |
| [CaPo: Cooperative Plan Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2403.06093) | ArXiv 2024 |
| [Coherent: Collaboration of heterogeneous robots via nature-language-based team reasoning](https://arxiv.org/abs/2405.00693) | ArXiv 2024 |
| [Theory of mind for multi-agent collaboration via large language models](https://arxiv.org/abs/2310.10701) | ArXiv 2023 |
| [How large language models encode theory-of-mind: a study on sparse parameter patterns](https://www.nature.com/articles/s44387-025-00031-9) | npj Artificial Intelligence 2025 |
| [Hypothetical Minds: Scaffolding theory of mind for multi-agent tasks with large language models](https://arxiv.org/abs/2407.07086) | ArXiv 2024 |
| [MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning](https://arxiv.org/abs/2411.12977) | ArXiv 2024 |
| [EMAC+: Enhanced Multimodal Agent for Continuous and Cooperative Tasks](https://arxiv.org/abs/2501.10705) | ArXiv 2025 |
| [COMBO: Compositional World Models for Embodied Multi-Agent Cooperation](https://arxiv.org/abs/2502.02534) | ArXiv 2025 |
| [VIKI-R: A VLM-Based Reinforcement Learning Approach for Heterogeneous Multi-Agent Cooperation](https://arxiv.org/abs/2502.06450) | ArXiv 2025 |
| [RoCo: Dialectic Multi-Robot Collaboration with Large Language Models](https://arxiv.org/abs/2307.04738) | ArXiv 2024 |

### 🏥 Healthcare & Medicine Agents

#### Foundational agentic reasoning

| Paper | Venue |
| --- | --- |
| [Autonomous artificial intelligence agents for clinical decision making in oncology](https://www.nature.com/articles/s43018-025-00991-6) | Nature Medicine 2024 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [PathFinder: Multimodal Multi-Agent Medical Diagnosis Framework](https://arxiv.org/abs/2501.12933) | ArXiv 2025 |
| [MedAgent-Pro: Evidence-Based Multimodal Medical Agent for Complex Reasoning](https://arxiv.org/abs/2502.13843) | ArXiv 2025 |
| [MedOrch: Medical Diagnosis through Tool-Augmented Agent Orchestration](https://arxiv.org/abs/2502.14088) | ArXiv 2025 |
| [ClinicalAgent: A Multi-Agent Framework for Clinical Trial Prediction](https://arxiv.org/abs/2410.16542) | ArXiv 2024 |
| [DoctorAgent-RL: Multi-Agent Collaborative Reinforcement Learning for Medical History Taking](https://arxiv.org/abs/2502.04947) | ArXiv 2025 |
| [DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making](https://arxiv.org/abs/2507.02616) | ArXiv 2025 |
| [MedOrch: Medical Diagnosis through Tool-Augmented Agent Orchestration](https://arxiv.org/abs/2502.14088) | ArXiv 2025 |
| [TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools](https://arxiv.org/abs/2503.10970) | ArXiv 2025 |
| [AgentMD: Empowering language agents for risk prediction with large-scale clinical tool learning](https://www.nature.com/articles/s41467-025-64430-x) | Nature Communications 2025 |
| [Large language model agents can use tools to perform clinical calculations](https://doi.org/10.1038/s41746-025-01475-8) | NPJ Digital Medicine 2025 |
| [MeNTi: Bridging Medical Calculator and Large Language Models for Clinical Calculation](https://arxiv.org/abs/2501.03719) | ArXiv 2025 |
| [MMedAgent: Learning to Use Medical Tools with Multi-modal Agents](https://arxiv.org/abs/2402.12649) | ArXiv 2024 |
| [VoxelPrompt: A Vision-Language Agent Grounded in 3D Volumetric Medical Images](https://arxiv.org/abs/2412.18042) | ArXiv 2024 |
| [Enhancing Surgical Robots with Embodied Agents for Autonomous 3D Ultrasound Scanning](https://arxiv.org/abs/2407.13280) | ArXiv 2024 |
| [Adaptive Reasoning and Acting in Medical Language Agents](https://arxiv.org/abs/2410.10020) | ArXiv 2024 |
| [MedRAX: Medical Reasoning Agent for Chest X-ray](https://arxiv.org/abs/2502.02673) | ArXiv 2025 |
| [PathFinder: Multimodal Multi-Agent Medical Diagnosis Framework](https://arxiv.org/abs/2501.12933) | ArXiv 2025 |
| [Conversational Health Agents: A Personalized LLM-Powered Agent Framework](https://arxiv.org/abs/2310.02374) | ArXiv 2024 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [MedAgentGym: Training LLM Agents for Medical Coding](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [DoctorAgent-RL: Multi-Agent Collaborative Reinforcement Learning for Medical History Taking](https://arxiv.org/abs/2502.04947) | ArXiv 2025 |
| [Simulated patient systems powered by large language model-based AI agents offer potential for transforming medical education](https://arxiv.org/abs/2409.18924) | ArXiv 2024 |
| [MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions](https://arxiv.org/abs/2501.08226) | MICCAI 2025 |
| [MeNTi: Bridging Medical Calculator and Large Language Models for Clinical Calculation](https://arxiv.org/abs/2501.03719) | ArXiv 2025 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [Conversational Health Agents: A Personalized LLM-Powered Agent Framework](https://arxiv.org/abs/2310.02374) | ArXiv 2024 |
| [RAG-Enhanced Collaborative LLM Agents for Drug Discovery](https://arxiv.org/abs/2501.07842) | ArXiv 2025 |
| [MedReason: Eliciting Factual Medical Reasoning in LLMs](https://arxiv.org/abs/2502.13876) | ArXiv 2025 |

#### Self-evolving agentic reasoning

| Paper | Venue |
| --- | --- |
| [Epidemic Modeling with Generative Agents](https://arxiv.org/abs/2307.04986) | ArXiv 2023 |
| [MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions](https://arxiv.org/abs/2501.08226) | MICCAI 2025 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [LLMs Simulate Standardized Patients: Evolution of a Medical Education Tool](https://arxiv.org/abs/2502.01633) | ArXiv 2025 |
| [Simulated patient systems powered by large language model-based AI agents offer potential for transforming medical education](https://arxiv.org/abs/2409.18924) | ArXiv 2024 |
| [MedOrch: Medical Diagnosis through Tool-Augmented Agent Orchestration](https://arxiv.org/abs/2502.14088) | ArXiv 2025 |
| [DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making](https://arxiv.org/abs/2507.02616) | ArXiv 2025 |
| [DoctorAgent-RL: Multi-Agent Collaborative Reinforcement Learning for Medical History Taking](https://arxiv.org/abs/2502.04947) | ArXiv 2025 |
| [MedAgentGym: Training LLM Agents for Medical Coding](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [MeNTi: Bridging Medical Calculator and Large Language Models for Clinical Calculation](https://arxiv.org/abs/2501.03719) | ArXiv 2025 |
| [Large language model agents can use tools to perform clinical calculations](https://doi.org/10.1038/s41746-025-01475-8) | NPJ Digital Medicine 2025 |

#### Collective multi-agent reasoning

| Paper | Venue |
| --- | --- |
| [MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making](https://arxiv.org/abs/2404.15155) | ArXiv 2024 |
| [DoctorAgent-RL: Multi-Agent Collaborative Reinforcement Learning for Medical History Taking](https://arxiv.org/abs/2502.04947) | ArXiv 2025 |
| [Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis](https://arxiv.org/abs/2401.16107) | ArXiv 2024 |
| [ClinicalAgent: A Multi-Agent Framework for Clinical Trial Prediction](https://arxiv.org/abs/2410.16542) | ArXiv 2024 |
| [PathFinder: Multimodal Multi-Agent Medical Diagnosis Framework](https://arxiv.org/abs/2501.12933) | ArXiv 2025 |
| [MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions](https://arxiv.org/abs/2501.08226) | MICCAI 2025 |
| [LLMs Simulate Standardized Patients: Evolution of a Medical Education Tool](https://arxiv.org/abs/2502.01633) | ArXiv 2025 |
| [DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making](https://arxiv.org/abs/2507.02616) | ArXiv 2025 |
| [MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning](https://arxiv.org/abs/2311.10537) | ArXiv 2024 |
| [RAG-Enhanced Collaborative LLM Agents for Drug Discovery](https://arxiv.org/abs/2501.07842) | ArXiv 2025 |
| [GMAI-VL-R1: Harnessing Reinforcement Learning for Multi-Modal Medical Reasoning](https://arxiv.org/abs/2502.03260) | ArXiv 2025 |

### 🌐 Autonomous Web Exploration & Research Agents

#### Foundational agentic reasoning

| Paper | Venue |
| --- | --- |
| [Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227) | ArXiv 2025 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | GitHub 2023 |
| [Chain of Ideas: Revolutionizing Research with AI-Driven Hypothesis Generation](https://arxiv.org/abs/2410.16010) | ArXiv 2024 |
| [IRIS: A Tree-Search Agent for Complex Knowledge-Intensive Tasks](https://arxiv.org/abs/2501.12933) | ArXiv 2025 |
| [Accelerating Scientific Research Through a Multi-LLM Framework (ARIA)](https://arxiv.org/abs/2502.07960) | ArXiv 2025 |
| [NovelSeek: When Agent Becomes the Scientist](https://arxiv.org/abs/2505.16938) | ArXiv 2025 |
| [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) | ArXiv 2021 |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | NeurIPS 2020 |
| [GPT-4V(ision) is a Generalist Web Agent, if Grounded](https://arxiv.org/abs/2401.01614) | ICML 2024 |
| [AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Agent for Automated Web Navigation](https://arxiv.org/abs/2404.03648) | ArXiv 2024 |
| [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199) | ArXiv 2024 |
| [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501) | ArXiv 2025 |
| [WebSailor: Uncertainty-Driven Post-Training for Web Agents](https://arxiv.org/abs/2501.03606) | ArXiv 2025 |
| [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/abs/2411.02337) | ArXiv 2024 |
| [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) | ArXiv 2025 |
| [Navigating WebAI: Training Agents to Complete Web Tasks with Large Language Models and Reinforcement Learning](https://arxiv.org/abs/2405.00516) | ArXiv 2024 |
| [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](https://arxiv.org/abs/2505.24332) | ArXiv 2025 |
| [EvolveSearch: An Iterative Self-Evolving Search Agent](https://arxiv.org/abs/2505.22501) | ArXiv 2025 |
| [WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model](https://arxiv.org/abs/2502.10126) | ArXiv 2025 |
| [ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL](https://arxiv.org/abs/2402.19446) | ICLR 2025 |
| [PAE: Autonomous Skill Discovery for Foundation Model Internet Agents](https://arxiv.org/abs/2411.10705) | ArXiv 2024 |
| [WebSeer: Reflective Reinforcement Learning for Web Agents](https://arxiv.org/abs/2505.01188) | ArXiv 2025 |
| [ZeroSearch: Incentivize the Search Capability of LLMs Without Searching](https://arxiv.org/abs/2505.04588) | ArXiv 2025 |
| [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](https://arxiv.org/abs/2505.06825) | ArXiv 2025 |
| [How to Train Your LLM Web Agent: A Statistical Diagnosis](https://arxiv.org/abs/2507.04103) | ArXiv 2025 |
| [OS-Copilot: Towards Generalist Computer Agents with Self-Improvement](https://arxiv.org/abs/2402.07456) | ArXiv 2024 |
| [Agent S: An Open Agentic Framework that Uses Computers Like a Human](https://arxiv.org/abs/2410.08164) | ArXiv 2024 |
| [InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection](https://arxiv.org/abs/2501.04164) | ArXiv 2025 |
| [MobA: A Two-Level Agent System for Efficient Mobile Task Automation](https://arxiv.org/abs/2410.13757) | ArXiv 2024 |
| [PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC](https://arxiv.org/abs/2502.14282) | ArXiv 2025 |
| [OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://arxiv.org/abs/2410.23218) | ArXiv 2024 |
| [OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning](https://arxiv.org/abs/2410.18963) | ArXiv 2024 |
| [UItron: Foundational GUI Agent with Advanced Perception and Planning](https://arxiv.org/abs/2508.21767) | ArXiv 2025 |
| [ARPO: Aligning GUI Agents via Preference Optimization](https://arxiv.org/abs/2501.02675) | ArXiv 2025 |
| [ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents](https://arxiv.org/abs/2508.14040) | ArXiv 2025 |
| [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/abs/2503.21620) | ArXiv 2025 |
| [GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/abs/2504.10458) | ArXiv 2025 |
| [InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/abs/2504.14239) | ArXiv 2025 |
| [UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning](https://arxiv.org/abs/2509.11543) | ArXiv 2025 |
| [GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration](https://aclanthology.org/2025.emnlp-main.1688/) | EMNLP 2025 |
| [Learning GUI Grounding with Spatial Reasoning from Visual Feedback (SE-GUI)](https://arxiv.org/abs/2509.21552) | ArXiv 2025 |
| [UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning](https://arxiv.org/abs/2505.12493) | ArXiv 2025 |
| [UI-AGILE: Advancing GUI Agents With Effective Reinforcement Learning](https://arxiv.org/abs/2507.22025) | ArXiv 2025 |
| [ZeroGUI: Automating Online GUI Learning at Zero Human Cost](https://arxiv.org/abs/2505.23762) | ArXiv 2025 |
| [AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.01391) | ArXiv 2025 |
| [AutoGLM: Autonomous GUI Agent](https://arxiv.org/abs/2411.00820) | ArXiv 2024 |
| [Mobile-Agent-v3: Fundamental Agents for GUI Automation](https://arxiv.org/abs/2508.15144) | ArXiv 2025 |
| [WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models](https://arxiv.org/abs/2401.13919) | ACL 2024 |
| [BrowserAgent: A Generalist Agent for Web Navigation](https://arxiv.org/abs/2502.10092) | ArXiv 2025 |
| [WALT: Watch And Learn Tool-use for Web Agents](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [WebDancer: Towards Automated Web Information Seeking with Large Language Model Agents](https://arxiv.org/abs/2502.01026) | ArXiv 2025 |
| [WebShaper: Synthesizing Information-Seeking Web Agents](https://arxiv.org/abs/2507.15061) | ArXiv 2025 |
| [AutoDroid: LLM-powered Task Automation in Android](https://arxiv.org/abs/2308.15272) | MobiCom 2024 |
| [MobileExperts: Orchestrating Tool-Capable Specialists for Mobile Automation](https://arxiv.org/abs/2411.00622) | ArXiv 2024 |
| [AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant](https://arxiv.org/abs/2410.18603) | ArXiv 2024 |
| [OS-Copilot: Towards Generalist Computer Agents with Self-Improvement](https://arxiv.org/abs/2402.07456) | ArXiv 2024 |
| [OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning](https://arxiv.org/abs/2410.18963) | ArXiv 2024 |
| [OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://arxiv.org/abs/2410.23218) | ArXiv 2024 |
| [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935) | ArXiv 2024 |
| [Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](https://arxiv.org/abs/2502.04644) | ArXiv 2025 |
| [WebThinker: Empowering Large Reasoning Models for Deep Web Exploration](https://arxiv.org/abs/2502.10093) | ArXiv 2025 |
| [PaperQA: Retrieval-Augmented Generative Agent for Scientific Research](https://arxiv.org/abs/2312.07559) | ArXiv 2023 |
| [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740) | ArXiv 2024 |
| [Scideator: Human-AI Collaborative Scientific Idea Generation](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) | ArXiv 2025 |
| [Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227) | ArXiv 2025 |
| [MLR-Copilot: Autonomous Machine Learning Research based on Large Language Model Agents](https://arxiv.org/abs/2408.14033) | ArXiv 2024 |
| [Dolphin: A Code-Centric Autonomous Research Agent](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [The AI Scientist: Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) | ArXiv 2024 |
| [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066) | ArXiv 2025 |
| [NovelSeek: When Agent Becomes the Scientist](https://arxiv.org/abs/2505.16938) | ArXiv 2025 |
| [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501) | ArXiv 2025 |
| [WebSailor: Uncertainty-Driven Post-Training for Web Agents](https://arxiv.org/abs/2501.03606) | ArXiv 2025 |
| [RaDA: Retrieval-augmented Task Decomposition and Action Generation](https://arxiv.org/abs/2411.00820) | ArXiv 2024 |
| [Synapse: Trajectory-as-Exemplar Prompting for Computer Control](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3DZ3s4FqZ6j3) | ICLR 2024 |
| [LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark](https://arxiv.org/abs/2504.13805) | ArXiv 2025 |
| [MobileGPT: A Mobile Assistant with Human-like App Memory](https://arxiv.org/abs/2311.16528) | ArXiv 2023 |
| [Retrieval-augmented GUI Agents with Generative Guidelines (RAG-GUI)](https://arxiv.org/abs/2509.24183) | ArXiv 2025 |
| [WebThinker: Empowering Large Reasoning Models for Deep Web Exploration](https://arxiv.org/abs/2502.10093) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) | ArXiv 2025 |
| [PaperQA: Retrieval-Augmented Generative Agent for Scientific Research](https://arxiv.org/abs/2312.07559) | ArXiv 2023 |
| [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740) | ArXiv 2024 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | GitHub 2023 |
| [Chain of Ideas: Revolutionizing Research with AI-Driven Hypothesis Generation](https://arxiv.org/abs/2410.16010) | ArXiv 2024 |
| [Scideator: Human-AI Collaborative Scientific Idea Generation](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |

#### Self-evolving agentic reasoning

| Paper | Venue |
| --- | --- |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | ArXiv 2024 |
| [ICAL: In-Context Abstraction Learning for Vision-Language Agents](https://arxiv.org/abs/2406.14596) | ArXiv 2024 |
| [BrowserAgent: A Generalist Agent for Web Navigation](https://arxiv.org/abs/2502.10092) | ArXiv 2025 |
| [AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Agent for Automated Web Navigation](https://arxiv.org/abs/2404.03648) | ArXiv 2024 |
| [AgentOccam: A Simple Yet Strong Baseline for Text-Based Web Agents](https://arxiv.org/abs/2406.12658) | ArXiv 2024 |
| [LiteWebAgent: The Open-Source Suite for VLM-Based Web-Agent Applications](https://arxiv.org/abs/2503.02950) | ArXiv 2025 |
| [WebDancer: Towards Automated Web Information Seeking with Large Language Model Agents](https://arxiv.org/abs/2502.01026) | ArXiv 2025 |
| [WebShaper: Synthesizing Information-Seeking Web Agents](https://arxiv.org/abs/2507.15061) | ArXiv 2025 |
| [MobileGPT: A Mobile Assistant with Human-like App Memory](https://arxiv.org/abs/2311.16528) | ArXiv 2023 |
| [MobA: A Two-Level Agent System for Efficient Mobile Task Automation](https://arxiv.org/abs/2410.13757) | ArXiv 2024 |
| [Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks](https://arxiv.org/abs/2501.11733) | ArXiv 2025 |
| [Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227) | ArXiv 2025 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | GitHub 2023 |
| [Chain of Ideas: Revolutionizing Research with AI-Driven Hypothesis Generation](https://arxiv.org/abs/2410.16010) | ArXiv 2024 |
| [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066) | ArXiv 2025 |
| [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199) | ArXiv 2024 |
| [ReAP: Reflection-Based Memory For Web navigation Agents](https://arxiv.org/abs/2506.02158) | ArXiv 2025 |
| [Agent-E: From Autonomous Web Navigation to Foundational Design Principles in Agentic Systems](https://arxiv.org/abs/2407.13032) | ArXiv 2024 |
| [Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance](https://arxiv.org/abs/2509.21072) | ArXiv 2025 |
| [WINELL: Wikipedia Never-Ending Updating with LLM Agents](https://arxiv.org/abs/2508.03728) | ArXiv 2025 |
| [WebSeer: Reflective Reinforcement Learning for Web Agents](https://arxiv.org/abs/2505.01188) | ArXiv 2025 |
| [Zero-Shot GUI Automation via Self-Correction](https://arxiv.org/abs/2311.16528) | ArXiv 2023 |
| [Empowering Multimodal GUI Models with Self-Reflection Behavior (GUI-Reflection)](https://arxiv.org/abs/2506.08012) | ArXiv 2025 |
| [History-Aware Reasoning for GUI Agents](https://arxiv.org/abs/2511.09127) | ArXiv 2025 |
| [MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation](https://arxiv.org/abs/2507.16853) | ArXiv 2025 |
| [InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection](https://arxiv.org/abs/2501.04164) | ArXiv 2025 |
| [Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks](https://arxiv.org/abs/2501.11733) | ArXiv 2025 |
| [CycleResearcher: Improving Automated Research via Automated Review](https://arxiv.org/abs/2411.00816) | ArXiv 2024 |
| [MLR-Copilot: Autonomous Machine Learning Research based on Large Language Model Agents](https://arxiv.org/abs/2408.14033) | ArXiv 2024 |
| [Dolphin: A Code-Centric Autonomous Research Agent](https://arxiv.org/abs/2501.15783) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) | ArXiv 2025 |

#### Collective multi-agent reasoning

| Paper | Venue |
| --- | --- |
| [WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution](https://arxiv.org/abs/2408.15978) | ArXiv 2024 |
| [WINELL: Wikipedia Never-Ending Updating with LLM Agents](https://arxiv.org/abs/2508.03728) | ArXiv 2025 |
| [Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance](https://arxiv.org/abs/2509.21072) | ArXiv 2025 |
| [PAE: Autonomous Skill Discovery for Foundation Model Internet Agents](https://arxiv.org/abs/2411.10705) | ArXiv 2024 |
| [Agent-E: From Autonomous Web Navigation to Foundational Design Principles in Agentic Systems](https://arxiv.org/abs/2407.13032) | ArXiv 2024 |
| [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) | ArXiv 2025 |
| [Agentic Web: Weaving the Next Web with AI Agents](https://arxiv.org/abs/2507.21206) | ArXiv 2025 |
| [CoLA: Collaborative Low-Rank Adaptation](https://arxiv.org/abs/2505.15471) | ArXiv 2025 |
| [Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation, Operation and Correction](https://arxiv.org/abs/2406.01014) | ACL 2024 |
| [Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks](https://arxiv.org/abs/2501.11733) | ArXiv 2025 |
| [Mobile-Agent-V: Learning Mobile Device Operation Through Video-Guided Multi-Agent Collaboration](https://arxiv.org/abs/2502.17110) | ArXiv 2025 |
| [MobileExperts: Orchestrating Tool-Capable Specialists for Mobile Automation](https://arxiv.org/abs/2411.00622) | ArXiv 2024 |
| [SWIRL: Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736) | ArXiv 2025 |
| [PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC](https://arxiv.org/abs/2502.14282) | ArXiv 2025 |
| [AgentRxiv: Towards Collaborative Autonomous Research](https://arxiv.org/abs/2503.18102) | ArXiv 2025 |
| [Accelerating Scientific Research Through a Multi-LLM Framework (ARIA)](https://arxiv.org/abs/2502.07960) | ArXiv 2025 |
| [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | NeurIPS 2022 |
| [Emergent autonomous scientific research capabilities of large language models](https://arxiv.org/abs/2304.05332) | Nature 2023 |
| [Toward a Team of AI-made Scientists for Scientific Discovery from Gene Expression Data](https://arxiv.org/abs/2402.12391) | ArXiv 2024 |

---

## 📊 Benchmarks

![bench](figs/benchmark.png)


### ⚙️ Core Mechanisms of Agentic Reasoning


#### Tool Use


##### Single-Turn Tool Use

| Paper | Venue |
| --- | --- |
| [ToolQA: A Dataset for LLM Question Answering with External Tools](https://openreview.net/forum?id=pV1xV2RK6I) | NeurIPS 2023 |
| [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) | ArXiv 2023 |
| [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) | ICLR 2024 |
| [MetaTool: A Benchmark for Controlling Special-purpose Large Language Models](https://arxiv.org/abs/2310.03128) | ICLR 2024 |
| [T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step](https://arxiv.org/abs/2312.14033) | ACL 2024 |
| [GTA: A Benchmark for General Tool Agents](https://arxiv.org/abs/2407.08713) | NeurIPS 2024 |
| [Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](https://arxiv.org/abs/2503.01763) | ArXiv 2025 |

##### Multi-Turn Tool Use

| Paper | Venue |
| --- | --- |
| [ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301) | ArXiv 2023 |
| [On the Tool Manipulation Capability of Open-source Large Language Models](https://arxiv.org/abs/2305.16504) | ArXiv 2023 |
| [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://aclanthology.org/2023.emnlp-main.187/) | EMNLP 2023 |
| [Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios](https://arxiv.org/abs/2401.17167) | ACL 2024 |
| [MTU-Bench: A Multi-granularity Tool-Use Benchmark for Large Language Models](https://openreview.net/forum?id=6guG2OlXsr) | ICLR 2025 |

#### Memory

##### Long-Horizon Episodic Memory

| Paper | Venue |
| --- | --- |
| [PerLTQA: A Persona-based Long-term Memory Benchmark for RAG](https://arxiv.org/abs/2402.16288) | ArXiv 2024 |
| [ELITR-Bench: A Meeting Assistant Benchmark for Long-Context LLMs](https://arxiv.org/abs/2403.20262) | ArXiv 2024 |
| [Multi-IF: A Benchmark for Multi-turn Instruction Following](https://arxiv.org/abs/2410.15553) | ArXiv 2024 |
| [MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs](https://arxiv.org/abs/2501.17399v1) | ArXiv 2025 |
| [TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models](https://arxiv.org/abs/2506.01341) | ArXiv 2025 |
| [StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns](https://arxiv.org/abs/2506.13356) | ArXiv 2025 |
| [MemBench: Long-Context LLMs are not Enough](https://arxiv.org/abs/2501.12745) | ArXiv 2025 |
| [MMRC: A Large-Scale Benchmark for Understanding Multimodal Large Language Model in Real-World Conversation](https://arxiv.org/abs/2502.11903) | ArXiv 2025 |

##### Multi-session Recall

| Paper | Venue |
| --- | --- |
| [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) | ArXiv 2024 |
| [MemSim: A Bayesian Simulator for Evaluating Memory of LLM-based Personal Assistants](https://arxiv.org/abs/2409.20163) | ArXiv 2024 |
| [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) | ArXiv 2024 |
| [REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation](https://arxiv.org/abs/2502.13270) | ArXiv 2025 |
| [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257) | ArXiv 2025 |
| [Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents](https://arxiv.org/abs/2601.03515) | ArXiv 2026 |
| [Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory](https://arxiv.org/abs/2511.20857) | ArXiv 2025 |


#### Multi-Agent System

##### Game-based reinforcement learning evaluation

| Paper | Venue |
| --- | --- |
| [MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence](https://arxiv.org/abs/1712.00600) | AAAI 2018 |
| [Pommerman: A Multi-Agent Playground](https://arxiv.org/abs/1809.07124) | ArXiv 2018 |
| [The StarCraft Multi-Agent Challenge](https://arxiv.org/abs/1902.04043) | NeurIPS 2019 |
| [MineLand: Simulating Large-Scale Multi-Agent Interactions with Limited Multimodal Senses and Physical Needs](https://arxiv.org/abs/2403.19267) | ArXiv 2024 |
| [TeamCraft: A Benchmark for Multi-Modal Multi-Agent Systems in Minecraft](https://arxiv.org/abs/2412.05255) | ArXiv 2024 |
| [Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot](https://arxiv.org/abs/2107.06857) | ICML 2021 |
| [BenchMARL: Benchmarking Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2312.01472) | ArXiv 2023 |
| [Arena: A General Evaluation Platform and Building Toolkit for Multi-Agent Intelligence](https://arxiv.org/abs/1905.08085) | AAAI 2020 |

#### Simulation-centric real-world assessment

| Paper | Venue |
| --- | --- |
| [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776) | CoRL 2020 |
| [Nocturne: a scalable driving benchmark for bringing multi-agent learning one step closer to the real world](https://arxiv.org/abs/2206.09889) | NeurIPS 2022 |
| [A Versatile Multi-Agent Reinforcement Learning Benchmark for Inventory Management](https://arxiv.org/abs/2306.07542) | ArXiv 2023 |
| [IMP-MARL: a Suite of Environments for Infrastructure Management Planning with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2306.11551) | NeurIPS 2023 |
| [POGEMA: Partially Observable Grid Environment for Multiple Agents](https://arxiv.org/abs/2206.10944) | Arxiv 2022 |
| [IntersectionZoo: Eco-driving for Benchmarking Multi-Agent Contextual Reinforcement Learning](https://arxiv.org/abs/2410.15221) | NeurIPS 2024 |
| [REALM-Bench: A Benchmark for Evaluating Multi-Agent Systems on Real-world, Dynamic Planning and Scheduling Tasks](https://arxiv.org/abs/2502.18836v2) | ArXiv 2025 |

#### Language, Communication, and Social Reasoning

| Paper | Venue |
| --- | --- |
| [LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models](https://arxiv.org/abs/2310.03903) | ArXiv 2023 |
| [AvalonBench: Evaluating LLMs Playing the Game of Avalon](https://arxiv.org/abs/2310.05036) | ArXiv 2023 |
| [Welfare Diplomacy: Benchmarking Language Model Cooperation](https://arxiv.org/abs/2310.08901) | ArXiv 2023 |
| [MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration](https://arxiv.org/abs/2311.08562) | 	EMNLP 2024 |
| [BattleAgentBench: A Benchmark for Evaluating Cooperation and Competition Capabilities of Language Models in Multi-Agent Systems](https://arxiv.org/abs/2408.15971) | ArXiv 2024 |
| [COMMA: A Benchmark for Inter-Agent Communication in Multi-Agent Systems](https://arxiv.org/abs/2410.07553) | ArXiv 2024 |
| [IntellAgent: A Benchmark for Evaluating Conversational Agents in Realistic Scenarios](https://arxiv.org/abs/2501.11067v1) | ArXiv 2025 |
| [MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents](https://arxiv.org/abs/2503.01935) | ArXiv 2025 |





### 🎯 Applications of Agentic Reasoning



#### Embodied Agents

| Paper | Venue |
| --- | --- |
| [Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks](https://arxiv.org/abs/2505.24876) | ArXiv 2025 |
| [BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games](https://arxiv.org/abs/2411.13543) | NeurIPS 2024 |
| [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768) | ICLR 2021 |
| [Understanding the Weakness of Large Language Model Agents within a Complex Android Environment](https://arxiv.org/abs/2402.06596) | ArXiv 2024 |
| [MindAgent: Emergent Gaming Interaction](https://arxiv.org/abs/2309.09971) | ArXiv 2023 |
| [Playing repeated games with Large Language Models](https://arxiv.org/abs/2305.16867) | ArXiv 2023 |
| [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972) | NeurIPS 2024 |



#### Scientific Discovery Agents

| Paper | Venue |
| --- | --- |
| [DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents](https://arxiv.org/abs/2406.06769) | NeurIPS 2024 |
| [ScienceWorld: Is your Agent Smarter than a 5th Grader?](https://arxiv.org/abs/2203.07540) | EMNLP 2022 |
| [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/abs/2410.05080) | NeurIPS 2024 |
| [The AI Scientist: Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) | ArXiv 2024 |
| [LAB-Bench: Measuring Capabilities of Language Models for Biology Research](https://arxiv.org/abs/2407.10362) | ArXiv 2024 |
| [MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation](https://arxiv.org/abs/2310.03302) | ArXiv 2023 |



#### Autonomous Research Agents

| Paper | Venue |
| --- | --- |
| [WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?](https://arxiv.org/abs/2403.07718) | ICML 2024 |
| [WorkArena++: Towards Agents that Act Like Employees](https://arxiv.org/abs/2407.05291) | ArXiv 2024 |
| [OfficeBench: A Benchmark for Office Tool Agents](https://arxiv.org/abs/2407.02685) | ArXiv 2024 |
| [PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change](https://arxiv.org/abs/2206.10498) | NeurIPS 2022 |
| [FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents](https://arxiv.org/abs/2406.14884) | ArXiv 2024 |
| [ACPBench: Reasoning about Action, Change, and Planning](https://arxiv.org/abs/2410.05669) | ArXiv 2024 |
| [TRAIL: Trace Reasoning and Agentic Issue Localization](https://arxiv.org/abs/2505.08638) | ArXiv 2025 |
| [CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization](https://arxiv.org/abs/2310.10134) | NeurIPS 2023 |
| [Agent-as-a-Judge: Evaluate Agents with Agents](https://arxiv.org/abs/2410.10934) | ArXiv 2024 |
| [InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation](https://arxiv.org/abs/2505.15872) | ArXiv 2025 |



#### Medical and Clinical Agents

| Paper | Venue |
| --- | --- |
| [AgentClinic: a multimodal agent benchmark for clinical environments](https://arxiv.org/abs/2405.07960) | NeurIPS 2024 |
| [MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents](https://www.researchgate.net/publication/395098333_MedAgentBench_A_Virtual_EHR_Environment_to_Benchmark_Medical_LLM_Agents) | NEJM AI 2025 |
| [EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/abs/2401.07128) | ArXiv 2024 |
| [MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning](https://arxiv.org/abs/2311.10537) | ArXiv 2023 |
| [GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning](https://arxiv.org/abs/2406.09187) | ArXiv 2024 |



#### Web Agents

| Paper | Venue |
| --- | --- |
| [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854) | ICLR 2024 |
| [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks](https://arxiv.org/abs/2401.13649) | NeurIPS 2024 |
| [WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models](https://arxiv.org/abs/2401.13919) | ACL 2024 |
| [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070) | NeurIPS 2023 |
| [WebCanvas: Benchmarking Web Agents in Online Canvas](https://arxiv.org/abs/2406.12373) | NeurIPS 2024 |
| [WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930) | CVPR 2024 |
| [LASER: LLM Agent with State-Space Exploration for Web Navigation](https://arxiv.org/abs/2309.08172) | NeurIPS 2023 |
| [AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Agent for Automated Web Navigation](https://arxiv.org/abs/2404.03648) | ArXiv 2024 |



#### General Tool-Use Agents

| Paper | Venue |
| --- | --- |
| [GTA: A Benchmark for General Tool Agents](https://arxiv.org/abs/2407.08713) | NeurIPS 2024 |
| [NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls](https://arxiv.org/abs/2409.03797) | ArXiv 2024 |
| [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030) | ICML 2024 |
| [RestGPT: Connecting Large Language Models with Real-World RESTful APIs](https://arxiv.org/abs/2306.06624) | EMNLP 2023 |
| [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366) | ArXiv 2025 |
| [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441) | ArXiv 2025 |
| [ActionReasoningBench: Reasoning about Actions with and without Ramification Constraints](https://arxiv.org/abs/2406.04046) | ArXiv 2024 |
| [R-Judge: Benchmarking Safety-Critical Decision Making for LLM Agents](https://arxiv.org/abs/2401.10019) | ArXiv 2024 |





## License

This repository is licensed under the MIT License.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weitianxin/Awesome-Agentic-Reasoning&type=Date)](https://star-history.com/#weitianxin/Awesome-Agentic-Reasoning&Date)
