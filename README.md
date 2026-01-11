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
      - [Workflow Design](#workflow-design)
      - [Tree Search / Algorithm Simulation](#tree-search--algorithm-simulation)
      - [Process Formalization](#process-formalization)
      - [Decoupling / Decomposition](#decoupling--decomposition)
      - [External Aid / Tool Use](#external-aid--tool-use)
      - [3.1.2 Post-training Planning](#312-post-training-planning)
    - [🛠️ 3.2 Tool-Use Optimization](#️-32-tool-use-optimization)
      - [3.2.1 In-Context Tool-Integration](#321-in-context-tool-integration)
        - [Interleaving Reasoning and Tool Use](#interleaving-reasoning-and-tool-use)
        - [Optimizing Context for Tool Interaction](#optimizing-context-for-tool-interaction)
      - [3.2.2 Post-training Tool-Integration](#322-post-training-tool-integration)
        - [Bootstrapping of Tool Use via SFT](#bootstrapping-of-tool-use-via-sft)
        - [Mastery of Tool Use via RL](#mastery-of-tool-use-via-rl)
      - [3.2.3 Orchestration-based Tool-Integration](#323-orchestration-based-tool-integration)
        - [Agentic Pipelines for Tool Orchestration](#agentic-pipelines-for-tool-orchestration)
        - [Tool Representations for Orchestration](#tool-representations-for-orchestration)
    - [🔍 3.3 Agentic Search](#-33-agentic-search)
      - [3.3.1 In-Context Search](#331-in-context-search)
        - [Interleaving Reasoning and Search](#interleaving-reasoning-and-search)
        - [Structure-Enhanced Search](#structure-enhanced-search)
      - [3.3.2 Post-Training Search](#332-post-training-search)
        - [SFT-Based Agentic Search](#sft-based-agentic-search)
        - [RL-Based Agentic Search](#rl-based-agentic-search)
  - [4. 🧬 Self-evolving Agentic Reasoning](#4--self-evolving-agentic-reasoning)
    - [🔄 4.1 Agentic Feedback Mechanisms](#-41-agentic-feedback-mechanisms)
      - [4.1.1 Reflective Feedback](#411-reflective-feedback)
      - [4.1.2 Parametric Adaptation](#412-parametric-adaptation)
      - [4.1.3 Validator-Driven Feedback](#413-validator-driven-feedback)
    - [🧠 4.2 Agentic Memory](#-42-agentic-memory)
      - [4.2.1 Agentic Use of Memory](#421-agentic-use-of-memory)
        - [Conversational Memory and Factual Memory](#conversational-memory-and-factual-memory)
        - [Reasoning Memory and Experience Reuse](#reasoning-memory-and-experience-reuse)
        - [Multimodal Extensions](#multimodal-extensions)
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

#### Workflow Design

| Paper | Venue |
| --- | --- |
| [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477) | ArXiv 2023 |
| [PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change](https://arxiv.org/abs/2206.10498) | NeurIPS 2024 |
| [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323) | NeurIPS 2023 |
| [LLM-Reasoners: New Evaluation Approaches for Large Language Models](https://arxiv.org/abs/2404.05663) | ArXiv 2024 |
| [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) | ICLR 2023 |
| [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091) | ACL 2023 |
| [Algorithm of Thoughts: Enhancing LLM Reasoning Capabilities via Algorithmic Reasoning](https://arxiv.org/abs/2308.10379) | ArXiv 2023 |
| [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/abs/2303.17580) | NeurIPS 2023 |
| [Plan, Eliminate, and Track -- Language Models are Good Teachers for Embodied Agents](https://arxiv.org/abs/2305.02412) | ArXiv 2023 |
| [PERIA: A Unified Multimodal Workflow](https://arxiv.org/abs/2404.16836) | ArXiv 2024 |
| [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) | ArXiv 2025 |
| [CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499) | FSE 2024 |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070) | NeurIPS 2023 |
| [Wilbur: Adaptive In-Context Learning for Robust and Accurate Web Agents](https://arxiv.org/abs/2404.05902) | ArXiv 2024 |
| [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.10312) | ICML 2024 |
| [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) | NeurIPS 2024 |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [CodeNav: Beyond Tool-Use to Using Real-World Codebases with LLM Agents](https://arxiv.org/abs/2402.13463) | ICLR 2024 |
| [MARCO: A Multi-Agent System for Optimizing HPC Code Generation](https://arxiv.org/abs/2505.03906) | ArXiv 2025 |
| [Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection Agents](https://arxiv.org/abs/2501.00430) | ArXiv 2024 |
| [Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents](https://arxiv.org/abs/2505.09970) | ArXiv 2025 |
| [REST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2311.07000) | ArXiv 2023 |
| [Self-Planning Code Generation with Large Language Models](https://arxiv.org/abs/2306.02907) | TOSEM 2024 |
| [Real-Time Conversational Agent with Adaptive Planning](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/abs/2207.04429) | CoRL 2023 |

#### Tree Search / Algorithm Simulation

| Paper | Venue |
| --- | --- |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | NeurIPS 2023 |
| [Tree Search for Language Model Agents](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Large Language Model Guided Tree Search for Program Synthesis](https://arxiv.org/abs/2305.03742) | ArXiv 2023 |
| [Tree-Planner: Efficient Planning with Large Language Models](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283) | ArXiv 2024 |
| [LLM-A*: Large Language Model Guided A* Search](https://www.google.com/search?q=https://arxiv.org/abs/2401.00000) | ArXiv 2024 |
| [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) | ArXiv 2024 |
| [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992) | NeurIPS 2023 |
| [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Monte Carlo Tree Search with Large Language Models](https://www.google.com/search?q=https://arxiv.org/abs/2310.00000) | ArXiv 2023 |
| [Prompt-Based Monte-Carlo Tree Search for Goal-Oriented Dialogue](https://www.google.com/search?q=https://arxiv.org/abs/2305.00000) | ArXiv 2023 |
| [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126) | ICLR 2024 |
| [Everything of Thoughts: Defying the Laws of Pen and Paper](https://arxiv.org/abs/2311.04254) | ArXiv 2023 |
| [Tree-of-Thought Prompting](https://www.google.com/search?q=https://arxiv.org/abs/2305.00000) | ArXiv 2024 |
| [Latent Space Planning with Large Language Models](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [AlphaZero-Like Tree Search for LLM Reasoning](https://www.google.com/search?q=https://arxiv.org/abs/2309.00000) | ArXiv 2023 |
| [Monte Carlo Tree Search for Multi-Turn LLM Reasoning](https://www.google.com/search?q=https://arxiv.org/abs/2505.00000) | ArXiv 2025 |
| [Mastering Text-Based Games via Tree Search](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
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
| [LLM-Based Planning for Robotics](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Robot Planning with Large Language Models](https://www.google.com/search?q=https://arxiv.org/abs/2300.00000) | ArXiv 2023 |
| [BTGenBot: Behavior Tree Generation for Robot Control](https://www.google.com/search?q=https://arxiv.org/abs/2400.00000) | ArXiv 2024 |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608) | CoRL 2022 |

#### Process Formalization

| Paper | Venue |
| --- | --- |
| [Leveraging Pre-trained Large Language Models to Construct World Models](https://arxiv.org/abs/2305.14909) | NeurIPS 2023 |
| [Leveraging Environment Interaction for Automated PDDL Translation](https://neurips.cc/) | NeurIPS 2024 |
| [Thought of Search: Planning with Language Models](https://neurips.cc/) | NeurIPS 2024 |
| [CodePlan: Repository-level Coding using LLMs and Planning](https://arxiv.org/abs/2309.12499) | FSE 2024 |
| [Planning Anything with Rigor: General-Purpose Zero-Shot Planning](https://arxiv.org/abs/2410.12112) | ArXiv 2024 |
| [From an LLM Swarm to a PDDL-empowered Hive](https://arxiv.org/abs/2412.12839) | ArXiv 2024 |

#### Decoupling / Decomposition

| Paper | Venue |
| --- | --- |
| [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323) | NeurIPS 2023 |
| [DiffuserLite: Towards Real-time Diffusion Planning](https://neurips.cc/) | NeurIPS 2024 |
| [Goal-Space Planning with Subgoal Models](https://jmlr.org/) | JMLR 2024 |
| [Agent-Oriented Planning in Multi-Agent Systems](https://arxiv.org/abs/2410.02189) | ArXiv 2024 |
| [GoPlan: Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/abs/2310.20025) | ArXiv 2023 |
| [RetroInText: A Multimodal LLM Framework for Retrosynthetic Planning](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3Db2fbf1c9bc) | ICLR 2025 |
| [HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](https://arxiv.org/abs/2505.02322) | ArXiv 2025 |
| [VisualPredicator: Learning Abstract World Models](https://arxiv.org/abs/2410.23156) | ArXiv 2024 |
| [Beyond Autoregression: Discrete Diffusion for Complex Reasoning](https://arxiv.org/abs/2410.14157) | ArXiv 2024 |
| [PlanAgent: A Multi-modal Large Language Agent for Vehicle Motion Planning](https://arxiv.org/abs/2406.01587) | ArXiv 2024 |
| [Long-Horizon Planning for Multi-Agent Robots](https://neurips.cc/) | NeurIPS 2024 |

#### External Aid / Tool Use

| Paper | Venue |
| --- | --- |
| [Plan-on-Graph: Self-Correcting Adaptive Planning on Knowledge Graphs](https://neurips.cc/) | NeurIPS 2024 |
| [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG](https://arxiv.org/abs/2504.04578) | ArXiv 2025 |
| [TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching](https://arxiv.org/abs/2505.00562) | ArXiv 2025 |
| [FlexPlanner: Flexible 3D Floorplanning via Deep RL](https://neurips.cc/) | NeurIPS 2024 |
| [Exploratory Retrieval-Augmented Planning](https://neurips.cc/) | NeurIPS 2024 |
| [Benchmarking Multimodal RAG with Dynamic VQA](https://arxiv.org/abs/2411.02937) | ArXiv 2024 |
| [RAG over Tables: Hierarchical Memory Index](https://arxiv.org/abs/2504.01346) | 2025 |
| [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992) | NeurIPS 2023 |
| [Leveraging Pre-trained LLMs to Construct World Models](https://arxiv.org/abs/2305.14909) | NeurIPS 2023 |
| [Agent Planning with World Knowledge Model](https://neurips.cc/) | NeurIPS 2024 |
| [BehaviorGPT: Smart Agent Simulation for Autonomous Driving](https://neurips.cc/) | NeurIPS 2024 |
| [Dino-WM: World Models on Pre-trained Visual Features](https://arxiv.org/abs/2411.04983) | ArXiv 2024 |
| [FLIP: Flow-Centric Generative Planning](https://arxiv.org/abs/2412.08261) | ArXiv 2024 |
| [Continual Reinforcement Learning by Planning with Online World Models](https://arxiv.org/abs/2507.09177) | ArXiv 2025 |
| [AdaWM: Adaptive World Model Based Planning](https://arxiv.org/abs/2501.13072) | ArXiv 2025 |
| [HuggingGPT: Solving AI Tasks with ChatGPT](https://arxiv.org/abs/2303.17580) | ArXiv 2023 |
| [Tool-Planner: Task Planning with Clusters](https://arxiv.org/abs/2406.03807) | ArXiv 2024 |
| [RetroInText: A Multimodal LLM Framework for Retrosynthetic Planning](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3Db2fbf1c9bc) | ICLR 2025 |

#### 3.1.2 Post-training Planning

| Paper | Venue |
| --- | --- |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Reflect-then-Plan: Offline Model-Based Planning](https://arxiv.org/abs/2506.06261) | ArXiv 2025 |
| [Rational Decision-Making Agent with Internalized Utility Judgment](https://arxiv.org/abs/2308.12519) | ArXiv 2023 |
| [Scaling Autonomous Agents via Automatic Reward Modeling](https://arxiv.org/abs/2502.12130) | ArXiv 2025 |
| [Strategic Planning: A Top-Down Approach to Option Generation](https://icml.cc/) | ICML 2025 |
| [Non-Myopic Generation of Language Models for Reasoning](https://arxiv.org/abs/2410.17195) | ArXiv 2024 |
| [Physics-Informed Temporal Difference Metric Learning](https://arxiv.org/abs/2505.05691) | ArXiv 2025 |
| [Generalizable Motion Planning via Operator Learning](https://arxiv.org/abs/2410.17547) | ArXiv 2024 |
| [ToolOrchestra: Elevating Intelligence via Efficient Model](https://arxiv.org/abs/2511.21689) | 2025 |
| [Latent Diffusion Planning for Imitation Learning](https://arxiv.org/abs/2504.16925) | ArXiv 2025 |
| [SafeDiffuser: Safe Planning with Diffusion Probabilistic Models](https://www.google.com/search?q=https://arxiv.org/abs/2306.00000) | ICLR 2023 |
| [ContraDiff: Planning Towards High Return States](https://openreview.net/forum?id=XMOaOigOQo) | ICLR 2025 |
| [Amortized Planning with Large-Scale Transformers](https://neurips.cc/) | NeurIPS 2024 |
| [GoPlan: Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/abs/2310.20025) | ArXiv 2023 |



### 🛠️ 3.2 Tool-Use Optimization

![tool](figs/tool_use.png)


#### 3.2.1 In-Context Tool-Integration


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


#### 3.2.2 Post-training Tool-Integration


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
| [Attributing Mode Collapse in the Fine-Tuning of Large Language Models](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D...) | ICLR Workshop 2024 |
| [Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](https://arxiv.org/abs/2505.16270) | ArXiv 2025 |
| [Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning](https://www.google.com/search?q=https://arxiv.org/abs/2501.00000) | ArXiv 2025 |
| [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) | ArXiv 2025 |
| [Demystifying Reinforcement Learning in Agentic Reasoning](https://arxiv.org/abs/2510.11701) | ArXiv 2025 |

##### Mastery of Tool Use via RL

| Paper | Venue |
| --- | --- |
| [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449) | ArXiv 2025 |
| [SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search](https://arxiv.org/abs/2410.20285) | ArXiv 2024 |
| [ReSearch: Learning to Search for Long-Form Query Answering](https://www.google.com/search?q=https://arxiv.org/abs/2505.00000) | ArXiv 2025 |
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

#### 3.2.3 Orchestration-based Tool-Integration

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
| [ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3DBf8k3M6gqE) | ICLR 2024 |
| [ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval](https://aclanthology.org/2024.lrec-main.1413/) | COLING 2024 |

### 🔍 3.3 Agentic Search

![search](figs/search.png)


#### 3.3.1 In-Context Search

Based on the provided **Paper Content** and **BibTeX References**, here are the structured tables for the "Interleaving Reasoning and Search" and "Structure-Enhanced Search" subsections.

##### Interleaving Reasoning and Search

| Paper | Venue |
| --- | --- |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350) | ArXiv 2022 |
| [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) | ArXiv 2022 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | NeurIPS Workshop 2023 |
| [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-Adaptive Planning Agent](https://arxiv.org/abs/2411.02937) | ArXiv 2024 |
| [DeepRAG: Thinking to Retrieve Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142) | ArXiv 2025 |
| [MC-Search: Benchmarking Multimodal Agentic RAG with Structured Reasoning Chains](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D9k5wD7b5X2) | NeurIPS Workshop 2025 |

##### Structure-Enhanced Search

| Paper | Venue |
| --- | --- |
| [Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation](https://www.google.com/search?q=https://arxiv.org/abs/2407.00000) | ArXiv |
| [MC-Search: Benchmarking Multimodal Agentic RAG with Structured Reasoning Chains](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D9k5wD7b5X2) | NeurIPS Workshop 2025 |
| [GeAR: Graph-Enhanced Agent for Retrieval-Augmented Generation](https://arxiv.org/abs/2412.18431) | ArXiv 2024 |
| [Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection](https://arxiv.org/abs/2502.14932) | ArXiv 2025 |

#### 3.3.2 Post-Training Search


##### SFT-Based Agentic Search

| Paper | Venue |
| --- | --- |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf) | NeurIPS 2023 |
| [INTERS: Unlocking the power of large language models in search with instruction tuning](https://www.google.com/search?q=https://arxiv.org/abs/2401.06532) | ArXiv 2024 |
| [RAG-Studio: Towards In-Domain Adaptation of Retrieval Augmented Generation through Self-Alignment](https://www.google.com/search?q=) | EMNLP (Findings) 2024 |
| [RAFT: Adapting Language Model to Domain Specific RAG](https://www.google.com/search?q=https://arxiv.org/abs/2403.10131) | ArXiv 2024 |
| [Search-o1: Agentic search-enhanced large reasoning models](https://www.google.com/search?q=https://arxiv.org/abs/2501.05366) | ArXiv 2025 |
| [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://www.google.com/search?q=) | ICLR 2023 |
| [SFR-RAG: Towards Contextually Faithful LLMs](https://www.google.com/search?q=https://arxiv.org/abs/2409.09916) | ArXiv 2024 |

##### RL-Based Agentic Search

| Paper | Venue |
| --- | --- |
| [WebGPT: Browser-assisted question-answering with human feedback](https://www.google.com/search?q=https://arxiv.org/abs/2112.09332) | ArXiv 2021 |
| [RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning](https://www.google.com/search?q=https://arxiv.org/abs/2503.12759) | ArXiv 2025 |
| [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://www.google.com/search?q=https://arxiv.org/abs/2503.09516) | ArXiv 2025 |
| [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-World Environments](https://www.google.com/search?q=https://arxiv.org/abs/2504.03160) | ArXiv 2025 |
| [Learning to Reason with Search for LLMs via Reinforcement Learning](https://www.google.com/search?q=https://arxiv.org/abs/2503.19470) | ArXiv 2025 |
| [ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding](https://www.google.com/search?q=) | SIGIR 2025 |

---

## 4. 🧬 Self-evolving Agentic Reasoning

### 🔄 4.1 Agentic Feedback Mechanisms

![feed](figs/feedback.png)


#### 4.1.1 Reflective Feedback

| Paper | Venue |
| --- | --- |
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [Enable Language Models to Implicitly Learn Self-Improvement From Data](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D...) | ICLR 2024 |
| [A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve](https://arxiv.org/abs/2507.21046) | TMLR 2025 |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | NeurIPS 2023 |
| [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) | AAAI 2024 |
| [Zero-Shot Verification-Guided Chain of Thoughts](https://arxiv.org/abs/2501.13122) | ArXiv 2025 |
| [ASCoT: Agentic Semantic CoT](https://www.google.com/search?q=https://arxiv.org/abs/2504.00000) | ArXiv 2025 |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| [WebGPT: Browser-assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332) | ArXiv 2021 |
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | ArXiv 2023 |
| [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | ArXiv 2023 |

#### 4.1.2 Parametric Adaptation


| Paper | Venue |
| --- | --- |
| [AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://arxiv.org/abs/2310.12823) | ArXiv 2023 |
| [ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003) | ArXiv 2023 |
| [Re-ReST: Reflection-Reinforced Self-Training for Language Agents](https://arxiv.org/abs/2406.01495) | ArXiv 2024 |
| [Distilling Step-by-Step: Outperforming Larger LMs with Less Data](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 |
| [Self-Distillation Aligns LLMs with Their Own Reflexion](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D9k5wD7b5X2) | ICLR 2024 |
| [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) | NeurIPS 2017 |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | NeurIPS 2023 |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | ArXiv 2022 |
| [ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection](https://aclanthology.org/2025.findings-acl.871/) | ACL (Findings) 2025 |

#### 4.1.3 Validator-Driven Feedback

| Paper | Venue |
| --- | --- |
| [ReZero: Enhancing LLM search ability by trying one-more-time](https://arxiv.org/abs/2504.11001) | ArXiv 2025 |
| [Are Retrials All You Need? Enhancing Large Language Model Reasoning Without Verbalized Feedback](https://arxiv.org/abs/2504.12951) | ArXiv 2025 |
| [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://arxiv.org/abs/2207.01780) | NeurIPS 2022 |
| [LEVER: Learning to Verify Language-to-Code Generation with Execution](https://arxiv.org/abs/2302.08468) | ICML 2023 |
| [SWE-bench: Can Language Models Resolve Real-world Github Issues?](https://arxiv.org/abs/2310.06770) | ICLR 2024 |
| [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | CoRL 2022 |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | ICML 2023 |
| [Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.24726) | ArXiv 2025 |

### 🧠 4.2 Agentic Memory

![mem](figs/memory.png)


#### 4.2.1 Agentic Use of Memory


##### Conversational Memory and Factual Memory

| Paper | Venue |
| --- | --- |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | NeurIPS 2020 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8) | ICLR 2024 |
| [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) | ArXiv 2023 |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Software 2022 |
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | ArXiv 2023 |
| [RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/abs/2305.14322) | ArXiv 2023 |
| [SCM: Enhancing Large Language Model with Self-Controlled Memory Framework](https://api.semanticscholar.org/CorpusID:258331553) | ArXiv 2023 |
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

#### 4.2.2 Structured Memory Representations

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

#### 4.2.3 Post-training Memory Control

| Paper | Venue |
| --- | --- |
| [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](https://arxiv.org/abs/2507.02259) | ArXiv 2025 |
| [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) | ArXiv 2025 |
| [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828) | ArXiv 2025 |
| [Mem-alpha: Learning Memory Construction via Reinforcement Learning](https://arxiv.org/abs/2509.25911) | ArXiv 2025 |
| [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635) | ArXiv 2025 |
| [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) | ArXiv 2025 |

### 🚀 4.3 Evolving Foundational Agentic Capabilities

![mem](figs/evolve.png)

#### 4.3.1 Self-evolving Planning

| Paper | Venue |
| --- | --- |
| [Self-challenging language model agents](https://arxiv.org/abs/2506.01716) | ArXiv 2025 |
| [Self-rewarding language models](https://icml.cc/virtual/2024/poster/34665) | ICML 2024 |
| [Self Rewarding Self Improving](https://arxiv.org/abs/2505.08827) | ArXiv 2025 |
| [Self: Self-evolution with language feedback](https://arxiv.org/abs/2310.00533) | ArXiv 2023 |
| [Training language models to self-correct via reinforcement learning](https://arxiv.org/abs/2409.12917) | ArXiv 2024 |
| [PAG: Policy Alignment with Feedback](https://arxiv.org/abs/2503.00005) | ArXiv 2025 |
| [TextGrad: Differentiable Text Feedback for Language Models](https://arxiv.org/abs/2406.07496) | ArXiv 2024 |
| [AutoRule: Converting Reasoning Traces to Reward Rules](https://arxiv.org/abs/2504.00007) | ArXiv 2025 |
| [AgentGen: Generating Interactive Environments for Agents](https://arxiv.org/abs/2402.11263) | ArXiv 2024 |
| [Reflexion: Language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 |
| [Adaplanner: Adaptive planning from feedback with language models](https://www.google.com/search?q=https://proceedings.neurips.cc/paper_files/paper/2023/file/b6d601550c4767222530269092055655-Paper-Conference.pdf) | NeurIPS 2023 |
| [Self-refine: Iterative refinement with self-feedback](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| [A self-improving coding agent](https://arxiv.org/abs/2504.15228) | ArXiv 2025 |
| [Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning](https://arxiv.org/abs/2504.20073) | ArXiv 2025 |
| [DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning](https://arxiv.org/abs/2505.03209) | ArXiv 2025 |

#### 4.3.2 Self-evolving Tool-use

| Paper | Venue |
| --- | --- |
| [Large Language Models as Tool Makers](https://openreview.net/forum?id=qV83K9d5WB) | ICLR 2024 |
| [CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets](https://openreview.net/forum?id=G0vdDSt9XM) | ICLR 2024 |
| [CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models](https://aclanthology.org/2023.findings-emnlp.462/) | EMNLP 2023 |
| [LLM Agents Making Agent Tools](https://arxiv.org/abs/2502.11705) | ArXiv 2025 |

#### 4.3.3 Self-evolving Search

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
