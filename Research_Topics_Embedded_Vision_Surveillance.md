# Novel Research Topics in Embedded Vision Systems for Smart Surveillance

> **Domain:** Embedded Systems · Edge AI · Computer Vision  
> **Focus:** Smart Surveillance on Embedded Hardware (Jetson, FPGA, Raspberry Pi, RISC-V SoC)  
> **Publication Target:** IEEE Transactions / IEEE Conferences (2024–2026 trends)

---

## Table of Contents

1. [Research Topic 1 – TinyML-Based Anomaly Detection with On-Device Continual Learning](#1-tinyml-based-anomaly-detection-with-on-device-continual-learning)
2. [Research Topic 2 – Federated Split Learning for Multi-Camera Surveillance Networks](#2-federated-split-learning-for-multi-camera-surveillance-networks)
3. [Research Topic 3 – NAS-Optimized Object Detection for Ultra-Low-Power RISC-V SoCs](#3-nas-optimized-object-detection-for-ultra-low-power-risc-v-socs)
4. [Research Topic 4 – Homomorphic Encryption-Enabled Privacy-Preserving Face Detection at the Edge](#4-homomorphic-encryption-enabled-privacy-preserving-face-detection-at-the-edge)
5. [Research Topic 5 – Event-Driven Neuromorphic Surveillance with Dynamic Vision Sensors](#5-event-driven-neuromorphic-surveillance-with-dynamic-vision-sensors)
6. [Research Topic 6 – FPGA-Accelerated Transformer Inference for Real-Time Crowd Analytics](#6-fpga-accelerated-transformer-inference-for-real-time-crowd-analytics)
7. [Research Topic 7 – Explainable Lightweight CNNs for Legally Compliant Surveillance Decisions](#7-explainable-lightweight-cnns-for-legally-compliant-surveillance-decisions)
8. [Research Topic 8 – Adaptive Neural Architecture Switching for Energy-Harvested Surveillance Nodes](#8-adaptive-neural-architecture-switching-for-energy-harvested-surveillance-nodes)
9. [Research Topic 9 – Differential Privacy Federated Learning for Distributed Pedestrian Re-Identification](#9-differential-privacy-federated-learning-for-distributed-pedestrian-re-identification)
10. [Research Topic 10 – Hardware-Software Co-Design for Spiking Neural Network Surveillance Accelerators](#10-hardware-software-co-design-for-spiking-neural-network-surveillance-accelerators)
11. [Research Topic 11 – Multi-Modal Edge Fusion of RGB and Thermal Streams for Night Surveillance](#11-multi-modal-edge-fusion-of-rgb-and-thermal-streams-for-night-surveillance)
12. [Research Topic 12 – Secure Multi-Party Computation for Collaborative Cross-Camera Tracking](#12-secure-multi-party-computation-for-collaborative-cross-camera-tracking)
13. [Research Topic 13 – Self-Supervised Domain Adaptation for Surveillance Cameras Under Distribution Shift](#13-self-supervised-domain-adaptation-for-surveillance-cameras-under-distribution-shift)
14. [Research Topic 14 – RISC-V Custom ISA Extensions for Real-Time Pose Estimation in Edge Surveillance](#14-risc-v-custom-isa-extensions-for-real-time-pose-estimation-in-edge-surveillance)
15. [Research Topic 15 – Zero-Shot Anomaly Detection via Foundation Model Distillation on Embedded Devices](#15-zero-shot-anomaly-detection-via-foundation-model-distillation-on-embedded-devices)
16. [Top 5 Highest-Impact Topics](#top-5-highest-impact-topics)

---

## 1. TinyML-Based Anomaly Detection with On-Device Continual Learning

**Title:**  
*AdaptiveTinyWatch: Continual-Learning TinyML Framework for Real-Time Anomaly Detection on Microcontroller-Class Surveillance Nodes*

**Problem Statement:**  
Current TinyML surveillance models are trained once and deployed statically. As scene conditions evolve (lighting, crowd density, seasonal changes), model accuracy degrades significantly without expensive cloud-based retraining. No existing framework enables on-device continual learning under strict SRAM/Flash constraints (< 512 KB) while avoiding catastrophic forgetting.

**Novel Contribution:**  
- A memory-efficient elastic weight consolidation (EWC) variant tailored for MCU-class devices (Cortex-M7, ESP32-S3).
- A task-boundary detection module that triggers selective micro-fine-tuning using synthetic replay buffers stored in compressed form.
- Demonstrated on a custom benchmark of 6 surveillance scenarios with only 256 KB SRAM overhead.

**Why It Is Important:**  
Surveillance environments change continuously. Enabling edge nodes to self-adapt without cloud dependency reduces operational cost, latency, and privacy risk. Aligns with 2024–2026 IEEE trends toward autonomous edge intelligence.

---

## 2. Federated Split Learning for Multi-Camera Surveillance Networks

**Title:**  
*SplitFedSurv: Communication-Efficient Federated Split Learning Across Heterogeneous Camera Edge Nodes*

**Problem Statement:**  
Federated learning (FL) applied to multi-camera surveillance suffers from high communication overhead when camera nodes (Raspberry Pi, Jetson Nano) have limited bandwidth. Standard FL requires uploading full model gradients; split learning requires continuous activation exchange. Neither is optimized for the heterogeneous compute-communication tradeoffs typical in surveillance deployments.

**Novel Contribution:**  
- A dynamic split-point selection algorithm that adapts the layer at which the model is split based on real-time bandwidth and compute budgets of each node.
- A gradient sparsification + quantization pipeline reducing inter-node traffic by 73% with < 1% accuracy loss on person re-identification tasks.
- Validated on a 12-node heterogeneous testbed (Mix of Jetson Nano, Raspberry Pi 4, and simulated MCU nodes).

**Why It Is Important:**  
Multi-camera networks are the backbone of city-scale surveillance. Communication-efficient FL enables privacy-preserving collaborative intelligence without centralizing raw video, directly addressing GDPR and data sovereignty concerns.

---

## 3. NAS-Optimized Object Detection for Ultra-Low-Power RISC-V SoCs

**Title:**  
*RVNASDetect: Hardware-Aware Neural Architecture Search for Sub-10mW Object Detection on RISC-V AI SoCs*

**Problem Statement:**  
Designing object detection networks for RISC-V-based AI SoCs (e.g., GreenWaves GAP9, SiFive) is currently a manual, expert-driven process that rarely achieves Pareto-optimal accuracy-vs-energy tradeoffs. Existing NAS approaches target GPU/TPU latency proxies that are misleading on in-order RISC-V pipelines.

**Novel Contribution:**  
- A RISC-V hardware-accurate latency model integrated into a differentiable NAS search space.
- A multi-objective evolutionary search that simultaneously minimizes FLOPs, SRAM footprint, and cycle-accurate latency on RISC-V simulation.
- Produces a family of architectures (RVNASDetect-S/M/L) achieving 42% lower energy than MobileNetV3-SSD at equivalent mAP on VisDrone.

**Why It Is Important:**  
RISC-V is emerging as the open-source standard ISA for edge AI chips. NAS-generated architectures specifically targeting RISC-V pipelines can dramatically lower the barrier for deploying surveillance AI on cost-effective open-hardware platforms.

---

## 4. Homomorphic Encryption-Enabled Privacy-Preserving Face Detection at the Edge

**Title:**  
*HE-FaceEdge: Practical Partially Homomorphic Inference for Face Detection Without Plaintext Pixel Exposure*

**Problem Statement:**  
Face detection in public surveillance raises severe privacy concerns. Current approaches either process plaintext images (full privacy breach) or rely on trusted execution environments (TEEs) that are vulnerable to side-channel attacks. Fully homomorphic encryption (FHE) is too computationally prohibitive for real-time edge inference.

**Novel Contribution:**  
- A novel hybrid scheme combining CKKS-based partial HE for the first convolutional layers (sensitive pixel processing) with plaintext inference for deeper layers operating on encrypted feature maps.
- A custom FPGA (Xilinx Zynq UltraScale+) accelerator for the HE polynomial multiplication bottleneck achieving 18× speedup over ARM Cortex-A53 software execution.
- Achieves 12 fps face detection with cryptographic pixel privacy guarantees and < 3% accuracy degradation vs. plaintext baseline.

**Why It Is Important:**  
As surveillance regulations tighten globally, cryptographic privacy guarantees will become mandatory. This work provides a practical path toward legally compliant edge surveillance that does not require trust in the hardware operator.

---

## 5. Event-Driven Neuromorphic Surveillance with Dynamic Vision Sensors

**Title:**  
*NeuroSurv: Sparse Spiking Neural Network Inference on Dynamic Vision Sensor Streams for Ultra-Low-Latency Intrusion Detection*

**Problem Statement:**  
Frame-based cameras waste energy processing redundant background pixels and introduce latency proportional to frame rate. Dynamic Vision Sensors (DVS/event cameras) generate sparse, microsecond-resolution event streams, but existing surveillance algorithms cannot natively process asynchronous event data on embedded platforms.

**Novel Contribution:**  
- A sparse spiking neural network (SNN) architecture co-designed with the event-stream data format, implemented on Intel Loihi 2 and mapped to an FPGA surrogate for cost-effective deployment.
- An event-to-feature encoding pipeline that preserves temporal fine structure with 94% fewer memory accesses than frame-based alternatives.
- Demonstrated at < 5 mW average power for continuous intrusion detection with 0.8 ms detection latency.

**Why It Is Important:**  
Event cameras represent the next generation of vision sensors for surveillance. Matching them with neuromorphic computing creates an end-to-end bio-inspired surveillance pipeline that is orders of magnitude more energy-efficient than conventional approaches.

---

## 6. FPGA-Accelerated Transformer Inference for Real-Time Crowd Analytics

**Title:**  
*FPGATransCrowd: Efficient Vision Transformer Acceleration on Mid-Range FPGAs for Real-Time Crowd Density Estimation and Behavior Analysis*

**Problem Statement:**  
Vision Transformers (ViT) achieve state-of-the-art accuracy for crowd analytics but require billions of MACs and GB of memory—far beyond GPU-free embedded systems. No existing FPGA implementation of ViT achieves real-time (≥ 25 fps) throughput on mid-range devices (Xilinx ZCU104, Intel Cyclone V) while maintaining < 5% accuracy loss from quantization.

**Novel Contribution:**  
- A block-sparse attention mechanism that prunes attention heads based on spatial locality in surveillance scenes, reducing attention FLOPs by 61%.
- A custom 4-bit integer attention datapath with fused softmax-normalization on FPGA DSP slices, achieving 28 fps on ZCU104.
- A crowd behavior recognition dataset (SurveillanceCrowd-24) of 50,000 annotated clips collected from 8 public cameras.

**Why It Is Important:**  
Crowd analytics is critical for public safety events, stadium management, and pandemic response. FPGA-based solutions offer deterministic real-time latency without cloud dependency, making them ideal for mission-critical deployments.

---

## 7. Explainable Lightweight CNNs for Legally Compliant Surveillance Decisions

**Title:**  
*XAI-Surv: Integrated Gradient Attribution Maps for Real-Time Explainability of Edge-Deployed Surveillance CNNs Under EU AI Act Constraints*

**Problem Statement:**  
Surveillance AI systems used for high-stakes decisions (access control, threat detection) will be subject to the EU AI Act's explainability requirements by 2026. Existing XAI methods (Grad-CAM, LIME, SHAP) are computationally too expensive for real-time edge inference and do not produce legally auditable decision trails.

**Novel Contribution:**  
- A training-time saliency module integrated directly into lightweight CNN architectures (EfficientDet-Lite, NanoDet) that generates attribution maps at no additional inference cost.
- A cryptographically signed decision-log format that packages the inference result, bounding boxes, and attribution heatmap into a tamper-evident audit record.
- Tested on NVIDIA Jetson Orin Nano achieving 30 fps detection with simultaneous explanation generation at 2.8W average power.

**Why It Is Important:**  
Legal compliance is becoming a hard constraint for surveillance system deployment. Embedding explainability into edge inference eliminates the audit latency gap and enables real-time operator oversight, which is a concrete market and regulatory requirement.

---

## 8. Adaptive Neural Architecture Switching for Energy-Harvested Surveillance Nodes

**Title:**  
*HarvestAdaptSurv: Runtime Neural Architecture Switching Driven by Energy-Harvesting State for Battery-Free Surveillance Nodes*

**Problem Statement:**  
Solar or RF energy-harvested surveillance nodes have highly variable available power. Fixed neural architectures either over-provision (wasting harvested energy) or under-provision (missing detections during high-energy periods). No existing framework links harvested energy state directly to inference model complexity selection.

**Novel Contribution:**  
- A once-for-all (OFA) network family where sub-network depth and width are selected at runtime based on a supercapacitor charge-level sensor reading.
- A Markov decision process (MDP) formulation for architecture selection that maximizes long-horizon detection coverage under stochastic energy arrivals.
- Prototype on a solar-harvested Raspberry Pi Zero 2W achieving 3.4× more detections per harvested joule versus a fixed MobileNetV2 baseline.

**Why It Is Important:**  
Deploying maintenance-free surveillance nodes in infrastructure-sparse environments (forests, borders, disaster zones) requires sustainable operation. Energy-adaptive inference is the critical missing link for truly autonomous surveillance nodes.

---

## 9. Differential Privacy Federated Learning for Distributed Pedestrian Re-Identification

**Title:**  
*DP-FedReID: Differentially Private Federated Re-Identification Across Non-IID Camera Domains Without Central Data Aggregation*

**Problem Statement:**  
Pedestrian re-identification (Re-ID) across cameras requires sharing appearance features that can reconstruct biometric information. Existing centralized Re-ID models require raw video aggregation; FL-based Re-ID approaches do not provide formal privacy guarantees and fail under severe non-IID data distributions across cameras with different viewpoints and lighting.

**Novel Contribution:**  
- A (ε, δ)-differentially private FL protocol for Re-ID with per-camera adaptive clipping norms that account for non-IID data heterogeneity.
- A domain-adversarial training head added to the shared federated backbone that aligns inter-camera feature distributions without requiring raw data sharing.
- Demonstrates rank-1 accuracy within 4% of centralized baseline on Market-1501 and DukeMTMC under ε = 2.0 privacy budget.

**Why It Is Important:**  
Biometric Re-ID at city scale is one of the most privacy-sensitive surveillance tasks. Providing formal differential privacy guarantees while preserving accuracy is essential for regulatory acceptance and ethical deployment.

---

## 10. Hardware-Software Co-Design for Spiking Neural Network Surveillance Accelerators

**Title:**  
*SNN-SurvASIC: An ASIC Micro-Architecture for Always-On Spiking Neural Network Surveillance with Sub-1mW Idle Power*

**Problem Statement:**  
Always-on surveillance requires chips that consume near-zero power during scene inactivity. Conventional CNN accelerators burn static power even when idle. Spiking neural networks (SNNs) are inherently sparse and event-driven but lack a co-designed ASIC micro-architecture optimized for surveillance workloads.

**Novel Contribution:**  
- A novel neuron tile architecture with gated clock domains that achieve zero switching power during background-only frames.
- A hardware-software co-design flow that maps surveillance-specific SNN workloads to tiles using a compiler-guided spike routing algorithm.
- Tape-out in 28nm CMOS achieving 0.7 mW idle / 8.2 mW peak for continuous pedestrian detection at 15 fps.

**Why It Is Important:**  
The proliferation of millions of IoT surveillance endpoints makes per-device power consumption a global energy concern. Sub-milliwatt always-on detection chips would enable truly ubiquitous, battery-operated surveillance with multi-year lifetimes.

---

## 11. Multi-Modal Edge Fusion of RGB and Thermal Streams for Night Surveillance

**Title:**  
*ThermoRGBFuse: Adaptive Calibration-Free RGB-Thermal Fusion on Embedded GPUs for 24/7 Surveillance*

**Problem Statement:**  
Night surveillance degrades drastically for RGB-only cameras. Thermal cameras compensate but suffer from low resolution and lack texture detail. Fusing both modalities on embedded platforms requires accurate calibration that drifts in outdoor deployments, and current fusion architectures are too heavy for Jetson-class devices.

**Novel Contribution:**  
- A calibration-free implicit neural alignment module that learns cross-modal correspondences from scene geometry cues without explicit geometric calibration.
- A compact dual-encoder cross-attention fusion network running at 20 fps on Jetson Orin with INT8 quantization.
- A new night-surveillance dataset (NightFuseDB) with 30,000 temporally synchronized RGB-Thermal pairs from 4 outdoor cameras across 6 months.

**Why It Is Important:**  
Most criminal activity occurs at night. Reliable 24/7 embedded vision without cloud processing is a high-impact, commercially viable research direction that directly addresses a gap in current smart city deployments.

---

## 12. Secure Multi-Party Computation for Collaborative Cross-Camera Tracking

**Title:**  
*SMPCTrack: Privacy-Preserving Cross-Organizational Camera Collaboration via Secure Multi-Party Computation for Person Tracking*

**Problem Statement:**  
Effective city-scale tracking requires cameras owned by different organizations (police, transport, retail) to share tracking data. Legal and competitive constraints prevent sharing raw video or identity data. Secure Multi-Party Computation (SMPC) could enable collaborative tracking without data disclosure, but no practical system exists that meets real-time latency constraints.

**Novel Contribution:**  
- An SMPC protocol based on secret-shared garbled circuits specifically designed for the feature-matching step of cross-camera tracking.
- A hardware offload for the garbled circuit evaluation using FPGA co-processors, reducing the cryptographic overhead to < 40 ms per handoff event.
- A simulation framework, SMPCTrack-Sim, modeling 8-party tracking under realistic network topologies.

**Why It Is Important:**  
Inter-agency surveillance collaboration is legally and politically blocked in most jurisdictions precisely due to privacy concerns. SMPC provides a cryptographic solution that enables crime investigation capability while maintaining jurisdictional data boundaries.

---

## 13. Self-Supervised Domain Adaptation for Surveillance Cameras Under Distribution Shift

**Title:**  
*SSDA-Surv: Test-Time Self-Supervised Domain Adaptation for Surveillance Object Detectors Deployed Across Geographically Diverse Scenes*

**Problem Statement:**  
Surveillance models trained on one geographic region fail when deployed in visually different cities or climates. Manual re-annotation and retraining for each deployment site is cost-prohibitive. Test-time domain adaptation without labeled target data is an unsolved problem in surveillance.

**Novel Contribution:**  
- A test-time adaptation (TTA) pipeline using masked autoencoders as self-supervised auxiliary tasks that are run concurrently on idle CPU cores of an edge device.
- A confidence-weighted pseudo-label refinement loop that filters noisy labels from the auxiliary task before updating batch normalization statistics.
- Demonstrated on a 6-city cross-domain surveillance benchmark achieving 14.7% mAP improvement over no-adaptation baseline with zero manual labeling.

**Why It Is Important:**  
Large-scale smart city rollouts deploy thousands of cameras across heterogeneous environments. Eliminating per-site annotation cost through self-supervised adaptation dramatically reduces total deployment cost and makes AI-powered surveillance economically viable at scale.

---

## 14. RISC-V Custom ISA Extensions for Real-Time Pose Estimation in Edge Surveillance

**Title:**  
*PoseRV: Custom RISC-V Vector ISA Extensions for Sub-5ms Human Pose Estimation in Embedded Surveillance Applications*

**Problem Statement:**  
Human pose estimation is a key primitive for action recognition in surveillance (fall detection, fighting, trespassing). Existing SIMD/vector extensions (RISC-V V-extension) are designed for generic linear algebra, not for the specific heatmap regression and part affinity field computations in pose networks.

**Novel Contribution:**  
- Three custom RISC-V ISA extensions (RVPOSE): heatmap_max_pool, paf_integral, and skeleton_assemble, implemented in an open-source Rocket Chip tile.
- A compiler backend (LLVM) that automatically maps pose estimation operators to RVPOSE instructions.
- Achieves 4.2ms latency for single-person pose estimation on a 1 GHz RISC-V core—5.8× faster than baseline V-extension implementation.

**Why It Is Important:**  
RISC-V's open ISA allows domain-specific extensions without licensing costs. Pose estimation accelerators built on RISC-V represent the next generation of open, customizable edge AI chips for surveillance applications, positioning this research at the intersection of computer architecture and AI.

---

## 15. Zero-Shot Anomaly Detection via Foundation Model Distillation on Embedded Devices

**Title:**  
*FoundationEdgeSurv: Knowledge Distillation from Large Vision-Language Foundation Models to Sub-10M Parameter Anomaly Detectors for Edge Surveillance*

**Problem Statement:**  
Foundation models (CLIP, SAM, Grounding DINO) achieve outstanding zero-shot anomaly detection in surveillance scenarios but are orders of magnitude too large for edge deployment. Naive quantization and pruning destroy the semantic generalization capability that makes them useful for zero-shot scenarios. No distillation framework preserves cross-domain anomaly detection capability in a sub-10M parameter student.

**Novel Contribution:**  
- A semantic-preserving distillation loss that aligns the student's intermediate feature manifold with the teacher's vision-language joint embedding space.
- A progressive distillation curriculum that transfers scene-specific anomaly concepts from CLIP/Grounding DINO to a MobileNetV4-sized student network.
- Achieves 87.3% AUROC on the UCSD Anomaly, ShanghaiTech, and a new EdgeAnomalyDB benchmark, running at 22 fps on Raspberry Pi 5.

**Why It Is Important:**  
Zero-shot capability is essential for surveillance scenarios where labeled anomaly data is rare or impossible to collect in advance (e.g., novel threats, unseen behaviors). Distilling this capability to edge devices eliminates cloud dependency for anomaly alerts, reducing latency from seconds to milliseconds.

---

## Top 5 Highest-Impact Topics

The following five topics are ranked by their combined research novelty, practical deployment impact, regulatory relevance, and publication potential in top-tier IEEE venues (IEEE Transactions on Image Processing, IEEE TNNLS, IEEE IoT Journal, DATE, CVPR Embedded AI workshops):

| Rank | Topic | Key Impact Driver |
|------|-------|------------------|
| 🥇 1 | **[Topic 4] HE-FaceEdge** – Homomorphic Encryption for Privacy-Preserving Face Detection | Addresses the most critical legal/ethical barrier in surveillance AI; unique FPGA+HE co-design angle; aligns with EU AI Act and global biometric regulations |
| 🥈 2 | **[Topic 15] FoundationEdgeSurv** – Foundation Model Distillation for Zero-Shot Anomaly Detection | Rides the 2024–2026 foundation model wave; zero-shot generalization on embedded hardware is an open, highly cited problem space |
| 🥉 3 | **[Topic 5] NeuroSurv** – Event-Driven Neuromorphic Surveillance with DVS | Combines two emerging hardware paradigms (neuromorphic chips + event cameras); < 5mW power is a hard-to-match claim; strong IEEE Sensors/DATE target |
| 4 | **[Topic 2] SplitFedSurv** – Federated Split Learning for Multi-Camera Networks | Federated learning for surveillance is a rapidly growing field; the dynamic split-point novelty differentiates from existing FL-surveillance work |
| 5 | **[Topic 14] PoseRV** – RISC-V Custom ISA Extensions for Pose Estimation | RISC-V AI is a hot topic in 2024–2026; custom ISA extensions for a specific CV task is novel; strong IEEE MICRO / ISLPED target |

### Why These Five Lead

1. **Regulatory alignment:** Topics 4 and 7 directly address the EU AI Act (2024 enforcement), GDPR, and emerging US AI regulations—making them immediately fundable and publishable.
2. **Emerging hardware paradigms:** Topics 5 and 14 leverage hardware platforms (neuromorphic, RISC-V) that are gaining rapid industry adoption, ensuring long citation tails.
3. **Foundation model integration:** Topic 15 bridges the most impactful AI trend (large foundation models) with the most constrained deployment environment (embedded edge), a combination that commands high attention in 2025–2026 venues.
4. **Quantifiable novelty:** All five present hard, measurable contributions (power numbers, accuracy benchmarks, latency figures) that reviewers can evaluate concretely.
5. **Reproducibility:** All five are feasible on commercially available hardware (Zynq UltraScale+, Raspberry Pi 5, Jetson Orin, Intel Loihi 2 evaluation boards, open-source RISC-V simulators).

---

## Summary Table of All 15 Topics

| # | Title (Short) | Hardware Platform | Key Technique | IEEE Target Venue |
|---|---------------|------------------|---------------|-------------------|
| 1 | AdaptiveTinyWatch | Cortex-M7 / ESP32-S3 | Continual Learning + EWC | IEEE IoT Journal |
| 2 | SplitFedSurv | Jetson Nano / RPi 4 | Federated Split Learning | IEEE TNNLS |
| 3 | RVNASDetect | RISC-V GAP9 / SiFive | Hardware-Aware NAS | IEEE TCAS-II |
| 4 | HE-FaceEdge | Zynq UltraScale+ | Homomorphic Encryption + FPGA | IEEE TIFS |
| 5 | NeuroSurv | Intel Loihi 2 / FPGA | Event Camera + SNN | IEEE Sensors Journal |
| 6 | FPGATransCrowd | Xilinx ZCU104 | Sparse ViT + FPGA | IEEE TCSVT |
| 7 | XAI-Surv | Jetson Orin Nano | Integrated Gradients + Audit Log | IEEE Access |
| 8 | HarvestAdaptSurv | RPi Zero 2W (Solar) | OFA Network + MDP | IEEE IoT Journal |
| 9 | DP-FedReID | Jetson Nano Cluster | DP-FL + Domain Adversarial | IEEE TIFS |
| 10 | SNN-SurvASIC | 28nm ASIC (tape-out) | SNN ASIC Co-Design | IEEE ISSCC / DATE |
| 11 | ThermoRGBFuse | Jetson Orin | Calibration-Free Fusion | IEEE TIP |
| 12 | SMPCTrack | FPGA + CPU Cluster | Secure Multi-Party Computation | IEEE S&P / CCS |
| 13 | SSDA-Surv | Jetson AGX Xavier | Test-Time Self-Supervised DA | IEEE CVPR Workshop |
| 14 | PoseRV | RISC-V Rocket Chip | Custom ISA Extensions | IEEE MICRO / ISLPED |
| 15 | FoundationEdgeSurv | Raspberry Pi 5 | Foundation Model Distillation | IEEE TPAMI |

---

*Generated for Seminar – Embedded Vision Systems for Smart Surveillance (2024–2026 research directions)*
