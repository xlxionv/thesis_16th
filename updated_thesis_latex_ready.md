**VIETNAM NATIONAL UNIVERSITY -- HO CHI MINH CITY**

**INTERNATIONAL UNIVERSITY**

**SCHOOL OF INDUSTRIAL ENGINEERING AND MANAGEMENT**


**PARALLEL-LINE CAPACITATED LOT-SIZING AND SCHEDULING PROBLEM WITH SEQUENCE_DENPENDENT SETUPS AND PREVENTIVE MAINTENANCE: A CASE STUDY OF LOOPSET MANUFACTURING**

Submitted in partial fulfilment of the requirements for the Degree of Bachelor of Engineering in Logistics and Supply Chain Management

**Student: TRƯƠNG NGỌC TUYẾT VÂN**

**ID: IELSIU22271**

**Thesis Advisor: Assoc. Prof. Dr. NGUYỄN VĂN HỢP**

Ho Chi Minh City, Vietnam

May 2026

**VIETNAM NATIONAL UNIVERSITY -- HO CHI MINH CITY**

**INTERNATIONAL UNIVERSITY**

**SCHOOL OF INDUSTRIAL ENGINEERING AND MANAGEMENT**


**PARALLEL-LINE CAPACITATED LOT-SIZING AND SCHEDULING PROBLEM WITH SEQUENCE_DENPENDENT SETUPS AND PREVENTIVE MAINTENANCE: A CASE STUDY OF LOOPSET MANUFACTURING**

Submitted in partial fulfillment of the requirements for the Degree of Bachelor of Engineering in Logistics and Supply Chain Management

**Student: TRƯƠNG NGỌC TUYẾT VÂN**

**ID: IELSIU22271**

**Thesis Advisor: Assoc. Prof. Dr. NGUYỄN VĂN HỢP**

Ho Chi Minh City, Vietnam

May 2026

**PARALLEL-LINE CAPACITATED LOT-SIZING AND SCHEDULING PROBLEM WITH SEQUENCE_DENPENDENT SETUPS AND PREVENTIVE MAINTENANCE: A CASE STUDY OF LOOPSET MANUFACTURIN**

By

TRƯƠNG NGỌC TUYẾT VÂN

> Submitted in partial fulfillment of the requirements for the Degree of Bachelor of Engineering in Logistics and Supply Chain Management

International University, Ho Chi Minh City

May 2026

Signature of Student:

Certified by

Thesis Advisor

Approved by

Dean of IEM school


# **Abstract**

Production scheduling in parallel-line manufacturing becomes especially difficult when sequence-dependent setups and preventive maintenance are included. Traditional exact methods and rolling-horizon heuristics scale poorly, while single-agent reinforcement learning struggles with heterogeneous decision structures. This study introduces a hybrid Recurrent Multi-Agent Proximal Policy Optimisation (RMAPPO) framework for the Parallel-Line Capacitated Lot-Sizing Problem with Sequence-Dependent Setup Times and Preventive Maintenance (PL-CLSP-SDST-PM). The proposed approach trains a multi-agent reinforcement learning layer for routing, sequencing, and maintenance control, then applies a post-hoc LP/MILP lot-sizing module that converts exported RL allocation masks into capacity-feasible production quantities for final evaluation. Using a CTDE training scheme, the method is tested on an industrial case and a benchmark set across multiple problem scales. RMAPPO performs comparably on small instances, improves medium-scale results, and delivers substantial cost reductions on large instances while cutting computation time to near real-time. Sensitivity analyses highlight robust parameter choices and confirm the framework's scalability for complex production environments.

> ***Keywords*:** multi-agent reinforcement learning, production scheduling, lot-sizing, preventive maintenance, proximal policy optimization.

# **Acknowledgements**

I would like to express my sincere and deepest gratitude to my thesis advisor Assoc. Prof. Dr. Nguyen Van Hop for the generous guidance, patience, and dedication provided throughout the entire development of this work. The insightful feedback, thorough review of each iteration, and consistent encouragement during moments of uncertainty were instrumental in shaping both the direction and quality of this thesis. The genuine investment in my progress and the warmth shown in every interaction have been a profound source of motivation that I will carry well beyond my academic years.

I would also like to extend my appreciation to the School of Industrial Engineering and Management for the opportunity to pursue this programme over the past four years. The education I have received, guided by knowledgeable and passionate lecturers who bring both academic rigour and industry relevance to their teaching, has equipped me with a foundation I am confident will serve me throughout my career in supply chain and operations management.

I am deeply grateful to my family, whose quiet and unwavering support sustained me through the most demanding periods of this journey. To my managers and colleagues at my workplace, I thank you for your understanding of the time commitments this research demanded and for the practical support you extended along the way. To my peers, who engaged in countless hours of discussion, challenged my thinking, and shared in both the difficulties and the small victories of this process - your companionship and encouragement made this journey far more meaningful than it would have been alone.

# **Table of Contents**

[Abstract [i](#abstract)](#abstract)

[Acknowledgements [ii](#acknowledgements)](#acknowledgements)

[List of Tables [v](#list-of-tables)](#list-of-tables)

[List of Figures [vi](#list-of-figures)](#list-of-figures)

[List of Abbreviations [vii](#list-of-abbreviations)](#list-of-abbreviations)

[Chapter 1: INTRODUCTION [1](#chapter-1-introduction)](#chapter-1-introduction)

[1.1. Background: [1](#background)](#background)

[1.2. Problem Statement: [4](#problem-statement)](#problem-statement)

[1.3. Objectives of Study: [5](#objectives-of-study)](#objectives-of-study)

[1.4. Scope and Limitation: [6](#scope-and-limitation)](#scope-and-limitation)

[Chapter 2: RELATED WORKS [8](#chapter-2-related-works)](#chapter-2-related-works)

[2.1. Overview: [8](#overview)](#overview)

[2.2. Literature Review: [8](#literature-review)](#literature-review)

[2.2.1. Heuristic and Matheuristic Solution Methods: [8](#heuristic-and-matheuristic-solution-methods)](#heuristic-and-matheuristic-solution-methods)

[2.2.2. Metaheuristic algorithms: [10](#metaheuristic-algorithms)](#metaheuristic-algorithms)

[2.2.3. Machine Learning Integration: [11](#machine-learning-integration)](#machine-learning-integration)

[2.2.4. Similarity and Differences Between the Key reference and the Thesis: [12](#similarity-and-differences-between-the-key-reference-and-the-thesis)](#similarity-and-differences-between-the-key-reference-and-the-thesis)

[2.2.5. Design Concepts Consideration: [13](#design-concepts-consideration)](#design-concepts-consideration)

[2.3. Key references/ Candidate Solution Methods: [15](#key-references-candidate-solution-methods)](#key-references-candidate-solution-methods)

[2.3.1. Matheuristic Approaches: [15](#matheuristic-approaches)](#matheuristic-approaches)

[2.3.2. Metaheuristic Algorithms: [16](#metaheuristic-algorithms-1)](#metaheuristic-algorithms-1)

[2.3.3. Machine Learning Integration: [17](#machine-learning-integration-1)](#machine-learning-integration-1)

[2.3.4. Pure RL: [17](#pure-rl)](#pure-rl)

[2.3.5. RL + Heuristic / Exact Hybrids: [18](#rl-heuristic-exact-hybrids)](#rl-heuristic-exact-hybrids)

[Chapter 3: METHODOLOGY [19](#chapter-3-methodology)](#chapter-3-methodology)

[3.1. Approaches Comparison and Selection: [19](#approaches-comparison-and-selection)](#approaches-comparison-and-selection)

[3.1.1. Comparative Analysis: [19](#comparative-analysis)](#comparative-analysis)

[3.1.2. Selection Rationale and Alignment with Objective: [22](#selection-rationale-and-alignment-with-objective)](#selection-rationale-and-alignment-with-objective)

[3.2. Proposed System Design or Proposed Solution Approach: [23](#proposed-system-design-or-proposed-solution-approach)](#proposed-system-design-or-proposed-solution-approach)

[Chapter 4: SOLUTION DEVELOPMENT [29](#chapter-4-solution-development-1)](#chapter-4-solution-development-1)

[4.1. Prototype Solution [29](#prototype-solution)](#prototype-solution)

[4.1.1. Problem Formulation [29](#problem-formulation)](#problem-formulation)

[4.1.2. Mathematical Model [30](#mathematical-model-1)](#mathematical-model-1)

[4.2. Two-Level Optimization [34](#two-level-optimization)](#two-level-optimization)

[4.2.1. Step 1: First-Level Optimization - Multi-Agent RL System (Routing and Sequencing) [34](#step-1-first-level-optimization---multi-agent-rl-system-routing-and-sequencing)](#step-1-first-level-optimization---multi-agent-rl-system-routing-and-sequencing)

[4.2.1.1. POMDP Formulation [34](#pomdp-formulation)](#pomdp-formulation)

[4.2.1.2. CTDE Framework [35](#ctde-framework)](#ctde-framework)

[4.2.1.3. Joint Policy and Recurrent Actor [36](#joint-policy-and-recurrent-actor)](#joint-policy-and-recurrent-actor)

[4.2.1.4. State and Observation Spaces [37](#state-and-observation-spaces)](#state-and-observation-spaces)

[4.2.1.5. Action Pool [40](#action-pool)](#action-pool)

[4.2.1.6. Reward Design [42](#reward-design)](#reward-design)

[4.2.1.7. GAE and Normalization [43](#gae-and-normalization)](#gae-and-normalization)

[4.2.1.8. Actor Objective: Clipping and Entropy [44](#actor-objective-clipping-and-entropy)](#actor-objective-clipping-and-entropy)

[4.2.1.9. Centralized Critic and Value Loss [45](#centralized-critic-and-value-loss)](#centralized-critic-and-value-loss)

[4.2.1.10. Combined Optimization Objective [46](#combined-optimization-objective)](#combined-optimization-objective)

[4.2.1.11. Parameters and Training Configuration [47](#parameters-and-training-configuration)](#parameters-and-training-configuration)

[4.2.2. Step 2: Second-Level Optimization - Post-hoc LP Lot Sizing [56](#step-2-second-level-optimization---post-hoc-lp-lot-sizing)](#step-2-second-level-optimization---post-hoc-lp-lot-sizing)

[4.2.3. STEP 3: Resolve the First-Level Optimization [58](#step-3-resolve-the-first-level-optimization)](#step-3-resolve-the-first-level-optimization)

[CHAPTER 5: RESULT ANALYSIS [60](#chapter-5-result-analysis)](#chapter-5-result-analysis)

[5.1. Experimental Design [60](#experimental-design)](#experimental-design)

[5.1.1. Dataset for real-case evaluation [60](#dataset-for-real-case-evaluation)](#dataset-for-real-case-evaluation)

[5.1.2. Dataset for Benchmark Comparison (Small, Medium, Large) [61](#dataset-for-benchmark-comparison-small-medium-large)](#dataset-for-benchmark-comparison-small-medium-large)

[5.2. Result Illustration and Explanation [64](#result-illustration-and-explanation)](#result-illustration-and-explanation)

[5.2.1. Real-Case Dataset Result [64](#real-case-dataset-result)](#real-case-dataset-result)

[5.2.2. Benchmark Comparison Results (Small, Medium, Large) [64](#benchmark-comparison-results-small-medium-large)](#benchmark-comparison-results-small-medium-large)

[5.3. Sensitivity Analysis [70](#sensitivity-analysis)](#sensitivity-analysis)

[5.3.1. Preventive-Maintenance Gate Threshold [71](#setup-time-estimation-mode)](#setup-time-estimation-mode)

[5.3.2. Kill-Switch Penalty [72](#capacity-safety-factor)](#capacity-safety-factor)

[5.3.3. Entropy Coefficient [73](#manager-operational-cost-weight)](#manager-operational-cost-weight)

[5.3.4 Learning Rate Scale [74](#learning-rate)](#learning-rate)

[5.3.5 PPO Clip Parameter [74](#ppo-clip-parameter)](#ppo-clip-parameter)

[5.4 Environmental, Social, and Economic Impacts [75](#environmental-social-and-economic-impacts)](#environmental-social-and-economic-impacts)

[5.4.1. Economic Impact [75](#economic-impact)](#economic-impact)

[5.4.2. Social Impact [76](#social-impact)](#social-impact)

[5.4.3. Environmental Impact [76](#environmental-impact)](#environmental-impact)

[CHAPTER 6: CONCLUSIONS [78](#chapter-6-conclusions)](#chapter-6-conclusions)

[6.1 Results Discussion and Implication [78](#results-discussion-and-implication)](#results-discussion-and-implication)

[6.2 Recommendations for Future Research [79](#recommendations-for-future-research)](#recommendations-for-future-research)

[References [82](#references)](#references)

# **List of Tables**

[Table 1.1: Product release matrix. [2](#table-1.1-product-release-matrix.)](#table-1.1-product-release-matrix.)

[Table 2.1: Candidate Solution Method Families [15](#table-2.1-candidate-solution-method-families)](#table-2.1-candidate-solution-method-families)

[Table 3.1: Optimization Approach Comparison [22](#table-3.1-optimization-approach-comparison)](#table-3.1-optimization-approach-comparison)

[Table 3.2: RL/DRL Algorithm Comparison [23](#table-3.2-rldrl-algorithm-comparison)](#table-3.2-rldrl-algorithm-comparison)

[Table 4.1: Sets and Indices [30](#table-4.1-sets-and-indices)](#table-4.1-sets-and-indices)

[Table 4.2: System Parameters [30](#table-4.2-system-parameters)](#table-4.2-system-parameters)

[Table 4.3: Cost Parameters [31](#table-4.3-cost-parameters)](#table-4.3-cost-parameters)

[Table 4.4: Decision Variables [31](#table-4.4-decision-variables)](#table-4.4-decision-variables)

[Table 4.5: Algorithm 1 RMAPPO Training Procedure [44](#table-4.5-algorithm-1-rmappo-training-procedure)](#table-4.5-algorithm-1-rmappo-training-procedure)

[Table 4.6: Observation Vector Layout [49](#table-4.6-observation-vector-layout)](#table-4.6-observation-vector-layout)

[Table 4.7: RMAPPO Hyperparameters [56](#table-4.7-rmappo-hyperparameters)](#table-4.7-rmappo-hyperparameters)

[Table 4.8: Environment and Reward Configuration [57](#table-4.8-environment-and-reward-configuration)](#table-4.8-environment-and-reward-configuration)

[Table 5.1: Generated Instance Set [61](#table-5.1-generated-instance-set)](#table-5.1-generated-instance-set)

[Table 5.2: Parameter Settings of the Generated Instances [62](#table-5.2-parameter-settings-of-the-generated-instances)](#table-5.2-parameter-settings-of-the-generated-instances)

[Table 5.3: Real-Case Performance Summary [64](#table-5.3-real-case-performance-summary)](#table-5.3-real-case-performance-summary)

[Table 5.4: Small Instance Results [65](#table-5.4-small-instance-results)](#table-5.4-small-instance-results)

[Table 5.5: Medium Instance Results [67](#table-5.5-medium-instance-results)](#table-5.5-medium-instance-results)

[Table 5.6: Large Instance Results [68](#table-5.6-large-instance-results)](#table-5.6-large-instance-results)

[Table 5.7: Summary Across Scales [69](#table-5.7-summary-across-scales)](#table-5.7-summary-across-scales)

---

# **List of Figures**

[Figure 1.1: CVT Push belt [1](#_Toc229106952)](#_Toc229106952)

[Figure 3.1: Proposed solution approach - RMAPPO system with post-hoc LP/MILP lot-sizing integration [28](#_Toc229103289)](#_Toc229103289)

[Figure 4.1: End-to-end system information flow (one decision period) [35](#_Toc229106954)](#_Toc229106954)

[Figure 4.2: Deep Reinforcement Learning Archtecture [48](#_Toc229106955)](#_Toc229106955)

[Figure 4.3: Training curve of manager agent [59](#_Toc229106956)](#_Toc229106956)

[Figure 4.4: Training curve of machine agent [59](#_Toc229106957)](#_Toc229106957)

[Figure 4.5: Total evaluation cost [59](#_Toc229106958)](#_Toc229106958)

[Figure 5.1: Gantt chart schedule [66](#_Toc229106959)](#_Toc229106959)

[Figure 5.2: Sensitivity analysis for PM gating threshold [71](#_Toc229106960)](#_Toc229106960)

[Figure 5.3: Sensitivity analysis for kill-switch penalty [72](#_Toc229106961)](#_Toc229106961)

[Figure 5.4: Sensitivity analysis for entropy, learning-rate scale, and PPO clipping [73](#_Toc229106962)](#_Toc229106962)


---

# **List of Abbreviations**


| Abbreviation | Meaning |
|:---|:---|
| CM | Corrective Maintenance |
| CTDE | Centralised Training with Decentralised Execution |
| DRL | Deep Reinforcement Learning |
| GAE | Generalised Advantage Estimation |
| GRU | Gated Recurrent Unit |
| MARL | Multi-Agent Reinforcement Learning |
| MDP | Markov Decision Process |
| MILP | Mixed-Integer Linear Programme |
| MLP | Multi-Layer Perceptron |
| PL-CLSP-SDST-PM | Parallel-Line Capacitated Lot-Sizing Problem with Sequence-Dependent Setup Times and Preventive Maintenance |
| PM | Preventive Maintenance |
| POMDP | Partially Observable Markov Decision Process |
| PPO | Proximal Policy Optimisation |
| ReLU | Rectified Linear Unit |
| RH2 | Rolling Horizon Heuristic |
| RL | Reinforcement Learning |
| RMAPPO | Recurrent Multi-Agent Proximal Policy Optimisation |
| SKU | Stock Keeping Unit |
| TD | Temporal Difference |



# Chapter 1: INTRODUCTION

## **1.1. Background: **

The automotive industry is under constant pressure to improve vehicle fuel efficiency and reduce emissions. Continuously Variable Transmissions (CVT) push belts are a key technology in achieving these goals due to their ability to maintain engines at optimal operating points. The push belt operates under compressive force and is composed of hundreds of precisely manufactured steel elements assembled together with high-strength steel bands. The loopset is a core sub-component of the push belt. It forms the structural backbone that holds the elements together and ensures stable force transmission. Because loopsets feed directly into push belt assembly, any instability or shortage in loopset output immediately impacts downstream assembly operations.

*Figure 1.1: CVT Push belt image placeholder*

Loopset manufacturing represents a complex production environment with parallel production lines (LL5, LL6, LL7, LL9, LL10, LL11) that process multiple product families including Conventional, GU Light (GUL) and GU in discrete production lots. These production lines are heterogeneous. Due to differences in equipment configuration and technical capability, the same product may have materially different cycle times and output rates on different lines. In addition, not all product families can be freely assigned to all lines. Product line eligibility constraints result in certain lines being dedicated to specific product families. Taken together, these characteristics, including parallel unrelated lines, product-line eligibility restrictions, and time-varying demand, define loopset production as a practical instance of the Capacitated Lot Sizing and Scheduling Problem (CLSP) on parallel unrelated machines.

###### Table 1.1: Product release matrix.

| Line | 067 | 072 | 094 | 100 | 097 | 089 | 083 | 111 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LL5 | Yes | Yes |  | Yes |  |  |  |  |
| LL6 | Yes | Yes | Yes |  |  |  |  |  |
| LL7 |  |  | Yes | Yes | Yes |  |  |  |
| LL9 |  | Yes |  | Yes |  | Yes |  |  |
| LL10 |  |  |  |  |  | Yes | Yes |  |
| LL11 |  |  |  |  |  | Yes |  | Yes |


In addition to line heterogeneity, loopset production is characterized by sequence-dependent setup times and costs. When a line switches from one product family to another, the resulting changeover time and resource cost depend specifically on which product is produced next and which product was produced immediately before it, as well as on the particular line performing the changeover. As a result, the sequence in which lots are scheduled directly influences both the effective production capacity and the total operational cost. Because setup losses can be substantial, the ordering of production lots becomes a critical planning decision that interacts closely with lot sizing and line assignment.

Equipment reliability further complicates planning. Loopset lines require periodic preventive maintenance (PM) to sustain their performance levels and prevent costly unplanned failures. Unlike stochastic breakdowns, which are difficult to anticipate, preventive maintenance activities can be anticipated and incorporated into the production plan. When PM is explicitly considered in the scheduling model, planners can coordinate maintenance windows with production sequencing to minimize their impact on output. At the same time, unplanned corrective maintenance (CM) events impose additional cost and capacity loss that must be accounted for when evaluating production feasibility and total cost.

Despite this complexity, current planning practice relies heavily on manual adjustments and spreadsheet-based tools. Planners typically assign products to lines based on experience and rules of thumb. These methods fail to effectively optimize for the complexities of the systems. Additionally, manual planning struggles to look ahead far enough to minimize sequence-dependent setups, resulting in significant underestimates setup losses caused by poor sequencing, and cannot proactively integrate maintenance planning with production decisions. As a result, the system experiences unnecessary changeover time, avoidable tardiness, and inefficient use of available capacity.

This gap between industrial practice and planning potential motivates the need for present study. By formulating loopset production as a Parallel-Line Capacitated Lot-Sizing and Scheduling Problem with Sequence-Dependent Setups and Preventive Maintenance (PL-CLSP-SDST-PM), this research develops a structured, data-driven planning model that reflects the full complexities of the real production environment. The model integrates demand-driven lot sizing, product-line eligibility, unrelated machine speeds, sequence-dependent setup costs, and maintenance considerations into a unified optimization framework. In doing so, it replaces subjective manual scheduling with a systematic approach grounded in operational reality.

## **1.2. Problem Statement:**

The loopset production system presents a planning environment of considerable practical and scientific complexity. On the demand side, the production system must satisfy daily customer requirements for multiple product families simultaneously. On the supply side, six parallel but unrelated production lines, each with different cycle times and eligibility restrictions, must collectively absorb this demand within a finite planning horizon. The interaction between product mix, line capability, and demand timing creates a combinatorially difficult planning problem that cannot be solved effectively by manual methods.

Each production line operates under a finite capacity limit per planning period. This capacity is not unlimited; the total quantity that can be produced on a given line in a given period is bounded by the line\'s throughput rate and the available production time in that period. In a lot-sizing formulation, this constraint is fundamental because production lots must be sized and sequenced such that the total capacity consumed across all lots assigned to a line in any period does not exceed the line\'s available capacity. Since lines have different throughput rates for the same product, the capacity required by a given lot depends jointly on the product being manufactured and the specific line to which it is assigned.

Sequence-dependent setup times and costs add a further layer of difficulty. Because the changeover penalty depends on both the outgoing and incoming product on a given line, the cost incurred at any point in the schedule is a function of the entire sequence of lots, not just the current assignment. Minimizing total setup cost therefore requires reasoning over sequences, a task that is computationally intensive and highly sensitive to how far ahead planners look. Current spreadsheet-based methods lack the algorithmic capability to optimize sequences over a multi-period horizon, consistently leaving substantial setup savings unrealized.

Preventive maintenance introduces an additional source of capacity constraint that is largely absent from classical lot-sizing models. In practice, PM activities must be performed at regular intervals to maintain line performance and prevent unplanned failures that result in corrective maintenance and production losses. If maintenance windows are not coordinated with production lot assignments, they can cause avoidable disruptions, force costly sequence changes, and erode the capacity buffer available for meeting demand. A planning model that treats maintenance as an external event, rather than integrating it as a decision or constraint, will consistently overestimate available capacity and underestimate the true cost of the production plan.

From a scientific perspective, existing literature on Capacitated Lot-Sizing Problems has focused primarily on single-machine or identical-parallel-machine settings, where sequence-dependent setups are treated as an extension of moderate complexity. The combination of unrelated parallel machines, product-line eligibility, sequence-dependent setup costs, and integrated preventive maintenance in a demand-driven, multi-product environment has received comparatively limited attention. Loopset production therefore represents a valuable and practically grounded test case for extending CLSP theory toward realistic industrial conditions.

## **1.3. Objectives of Study: **

The objective of this study is to develop an integrated planning framework for the loopset production environment by formulating the problem as a PL-CLSP-SDST-PM. The model determines, for each line and each planning period, the appropriate production lot sizes, the sequencing of lots on each line, and the timing of preventive maintenance activities, while ensuring that all product demands with their due dates are met at minimal total cost. By doing so, the study generalizes the classical single machine lot sizing and scheduling framework to a more realistic multi line context that includes unrelated machine speeds, product line eligibility constraints, sequence dependent setups, and maintenance integration. The interaction between production and maintenance decisions is central to this problem, since maintenance directly reduces available production capacity, while production sequencing influences how often maintenance is required to avoid failures.

Beyond methodological development, the study aims to generate both industrial and academic value. Production planners and industrial engineers will gain a structured decision-support framework that makes production lot assignments, line scheduling, and changeover sequencing more transparent, consistent, and cost-effective, thereby reducing the reliance on experience-based judgment that is difficult to standardize or transfer. Operations managers will benefit from improved visibility into the trade-offs between inventory holding, backlogging, setup frequency, and maintenance timing, which support more informed and defensible planning decisions. From a scientific perspective, researchers working on lot-sizing, production scheduling, and maintenance integration will find in this study a rigorously formulated extension of the CLSP that brings together several underexplored complexities in a single, industrially validated context.

## **Scope and Limitation:**

This study focuses specifically on six parallel loop lines (LL5, LL6, LL7, LL9, LL10, LL11) which are the key bottleneck in the loopset manufacturing process. The upstream pipe line stages are treated as external inputs where semi-finished pipe material is assumed to be available at the loop line buffers at or before its specified release time. This boundary separates the lot-sizing and scheduling problem studied here from the operational variability inherent in upstream processes.

Production capacity on each line in each planning period is treated as a given finite parameter, reflecting the line\'s throughput rate and the total available production time for that period. This capacity parameter consolidates all practical sources of capacity loss, including equipment reliability issues, minor stoppages, and performance variability, into a single bound that limits the total production quantity that can be produced on each line. Corrective maintenance is incorporated through its impact on the reduction of available capacity in the periods in which it occurs and through an explicit stochastic representation of failure events.

The model operates at the production lot level rather than tracking individual pieces. A production lot corresponds to a batch of product of a single family processed continuously on a single line within a single planning period. This granularity aligns with the existing inventory control and reporting structure at the plant. Material handling and inter-stage transfer times are assumed to be negligible or absorbed within setup and processing durations, consistent with the scope of planning-level decisions rather than detailed operational control.

Several advanced features are intentionally excluded from the present study and identified as directions for future work. These include blocking constraints between production stages and dynamic re-planning triggered by real-time events. These exclusions ensure that the proposed model remains tractable and closely aligned with the planning-level decision-making context for which it is designed, while still capturing the essential complexities of loopset production. This scope and limitation ensure that the proposed model remains solvable and closely aligned with the real industrial practice while still capturing the essential complexity of loopset production.

# Chapter 2: RELATED WORKS

## Overview:

This study investigates the PL-CLSP-SDST-PM for the loop-set production lines of a major automotive enterprise in Vietnam. The PL-CLSD problem represents one of the most practically relevant and computationally challenging classes of production planning problems. When production is organised across parallel machines or lines, complexity increases due to machine assignment decisions, load balancing, and interactions across lines. This complexity is further amplified when setup times and costs vary according to the production sequence. An additional but often underexplored factor is the role of preventive maintenance. Real manufacturing systems experience equipment deterioration, and PM activities consume capacity that would otherwise be available for production. Neglecting PM can lead to overly optimistic capacity assumptions and infeasible schedules, while excessively conservative maintenance strategies raise operational costs. Integrating PM into lot-sizing and scheduling (LSS) models is therefore crucial for achieving realistic, cost-efficient, and industrially relevant solutions. A review of existing studies offers an understanding of the current research landscape on PL-CLSP-SDST-PM. It identifies the methodologies, findings, and gaps that inform the present research. By examining prior work, this chapter establishes the foundation for developing an optimisation approach tailored to the specific requirements and operational constraints of loop-set manufacturing, particularly in balancing lot-sizing, scheduling, sequencing, and maintenance decisions.

## Literature Review:

### Heuristic and Matheuristic Solution Methods:

Exact MIP approaches rarely scale for CLSSP. Therefore, heuristic and matheuristic methods dominate recent research. Deeratanasrikul & Mizuno (2016) model a multi-stage wheel-manufacturing CLSSP with sequence-dependent setups and test on real data. They use relax-and-fix MIP heuristics per period and iteratively fix variables. These heuristics find feasible solutions efficiently and improve the company's plan. Similarly, de Armas & Laguna (2016) study a parallel‐machine lot-sizing problem for the pipe insulation industry using a two-stage approach. They first solve a big-MIP model for lot sizes, then sequence lots with a constructive heuristic and Monte Carlo for stochastic rates. Avilés et al. (2022) developed a MILP model integrating PM for a multi-line production planning problem. Their formulation accommodates heterogeneous parallel lines with sequence-independent setups and periodic PM windows, offering a practical template for capital-intensive process industries. The model was validated against real mill data, demonstrating cost savings relative to manual scheduling. Alimian et al. (2022) formulate a parallel-line CLSP with sequence-dependent setups, due dates, and preventive maintenance as a mixed-integer model. To handle its complexity, they develop two MILP-based rolling-horizon (RH) heuristics (RH1 and RH2) that solve truncated time windows sequentially. Computational results show these RH methods significantly outperform a straight CPLEX solve in solution quality. In particular, RH2 provides reasonable production plans, better-adjusted schedules with respect to due dates, and a proper maintenance plan compared to alternatives. This demonstrates how mathematics can exploit problem structure to scale PL-CLSD models.

Matheuristics, hybrid algorithms that embed MILP solvers within heuristic frameworks, have grown in prominence because they combine the modelling precision of MILP with the scalability of heuristics. The relax-and-fix (R&F) and fix-and-optimize (F&O) paradigm is particularly widespread. Carvalho & Nascimento (2022) consider the integrated lot sizing and scheduling problem (ILSSP) on non-identical parallel machines with non-triangular sequence setups. They propose a R&F decomposition combined with path-relinking and kernel search heuristics. Their hybrid methods vastly outperform CPLEX solvers on most benchmark instances. Dansou et al. (2025) address a single-machine capacitated lot-sizing problem (CLSP) with preventive maintenance. They model machine deterioration using capacity reduction functions or age-based decay and solve by a two-stage R&F plus F&O heuristic. In this formulation, PM decisions reduce failure-induced downtime by preserving capacity. The proposed heuristics effectively balance lot-sizing and maintenance scheduling.

### Metaheuristic algorithms:

The practical necessity of near-optimal solutions for large instances has driven extensive development of metaheuristic algorithms for PL-CLSD problems. Babaei et al. (2014) applied GA to a multi-level CLSD with setup carry-over and backlogging in a capacitated flow shop. Yue et al. (2019) addressed a multi-objective CLSP with sequence-dependent setup times for flexible parallel lines using a Pareto-based Guided Artificial Bee Colony (PGABC) algorithm, demonstrating that population-based metaheuristics can effectively navigate the Pareto frontier of cost versus makespan on parallel-line problems.

For integrated production-PM models, Feng et al. (2018) proposed an imperfect PM optimisation framework for flexible flowshop manufacturing cells (FFMCs) with sequence-dependent group scheduling. To solve the model, a simulated annealing embedded genetic algorithm (SAGA) is developed. Experiments show that jointly optimising lot sequences and PM plans simultaneously results in considerable cost reductions of 12-18% over sequential approaches.

Hybrid metaheuristics are increasingly preferred over standalone algorithms. Xiao et al. (2015) hybridised Lagrangian relaxation with Simulated Annealing (SA) for parallel-machine CLSD with sequence-dependent setups. Ramezanian and Saidi-Mehrabad (2013) combined hybrid SA, Firefly algorithms, and MIP-based heuristics for stochastic CLSD with uncertain demand and processing times. An et al. (2022) addressed joint optimisation of PM and flexible job-shop rescheduling with new machine insertion and processing speed selection via a metaheuristic approach, reporting effective navigation of the combined scheduling-maintenance decision space in dynamic manufacturing environments. Ruiz-Rodríguez et al. (2024) compared GA-simheuristics (GA combined with Monte Carlo simulation), reinforcement learning, and classical dispatching rules for dynamic maintenance scheduling under uncertainty. The GA-simheuristic was most robust on high-uncertainty instances, highlighting that stochastic maintenance durations significantly decrease the performance of deterministic algorithms.

### Machine Learning Integration:

Although AI/ML applications in lot-sizing and scheduling are still in their early stages, some recent studies show their potential, especially when problems include dynamic parameters such as machine degradation, unpredictable demand, or the need for predictive maintenance. Azab et al. (2021) develop a data-driven, ML-integrated scheduling framework for a pharmaceutical flow-shop where preventive maintenance windows must be predicted in advance. By training ML models on IoT sensor data, the system predicts the remaining useful life (RUL) of equipment and proposes proactive maintenance periods. These predictions feed into a discrete-event simulator that then schedules jobs and maintenance jointly. The results show substantial reductions in makespan and unplanned downtime compared to reactive maintenance approaches, demonstrating the value of integrating ML predictions into lot-sizing and scheduling decisions. Tang et al. 2022 uses an Elman neural network to predict both demand and machine failure rates, feeding these forecasts into a lot-sizing and scheduling model solved by Particle Swarm Optimization (PSO). Such techniques show that incorporating demand learning can improve schedule quality under uncertainty. However, a recent survey by Adetunji (2024) noted that AI/ML use in lot-sizing is "still in its infancy," with limited adoption compared to routing, forecasting, or predictive maintenance.

Existing AI-driven CLSSP research generally falls into two categories: pure RL/DRL schedulers and RL-plus-optimization hybrids. Pure RL studies include DRL dispatching and sequencing methods for job-shop or flexible job-shop scheduling, such as Zhang et al. (2020), Luo (2020), and Wang et al. (2021), where the trained policy directly selects dispatching or scheduling actions. RL-plus-optimization hybrids include frameworks that combine a learned policy with a heuristic, genetic algorithm, or mathematical programming module, such as Chien and Lan (2021), He et al. (2021), and Yu et al. (2025). This distinction is important for the present study because the proposed system does not ask RL to solve all lot-sizing and scheduling decisions alone. Instead, it uses RMAPPO to learn the discrete routing and sequencing policy, then uses a post-hoc LP/MILP quantity model to compute feasible production quantities under the learned routing mask.

### Similarity and Differences Between the Key reference and the Thesis:

Among the studies reviewed, Alimian et al. (2022) stands as the most directly relevant work to the formulation adopted in this thesis. It presents a deterministic Mixed-Integer Nonlinear Programming (MINLP) model for a parallel-line Capacitated Lot-Sizing Problem with sequence-dependent setup time/cost, due dates, and integrated Preventive Maintenance (PM) planning. The model is solved using a MIP-based RH heuristic, and its findings are validated against a set of generated benchmark instances. This study shares the same core problem class and builds upon the same foundational structure, while introducing several extensions motivated by the industrial context of loopset production.

On the side of similarity, the present study inherits from Alimian et al. (2022) the same production-planning structure: parallel unrelated lines, sequence-dependent setup time and cost, demand satisfaction through inventory/backlog balance, and preventive-maintenance/corrective-maintenance cost evaluation. These elements remain essential because they accurately represent the operational structure of loopset production and have been validated as effective modelling choices in prior work.

The primary structural extension is the addition of product-line eligibility constraints. In the loopset environment, not all product families can be freely assigned to all lines as certain lines are technically restricted to specific product families, as described in Chapter I. This restriction is absent from Alimian et al. (2022), where any product can be assigned to any line.

The more fundamental difference lies in the treatment of failure risk and maintenance planning. Alimian et al. (2022) use fixed statistical parameters, specifically a constant hazard rate and an assumed exponential lifetime distribution, to compute the expected number of failures analytically. These parameters remain unchanged across planning periods and the model does not adjust in response to recent operating conditions, such as heavy utilization, early signs of wear, or unusually stable performance. As a result, the RH procedure functions in a reactive manner. It optimizes based on the conditions observed at the beginning of each cycle, but it cannot anticipate deteriorating equipment conditions before it materializes as capacity loss or unplanned downtime.

A second limitation concerns the separation between repeated optimization and learned decision making. In the rolling-horizon framework, lot sizing, line assignment, sequencing, and preventive maintenance scheduling are re-optimised within each window, but the policy implicit in the solution is reconstructed from the solver at every cycle. The system therefore does not retain accumulated knowledge about which routing and sequencing patterns are effective under specific combinations of machine age, inventory position, and demand pressure. Each planning cycle begins independently of previous ones.

The research gap addressed in the present study can therefore be stated as follows: exact and rolling-horizon methods handle continuous lot sizes well, but become computationally expensive as sequence-dependent setups and instance size grow; pure RL methods can learn fast scheduling policies, but a monolithic action space containing exact quantities, routing, sequencing, and PM is unstable and difficult to train. The proposed answer is a decoupled hybrid method. RMAPPO learns the first-level discrete decisions, namely product-line activation, sequencing, PM, and End-Shift actions. A post-hoc LP/MILP stage then computes continuous production quantities from the fixed RL routing mask. This keeps the learning problem tractable while preserving optimization discipline for quantity allocation.

### Design Concepts Consideration:

The present study adopts the following design concepts, each aligned with the proposed hybrid planning framework.

First, the production system is formulated as a cooperative partially observable sequential decision problem. Loopset production unfolds over periods and machine micro-steps, and decisions affect later feasibility through inventory, backlog, queue states, setup state, machine age, and remaining capacity. A POMDP formulation is therefore more accurate than a fully observable MDP because each machine agent observes only its own line context while the centralised critic can access the full shared state during training.

Second, the planning problem is decomposed into two levels rather than solved as one monolithic RL action space. The first level is a multi-agent RMAPPO scheduler that learns discrete product-line activation masks, product execution actions, gated preventive maintenance actions, and End-Shift decisions. The second level is a post-hoc LP/MILP lot-sizing model that receives the fixed RL routing mask and computes continuous production quantities, inventory, and backlog. This design assigns routing and sequencing to RL while preserving mathematical optimization for the continuous quantity trade-off.

Third, the agent roles are separated. The Process Agent learns a binary L x P allocation mask at each decision point. During Step 1 training, the JIT allocator uses that mask to create temporary queues from remaining demand. Machine agents then choose local actions: produce an available product, perform preventive maintenance when the gated PM rule allows it, or end the period. In execution, the machine agents drain physical queues generated from the post-hoc lot-size file.

Fourth, the observation design follows CTDE. The manager receives global demand, inventory, backlog, line availability, eligibility, and allocation-pressure information. Machine agents receive local line information while unrelated line queues are zero-filled; a one-hot line identity lets the shared machine policy distinguish lines. The active benchmark configuration uses binary observation encoding, so inventory, backlog, queue, demand, and shortfall segments are represented as presence indicators rather than continuous log-scaled magnitudes.

Fifth, the reward design reflects the first-level training objective. It combines dense production/setup/maintenance shaping with a period-end shared team reward and a proportional unmet-demand kill-switch penalty. The final reported cost is not the raw training reward. It is computed during deterministic evaluation from production, setup, preventive maintenance, expected corrective maintenance, inventory, and backlog components after the post-hoc lot sizes have been applied.

Sixth, feasibility is protected by action masking and post-hoc optimization constraints. Eligibility and capacity masks prevent invalid product and PM actions in the actor output. The post-hoc lot-sizing model also enforces eligibility, backlog/inventory balance, and capacity after subtracting conservative setup overhead implied by the active RL mask.

## Key references/ Candidate Solution Methods:

The candidate methods are organised into four practical families: heuristic/matheuristic MIP methods, metaheuristics, pure RL/DRL methods, and RL-plus-optimization hybrids. This grouping is used because the final implementation is not selected from a single family. RH2 is retained as the mathematical-programming benchmark, RMAPPO is selected for the first-level scheduling policy, and post-hoc LP/MILP is selected for second-level lot sizing.

###### Table 2.1: Candidate Solution Method Families

| Method family | Representative methods | Role in this thesis |
|:---|:---|:---|
| Heuristic and hybrid MIP | Relax-and-Fix, Fix-and-Optimize, Rolling Horizon, RH2 | Used as literature foundation and benchmark comparator |
| Metaheuristics | GA, SA, TS, PSO, ABC | Reviewed as scalable alternatives, but not selected because feasibility and optimality guarantees are weaker |
| Pure RL/DRL | REINFORCE, Q-learning, DQN, Double DQN, Basic Actor-Critic, A2C, A3C, DPG, DDPG, PPO, MAPPO/RMAPPO | Reviewed as policy-learning approaches; pure end-to-end quantity scheduling is not selected |
| RL + heuristic / exact hybrids | RL + GA, RL + OR/MIP, RL + post-hoc LP/MILP, two-stage RL search-space reduction | Selected approach: RL learns routing/sequencing, LP/MILP computes quantities |


### Matheuristic Approaches:

#### **Relax-and-Fix (R&F):**

The Relax-and-Fix (R&F) approach partitions the planning horizon into time buckets and solves a sequence of restricted MIP subproblems, treating the binary decision variables of one bucket as true integers while relaxing the integrality of all other buckets to their continuous LP relaxation. This relaxation substantially reduces the computational burden of each subproblem while preserving the coupling between periods through the shared inventory and machine-state constraints. Once a bucket is solved, its binary variables are fixed at the obtained values, and the window advances to the next bucket.

#### **Fix-and-Optimize(F&O):**

Fix-and-Optimize (F&O) starts from a feasible initial solution and iteratively improves it by unfixing a carefully chosen subset of binary variables and re-optimizing while the remaining variables are held constant. The partition of variables can be defined by time period, by product family, or by production line, enabling targeted neighborhood exploration. R&F and F&O are often combined, where R&F generates a good initial feasible solution, which F&O then refines through successive local re-optimization passes.

#### **Rolling Horizon (RH):**

The Rolling Horizon approach is conceptually related to R&F but differs in an important respect: rather than relaxing integrality for future buckets, RH freezes all decision variables of past periods at their previously solved values and solves the full MIP for the current and future periods at each iteration. This design choice preserves the causal integrity of inter-period dependencies because inventory levels, machine ages, and capacity states carry forward from frozen decisions and allows the solver to re-optimize future periods at each step as new information becomes available.

### **Metaheuristic Algorithms:**

#### **Genetic Algorithms (GA):**

Genetic Algorithms (GA) operate by maintaining a population of candidate solutions (chromosomes) and iteratively improving it through selection, crossover, and mutation operators that mimic biological evolution. In the CLSP context, each chromosome commonly represents a complete production schedule and a fitness function evaluates its total cost. GA is well-suited to problems with complex constraint interactions and multimodal objective landscapes, but requires careful encoding design to prevent the generation of infeasible solutions, particularly when the problem involves binary decisions and hard eligibility restrictions.

#### **Simulated Annealing (SA):**

Simulated Annealing (SA) is a trajectory-based method that probabilistically accepts worsening moves according to a temperature schedule, allowing the search to escape local optima during early iterations and converge toward a high-quality solution as the temperature decreases. SA is most effective as a local search component embedded within a larger hybrid framework, where it refines solutions found by a constructive or population-based method.

#### **Tabu Search (TS):**

Tabu Search (TS) maintains a short-term memory of recently visited solutions, referred to as the tabu list, and forbids returning to them, encouraging the search to explore new regions of the solution space. It is particularly effective for sequencing and scheduling problems where solution neighborhoods can be defined through well-structured move operations such as lot swaps or sequence reversals.

#### **Particle Swarm Optimization:**

Particle Swarm Optimization (PSO) updates a population of candidate solutions by combining each particle\'s own historical best position with the swarm\'s global best position, guiding the search toward promising regions. PSO handles continuous solution spaces naturally and is often extended to discrete and binary problems through velocity discretization or probability-based encoding. It is well-suited to multi-objective formulations where the Pareto frontier must be approximated.

#### **Artificial Bee Colony (ABC):**

Artificial Bee Colony algorithms divide the search population into employed bees that exploit known solutions, onlooker bees that probabilistically select solutions to improve, and scout bees that explore new regions randomly. ABC is effective at balancing exploitation and exploration, and Pareto-based variants have been developed to handle problems with multiple competing objectives.

### **Machine Learning Integration:**

Machine learning methods offer a fundamentally different capability: rather than optimizing only a fixed mathematical model, they learn policies from data or simulation and use those policies to guide scheduling decisions. In the context of production planning and maintenance scheduling, the relevant learning-based methods are pure RL/DRL schedulers and RL-plus-heuristic or RL-plus-exact hybrids.

### **Pure RL:**

Pure RL methods learn a scheduling policy directly from interaction with a simulated production environment. In this setting, an RL agent observes inventory, backlog, demand pressure, line state, and maintenance risk, then selects actions that aim to minimise cumulative cost. Their main advantage is fast inference after training, since a trained policy can produce decisions without repeatedly solving a mathematical model. Their main limitation is that a single end-to-end policy over exact quantities, routing, sequencing, and maintenance creates a very large action space and can be unstable to train.

#### **REINFORCE:**

REINFORCE is a Monte Carlo policy-gradient method that directly adjusts policy parameters using sampled returns. It is conceptually simple and suitable for discrete scheduling actions, but its high variance makes it inefficient for long-horizon production planning where rewards are delayed through inventory and backlog effects.

#### **Q-learning:**

Q-learning estimates the value of state-action pairs and selects actions that maximise expected future reward. It is useful for small discrete control problems, but a tabular representation cannot handle the large state space created by products, lines, periods, setup states, queues, and maintenance ages.

#### **Deep Q-Network (DQN):**

DQN replaces the tabular Q-function with a neural network. Luo (2020) applies a DQN-based approach to dynamic flexible job-shop scheduling with new job insertions. DQN improves scalability compared with tabular Q-learning, but it remains difficult to apply directly to cooperative multi-agent scheduling with recurrent memory and hard action masks.

#### **Double DQN:**

Double DQN reduces the value-overestimation bias of DQN by separating action selection from action evaluation. This can improve stability in discrete scheduling tasks, but it still inherits DQN's limitations for multi-agent credit assignment and complex combinatorial action spaces.

#### **Basic Actor-Critic:**

Basic actor-critic methods learn both a policy and a value function. The actor selects scheduling actions, while the critic estimates expected return. This structure is more flexible than value-only learning, but basic actor-critic methods can be unstable when rewards are sparse, delayed, or dominated by large backlog penalties.

#### **A2C:**

Advantage Actor-Critic (A2C) improves actor-critic learning by using advantage estimates rather than raw returns. It is more stable than a basic actor-critic, but it is less directly suited than MAPPO/RMAPPO for decentralised execution with a centralised critic.

#### **A3C:**

Asynchronous Advantage Actor-Critic (A3C) trains multiple workers in parallel to improve exploration and throughput. Its asynchronous updates are useful for some RL settings, but they add implementation complexity and do not directly address cooperative multi-agent credit assignment.

#### **DPG:**

Deterministic Policy Gradient (DPG) learns deterministic policies for continuous control. It is less suitable for the current scheduling layer because the first-level decisions are masked discrete actions such as product selection, PM, and End-Shift.

#### **DDPG:**

Deep Deterministic Policy Gradient (DDPG) extends DPG with deep actor and critic networks. It is powerful for continuous control but sensitive to hyperparameters and mismatched with the discrete routing and sequencing action structure used in this study.

#### **PPO:**

Proximal Policy Optimisation (PPO) uses clipped policy updates to improve training stability. Wang et al. (2021) use PPO for dynamic job-shop scheduling in smart manufacturing. PPO is a strong baseline for learned scheduling, but in its standard form it is not designed for cooperative multi-agent CTDE.

#### **MAPPO / RMAPPO:**

MAPPO extends PPO to cooperative multi-agent problems using centralised training and decentralised execution. RMAPPO adds recurrent policies, which are important when each agent sees only partial production state. This makes RMAPPO suitable for the proposed first-level scheduler because it supports multiple agents, action masking, and line-level hidden state.

### **RL + Heuristic / Exact Hybrids:**

RL-plus-heuristic or RL-plus-exact hybrids divide the planning task between learning and optimization. RL is used for the combinatorial decisions that benefit from fast learned policies, such as line-product routing and sequencing, while a heuristic or exact optimizer is used for quantity allocation and feasibility repair. This structure is suitable for the present problem because it avoids a monolithic RL action space while retaining an optimization model for continuous lot sizing.

#### **RL + Genetic Algorithm:**

RL can be combined with a genetic algorithm when the learned component guides high-level decisions and the GA searches over detailed schedules. Chien and Lan (2021) integrate DRL with a hybrid genetic algorithm for unrelated parallel-machine scheduling with sequence-dependent setup time. This approach is flexible but can require careful chromosome design to preserve feasibility.

#### **RL + OR/MIP:**

In RL-plus-OR or RL-plus-MIP frameworks, the RL policy narrows the search space or proposes decisions, and a mathematical optimisation model solves the remaining structured problem. He et al. (2021) propose a two-stage framework where RL supports search-space reduction and an optimisation method completes the scheduling decision. This is attractive when exact optimisation is too slow on the full model but remains useful on a reduced model.

#### **RL + Post-hoc LP/MILP:**

In a post-hoc LP/MILP hybrid, the learned policy first outputs routing or sequencing decisions, then the optimisation model computes quantities or repairs feasibility under those fixed decisions. Yu et al. (2025) combine PPO with MILP for flexible job-shop scheduling with variable lot-sizing. The present study follows this logic: RMAPPO learns the first-level routing and sequencing policy, while the post-hoc lot-sizing model computes feasible production quantities under the learned mask.

#### **Two-stage RL Search-space Reduction:**

Two-stage RL search-space reduction uses RL to eliminate poor decisions before applying a heuristic or exact optimiser. This is useful for large-scale scheduling because it keeps the optimisation model focused on a smaller set of plausible decisions instead of the full combinatorial search space.

# **Chapter 3: METHODOLOGY**

## **3.1. Approaches Comparison and Selection:**

### **3.1.1. Comparative Analysis:**

The comparison is organised into two selection layers: the optimization method used for quantity allocation, and the RL/DRL algorithm used for the learned scheduling policy.

###### Table 3.1: Optimization Approach Comparison

| Approach | Optimality guarantee | Handles sequence-dependent setups | Scalability | Integrates with RL output |
|:---|:---|:---|:---|:---|
| Greedy / priority rule | None | Weak | High | Easy |
| Metaheuristics | None or empirical only | Possible with custom encoding | Medium to high | Indirect |
| Constraint programming | Feasibility-focused | Strong for sequencing | Medium | Limited |
| MILP / LP | Exact within the reduced model | Strong when routing/setup implications are fixed or bounded | High after RL mask fixation | Direct |


The selected framework uses the final row. The full routing and sequencing problem is not sent to the solver. Instead, RMAPPO exports fixed binary masks. With those masks fixed, the post-hoc quantity model solves the reduced continuous allocation problem with inventory, backlog, capacity, production, and expected corrective-maintenance costs.

###### Table 3.2: RL/DRL Algorithm Comparison

| Algorithm | Advantage | Disadvantage |
|:---|:---|:---|
| REINFORCE | Simple policy-gradient method and easy to implement | High variance and poor sample efficiency |
| Q-learning | Clear value-learning principle for discrete actions | Does not scale well to large, multi-agent combinatorial scheduling states |
| DQN | Uses neural networks to approximate value functions | Difficult to combine with recurrent multi-agent action masking at large scale |
| Basic Actor-Critic | Learns both policy and value function | Can be unstable under sparse and delayed production-planning rewards |
| Double DQN | Reduces value overestimation compared with DQN | Still limited for cooperative multi-agent recurrent scheduling |
| A2C | More stable than basic actor-critic through advantage estimation | Less suitable for decentralised multi-agent execution than MAPPO |
| A3C | Parallel workers improve exploration and training throughput | Asynchronous updates add implementation complexity and do not directly solve multi-agent credit assignment |
| DPG | Suitable for deterministic continuous control | Not appropriate for masked discrete production actions |
| DDPG | Extends actor-critic learning to continuous actions | Sensitive to hyperparameters and mismatched with the discrete first-level scheduler |
| PPO | Stable clipped policy updates, good sample efficiency, and simple implementation | Designed for single-agent use unless extended to multi-agent CTDE |
| MAPPO / RMAPPO | Supports cooperative multi-agent CTDE, recurrent policies, and masked discrete actions | Requires careful reward design, recurrent buffers, and training stabilisation |


The selected policy algorithm is RMAPPO. PPO clipping limits destructive updates, recurrent actors handle partially observed line histories, and the centralised critic provides a shared training signal while preserving decentralised execution.

### **3.1.2. Selection Rationale and Alignment with Objective:**

The selected method is a hybrid RMAPPO plus post-hoc LP/MILP pipeline, not a pure DRL-only solution. This choice follows directly from the research gap and the comparative analysis.

Pure rolling-horizon MILP methods, represented by RH2, are strong on small cases because they solve structured mathematical subproblems with explicit feasibility constraints. However, as the number of products and lines increases, the sequence-dependent setup and routing search space grows rapidly and the solver repeatedly reaches the time limit.

Pure RL methods avoid repeated solver calls, but asking a policy network to choose exact production quantities, routing, sequencing, and maintenance in a single action space is unstable and difficult to validate. The proposed framework therefore assigns only the discrete first-level decisions to RMAPPO: product-line activation, product execution, gated PM, and End-Shift.

The post-hoc LP/MILP stage is selected for the second level because it receives a fixed RL routing mask. Once those binary routing decisions are fixed, the remaining quantity-allocation problem is much smaller and can be solved directly for production quantities, inventory, and backlog. This lets the system combine the speed and generalisation of learned scheduling with the feasibility discipline of mathematical programming.

RMAPPO is selected as the policy algorithm because the environment is cooperative, multi-agent, partially observable, and recurrent. The Process Agent has a dedicated actor for global allocation masks, while the Machine Agents share a recurrent actor differentiated by line identity. A centralised critic supports training, and decentralised actors are used in deterministic evaluation.


## **3.2. Proposed System Design or Proposed Solution Approach:**

The proposed system is organised as a three-step hybrid pipeline. The pipeline separates the discrete scheduling problem from the continuous lot-sizing problem: RMAPPO learns the high-level assignment and sequencing policy, while the post-hoc lot-sizing model computes production quantities from the learned routing masks. This structure avoids the old monolithic design in which the RL policy would have to learn both routing and exact quantities directly.

**Inputs.** Each benchmark instance provides the number of products, lines, and periods; demand profile; capacity; eligibility matrix; processing times; setup times and costs; production costs; preventive-maintenance parameters; corrective-maintenance cost and hazard rates; and training parameters. These data define the production environment used for training and evaluation.

**Step 1 - Multi-agent RL training (first-level optimisation).** The first level solves the routing and sequencing problem. The Process Agent A0 learns product-line activation decisions as a binary L x P mask. Machine agents A1, ..., AL learn local production, preventive-maintenance, and End-Shift actions. During training, a just-in-time allocator creates temporary queues from remaining demand; it is not the final lot-sizing method.

**Step 2 - Post-hoc LP/MILP lot sizing (second-level optimisation).** After training, deterministic inference exports the RL allocation masks. The post-hoc lot-sizing model then takes the frozen binary routing masks as fixed input and solves the quantity allocation problem. Because routing is fixed, the difficult binary assignment part is removed from the quantity model; the remaining problem is a tractable continuous lot-sizing problem with inventory, backlog, production, expected CM, eligibility, and capacity constraints. The output is a feasible production quantity plan by line, product, and period.

**Step 3 - Resolve the first-level schedule with corrected quantities.** The trained RMAPPO policies are run again with the post-hoc lot sizes loaded into the environment. The decentralised actors execute sequence-dependent setups, production, gated PM, and End-Shift actions without further weight updates. This final simulation evaluates production, setup, maintenance, backlog, and inventory costs and produces the final per-instance total cost.

This design is a two-level optimisation architecture: the multi-agent RL system solves assignment and sequencing, while post-hoc lot sizing finds the best quantity trade-off under the RL routing mask. It is computationally tractable because the expensive rolling-horizon baseline is replaced by fast learned inference plus a reduced lot-sizing solve.

**Centralised training with decentralised execution.** During training, a centralised critic uses the full shared state to compute advantages. During execution, only the decentralised actors are used: each agent acts from its local observation and recurrent hidden state. The manager has a dedicated policy, while all machine agents share one policy differentiated by a one-hot line-identity feature.

**Gated preventive maintenance.** The benchmark and sensitivity runs use the gated PM mechanism. PM is available only when capacity permits and either no productive work is available, or the line has produced during the period and its risk score lambda_l x Age_l reaches the maintenance threshold.

The proposed system is illustrated in Figure 3.1.

*Figure 3.1: Proposed solution approach - RMAPPO system with post-hoc LP/MILP lot-sizing integration*

---

# **Chapter 4: SOLUTION DEVELOPMENT**

## **4.1. Prototype Solution**


### **4.1.1. Problem Formulation**

The proposed problem is a two-level optimisation problem for parallel-line capacitated lot sizing and scheduling with sequence-dependent setups and preventive maintenance. The first level is a multi-agent RL scheduling problem: decide which products should be assigned to which lines and how each line should sequence production, PM, and shift termination over micro-steps. The second level is a post-hoc lot-sizing problem: once the learned binary routing masks are fixed, compute feasible quantities that minimise operating cost.

The multi-agent RL system solves the assignment and sequencing problem. The post-hoc LP/MILP lot-sizing stage finds the cost trade-off among inventory, backlog, production, and expected corrective maintenance under the fixed routing mask. The final Step 3 simulation then resolves the first-level schedule using the corrected lot sizes.

The framework therefore uses a three-step benchmark pipeline rather than a single end-to-end mathematical programme. One evaluation measures the trained policy with the first-level training allocator, while the full evaluation measures the complete pipeline after allocation export and post-hoc lot sizing. Both are compared with RH2 using the same benchmark instance files.

### **4.1.2. Mathematical Model**

###### Table 4.1: Sets and Indices

| Symbol | Definition |
|:---|:---|
| L = {1, ..., L} | Set of parallel production lines |
| P = {1, ..., P} | Set of distinct products |
| T = {0, ..., T - 1} | Set of discrete planning periods |
| l in L | Index for a production line |
| p in P | Index for a product |
| t in T | Index for a planning period |
| k | Index for a machine micro-step within a period |


###### Table 4.2: System Parameters

| Symbol | Definition |
|:---|----|
| D[p,t] | Customer demand for product p in period t (units) |
| C[l] | Maximum daily operational capacity of line l (hours) |
| pt[l,p] | Processing time per unit of product p on line l (hours/unit) |
| st[l,i,j] | Sequence-dependent setup time on line l when switching from product i to j (hours) |
| E[l,p] in {0, 1} | Eligibility indicator: 1 if line l can produce product p, else 0 |
| k_max | Maximum number of machine micro-steps per planning period |
| k_w | Demand lookahead horizon (number of future periods) |
| w | Post-hoc lot-sizing rolling window (number of lookahead periods for quantity allocation) |


###### Table 4.3: Cost Parameters

| Symbol | Definition |
|:---|:---|
| h | Holding cost per unit of inventory per period |
| b | Backlog cost per unit of unmet demand per period |
| beta[p] | Flat backlog penalty per period when product p has any outstanding backlog |
| sc[l,i,j] | Setup cost incurred on line l when switching from product i to j |
| ProdCost[l,p] | Variable production cost per unit of product p on line l |
| pm[l] | Cost of performing preventive maintenance on line l |
| cm[l] | Cost per expected corrective maintenance event on line l |
| lambda[l] | Hazard rate of line l (expected failures per unit of active runtime) |
| M_RL[l,p,t] | Fixed binary routing mask exported from the trained RL policy |
| setup_reserve_RL[l,t] | Setup-time overhead reserved from the RL mask/sequence before quantity allocation |


###### Table 4.4: Decision Variables

| Symbol | Type | Definition |
|:---|:---|:---|
| a[0,t] in {0, 1}^(L x P) | Binary matrix | Manager activation mask: a[0,t,l,p] = 1 if line l is activated for product p in period t |
| Q[l,p,t] >= 0 | Continuous | Production quantity allocated to line l's queue for product p in period t by the post-hoc lot-sizing stage |
| x[l,p,t] >= 0 | Continuous | Post-hoc lot-sizing quantity of product p allocated to line l in period t |
| M_RL[l,p,t] in {0, 1} | Fixed input | Exported RL activation mask; production is allowed only when this value is 1 |
| inv[t,p] >= 0 | Continuous | Post-hoc lot-sizing inventory state for product p in period t |
| back[t,p] >= 0 | Continuous | Post-hoc lot-sizing backlog slack for product p in period t |
| I[p,t] >= 0 | Continuous | Actual inventory of product p at end of period t |
| B[p,t] >= 0 | Continuous | Actual backlog of product p at end of period t |
| Age[l,t] >= 0 | Continuous | Cumulative degradation runtime of line l at period t |


**Objective Function**

The implemented framework has two cost expressions. The post-hoc lot-sizing LP minimises quantity-related cost under the fixed RL routing mask:

Equation (1):

$$
\min \sum_{t \in \mathcal{T}} \left[
\sum_{p \in \mathcal{P}} \left(h\,\mathrm{inv}_{t,p}+b\,\mathrm{back}_{t,p}\right)
+\sum_{l \in \mathcal{L}}\sum_{p \in \mathcal{P}} M^{RL}_{l,p,t}E_{l,p}
\left(\mathrm{ProdCost}_{l,p}+\lambda_l cm_l pt_{l,p}\right)x_{l,p,t}
\right]
$$

The final reported evaluation cost is then computed by deterministic simulation of the trained actors with the post-hoc lot sizes loaded:

Equation (2):

$$
\min \sum_{t\in\mathcal{T}}\left[
\sum_{p\in\mathcal{P}}\left(h\cdot I_{p,t}+b\cdot B_{p,t}+\beta_p\cdot\mathbf{1}[B_{p,t}>0]\right)
+\sum_{l\in\mathcal{L}}\sum_{p\in\mathcal{P}} \mathrm{ProdCost}_{l,p}\cdot Q_{l,p,t}
+\sum_{l\in\mathcal{L}}\left(C^{setup}_{l,t}+C^{PM}_{l,t}+C^{CM}_{l,t}\right)
\right]
$$

where the simulation cost components are defined as:

Equation (3):

$$
C^{setup}_{l,t}=\sum_{i\ne j} sc_{l,i,j}\cdot \mathbf{1}[\text{changeover } i\rightarrow j \text{ occurs on line } l \text{ in period } t]
$$

Equation (4):

$$
C^{PM}_{l,t}=pm_l\cdot\mathbf{1}[\text{PM performed on line }l\text{ in period }t]
$$

Equation (5):

$$
C^{CM}_{l,t}=\lambda_l\cdot cm_l\cdot\sum_{p\in\mathcal{P}} Q_{l,p,t}\cdot pt_{l,p}
$$

The post-hoc LP does not optimise setup and PM actions directly. Setup time is used as a capacity reservation derived from the fixed RL routing mask, while setup and PM costs are measured in the final simulation. The CM cost formulation uses a deterministic expectation based on hazard rate and active processing time.

**Constraints:**

**Post-hoc LP inventory and backlog balance:** In the post-hoc lot-sizing model, backlog is a nonnegative slack variable for the current lookahead period. The LP carries inventory forward but does not carry backlog forward inside the post-hoc state:

Equation (6):

$$
\mathrm{inv}_{t-1,p}+\sum_{l\in\mathcal{L}}x_{l,p,t}-\mathrm{inv}_{t,p}+\mathrm{back}_{t,p}=D_{p,t},\quad \forall p,t
$$

For final simulation, actual inventory and backlog are updated from realised production quantities:

Equation (7):

$$
I_{p,t}=\max\left(0, I_{p,t-1}+\sum_{l\in\mathcal{L}}Q_{l,p,t}-D_{p,t}-B_{p,t-1}\right)
$$

Equation (8):

$$
B_{p,t}=\max\left(0, D_{p,t}+B_{p,t-1}-I_{p,t-1}-\sum_{l\in\mathcal{L}}Q_{l,p,t}\right)
$$

**Fixed routing and eligibility constraints:** The post-hoc LP can allocate production only to line-product pairs activated by the RL mask and allowed by the eligibility matrix:

Equation (9):

$$
x_{l,p,t}=0 \quad \text{if } M^{RL}_{l,p,t}=0 \text{ or } E_{l,p}=0
$$

**Capacity constraint:** The post-hoc LP reserves setup overhead implied by the RL mask/sequence before allocating processing time:

Equation (10):

$$
\sum_{p\in\mathcal{P}}pt_{l,p}x_{l,p,t}\leq\max\left(0,C_l-setup\_reserve^{RL}_{l,t}\right),\quad \forall l,t
$$

**Machine age dynamics:**

Equation (11):

$$
\mathrm{Age}_{l,t}=\begin{cases}
0, & \text{if PM is performed on line }l\text{ in period }t,\\
\mathrm{Age}_{l,t-1}+\sum_{p\in\mathcal{P}}Q_{l,p,t}pt_{l,p}, & \text{otherwise.}
\end{cases}
$$

## **4.2. Two-Level Optimization**

The production system contains 1+L agents: one Process Agent and one Machine Agent per production line. The framework follows a three-step flow.

**Step 1 - first-level optimisation: multi-agent RL training.** RMAPPO policies are trained for the selected benchmark size. The Process Agent learns product-line allocation as a binary mask over (l,p) pairs. Machine agents learn product execution, gated PM, and End-Shift actions. During training, activating a pair [l,p]=1 causes the just-in-time allocator to load the remaining-demand quota for that product into the line queue. This is a training-time queueing mechanism, not the final lot-sizing solver.

**Step 2 - second-level optimisation: post-hoc lot sizing.** After training, deterministic inference exports the RL allocation masks. The post-hoc lot-sizing model receives those masks and solves the quantity model. Since the binary routing mask is fixed, the post-hoc model only decides continuous production quantities, inventory, and backlog. It respects eligibility and capacity limits, subtracts conservative setup overhead from capacity, and minimises holding, backlog, production, and expected corrective-maintenance costs.

**Step 3 - resolve first-level optimisation.** The final inference simulation loads the post-hoc lot sizes back into the production environment. The pre-trained actors run with no weight updates. The centralised critic is not used during this evaluation; only the decentralised actor networks choose actions from local observations. Machine agents drain the physical queues created from the post-hoc lot sizes and execute sequence-dependent setups and PM. The resulting inventory, backlog, setup, maintenance, and production costs form the final total cost.

### **4.2.1. Step 1: First-Level Optimization - Multi-Agent RL System (Routing and Sequencing)**

The first level trains the RMAPPO scheduling policy. The Process Agent performs routing by selecting the active line-product pairs for each period. Machine agents perform sequencing by deciding which queued product to process, when to execute gated preventive maintenance, and when to end the shift. The first level therefore covers the discrete scheduling decisions, while exact production quantities are left to the second-level post-hoc lot-sizing model.

#### **4.2.1.1. POMDP Formulation**

The multi-agent production scheduling problem is formalised as a cooperative Partially Observable Markov Decision Process (POMDP), defined by the tuple (S, {O_i}, {A_i}, T, {R_i}, gamma), where:

- S is the global state space, consisting of inventory levels, backlogs, machine ages, production queues, setup states, and remaining periods.
- O_i is the local observation space for agent i; each agent receives a partial view of the global state.
- A_i is the action space of agent i.
- T is the environment transition function from the current state and joint action to the next state.
- R_i is the reward function for agent i.
- gamma is the discount factor.

At each time step t, agent i observes o_i,t, which is a partial view of s_t, and selects action a_i,t from its actor policy conditioned on o_i,t and hidden state h_i,t. The hidden state denotes the GRU memory of past observations within the episode.

The partial observability arises structurally: machine agents cannot observe queue states on other lines, and agents do not observe the internal post-hoc lot-sizing solve during Step 1 training. This makes recurrent policies necessary, as agents must infer latent state from the history of their own observations.

#### **4.2.1.2. CTDE Framework**

RMAPPO adopts the Centralised Training with Decentralised Execution (CTDE) paradigm. During training, a centralised critic V_phi(s_t) has access to the full global state s_t and produces value estimates used to compute advantages. During execution, each agent's actor pi_theta_i uses only its local observation o_i,t and hidden state h_i,t, making the policy implementable in a fully decentralised manner.

The global state fed to the centralised critic is the concatenation of all agents' observations:

Equation (16):

$$
s_t=[o_{0,t}\parallel o_{1,t}\parallel\cdots\parallel o_{L,t}]\in\mathbb{R}^{dN}
$$

where d is the per-agent observation dimension and N = 1 + L is the total number of agents.

#### **4.2.1.3. Joint Policy and Recurrent Actor**

Under the CTDE assumption of conditional independence given local observations, the joint policy factorises as:

Equation (17):

$$
\pi_{\theta}(a_t\mid o_t)=\prod_{i=0}^{L}\pi_{\theta_i}(a_{i,t}\mid o_{i,t},h_{i,t})
$$

where a_t = (a0,t, a1,t, ..., aL,t) is the joint action vector and o_t = (o0,t, ..., oL,t) is the joint observation. In this work, the manager maintains a dedicated policy pi_theta_0, while all machine agents share a single policy pi_theta_m differentiated by a line-identity one-hot embedding in the observation. The joint policy thus involves two distinct parameter sets: theta_0 and theta_m.

Each recurrent actor is implemented as an MLP feature extractor followed by a single-layer Gated Recurrent Unit (GRU):

Equation (18):

$$
f_{i,t}=\mathrm{MLP}_{\theta_i}(o_{i,t}),\quad h_{i,t}=\mathrm{GRU}_{\theta_i}(f_{i,t},h_{i,t-1})
$$

Equation (19):

$$
a_{i,t}\sim\pi_{\theta_i}(\cdot\mid h_{i,t})
$$

The GRU hidden state h_i,t has dimension 256 and is reset to zero at episode boundaries, allowing the agent to maintain a within-episode belief state across the sequential micro-steps of each production period.

#### **4.2.1.4. State and Observation Spaces**

All agents receive an observation vector of identical dimension d, enabling the shared machine policy architecture without requiring separate network definitions per agent. Segments outside an agent's operational scope are zero-filled.

The observation dimension is:

Equation (30):

$$
d=2P+LP+3P+k_wP+1+L+LP+2L+L+P+LP+2P
$$

The terms correspond to inventory/backlog, queues, coverage/queue-total/shortfall, demand lookahead, remaining periods, line availability, line setup, ages/line identity, contention, urgency, eligibility, scarcity, and lines-needed features.

###### Table 4.6: Observation Vector Layout

| Segment | Dim | Description | Process Agent A0 | Machine Agent Al |
|:---|:---:|:---|:---:|:---:|
| inv | P | Inventory signal, binary in the active benchmark setting or log-scaled in continuous mode | Yes | Yes |
| back | P | Backlog signal, binary in the active benchmark setting or log-scaled in continuous mode | Yes | Yes |
| queue | L P | Production queue signal | Full matrix | Own row only |
| coverage | P | Fraction of lines holding any queue per product | Yes | Zero |
| queue_total | P | Total queued quantity per product | Yes | Zero |
| shortfall | P | Net unmet demand signal | Yes | Zero |
| demand_window | k_w P | Demand lookahead over the forecast window | Yes | Yes |
| remaining_periods | 1 | Remaining periods in the planning horizon | Yes | Yes |
| line_avail | L | Line availability flags | Yes | Yes |
| line_setup | L P | One-hot last-product state per line | Yes | Yes |
| ages | L | Machine degradation ages | Yes | Yes |
| line_id | L | One-hot identity for machine agents | Zero | Yes |
| contention | L | Per-line product contention score | Yes | Zero |
| urgency | P | Per-product demand urgency | Yes | Zero |
| eligibility | L P | Static eligibility matrix | Yes | Zero |
| scarcity | P | Product scarcity based on eligible lines | Yes | Zero |
| lines_needed | P | Periods of capacity needed to clear backlog | Yes | Zero |


**Observation scaling.** The environment supports both continuous log-scaled observations and binary observations. In the active benchmark setting, inventory, backlog, queue, demand-window, and shortfall features are represented as present/absent signals; the formulas below describe the continuous/log-scaled counterpart:

Equation (31):

$$
\mathrm{inv}_p=\ln(1+I_{p,t}),\quad \mathrm{back}_p=\ln(1+B_{p,t})
$$

**Shortfall** provides the Process Agent with an urgency signal that accounts for existing queue allocations. In binary mode this becomes a positive/zero indicator; in continuous mode it is:

Equation (32):

$$
\mathrm{shortfall}_p=\ln\left(1+\max\left(0,FD_{p,t}+B_{p,t}-I_{p,t}-\sum_{l\in\mathcal{L}}Q_{l,p,t}\right)\right)
$$

**Product scarcity** encodes structural supply risk arising from limited line eligibility:

Equation (33):

$$
\mathrm{scarcity}_p=\frac{1}{\sum_{l\in\mathcal{L}}E_{l,p}}
$$

**Lines needed** quantifies how many periods of full-factory capacity would be required to eliminate the current backlog:

Equation (34):

$$
\mathrm{lines\_needed}_p=\min\left(\frac{B_{p,t}}{\mathrm{CapUnits}_p},T\right),\quad \mathrm{CapUnits}_p=\sum_{l:E_{l,p}=1}\frac{C_l}{pt_{l,p}}
$$

**Product urgency** encodes the proximity of the next non-zero demand event within the lookahead window:

Equation (35):

$$
\mathrm{urgency}_p=\frac{1}{d_p^*+1}
$$

**Line contention** measures how many of a line's eligible products currently have active demand, normalised by the total number of products:

Equation (36):

$$
\mathrm{contention}_l=\frac{|\{p:E_{l,p}=1\land D_{t,p}>0\}|}{P}
$$

The **one-hot line identity** line\_id = e_l in R^L for machine agent A_l is the key feature enabling the shared machine policy to learn heterogeneous per-line behaviour while sharing a single set of network weights. Because setup times, hazard rates, and production costs differ per line, the policy must condition its decisions on which line it is controlling; the identity embedding provides this information explicitly.

#### **4.2.1.5. Action Pool**

##### **4.2.1.5.1. Process Agent Action Space (A0)**

The Process Agent's action space is a multi-discrete binary space, one dimension per (l,p) pair:

Equation (37):

$$
A_0=\{0,1\}^{L\times P}
$$

where a0,t[l,p] = 1 signals that line l should be activated for product p in period t. The action is encoded as a concatenation of L x P two-element one-hot vectors, giving a total action representation of length 2LP.

**Action pool.** The Process Agent selects one binary decision for every (l,p) pair simultaneously, forming an activation mask of shape L x P. There are two possible values per entry:

Activate (1). Line l is assigned to produce product p in period t. During Step 1, the JIT allocator queues demand for active slots; during Step 2, the post-hoc LP/MILP allocates quantity Q_l,p,t >= 0 to active slots subject to capacity, inventory balance, and eligibility constraints. Setting this to 1 on an ineligible pair (E_l,p = 0) is structurally prevented by the action mask.

Deactivate (0). Line l is not assigned to product p in period t. During Step 1, no JIT queue is created from that activation slot. During Step 2, the post-hoc lot-sizing model fixes the corresponding quantity to zero unless the pair appears in the exported active mask. This is the default state for all ineligible pairs and for eligible pairs the manager chooses not to activate.

Eligibility constraints are enforced via a hard availability mask applied before sampling: any (l,p) pair with E_l,p = 0 is constrained to the deactivate state by setting its logit to - 10^10, guaranteeing zero sampling probability without requiring explicit penalty rewards. The design motivation is the separation of concerns between routing and quantity decisions: the Process Agent determines where production effort should be directed, while the JIT allocator supports fast training and the post-hoc LP/MILP determines final production quantities given that routing. This reduces the effective action space from O(H^LP) - if the manager specified quantities directly - to O(2^LP), enabling the RL policy to focus on the combinatorial routing problem.


##### **4.2.1.5.2. Machine Agent Action Space (A1, ..., AL)**

Each machine agent controls one production line and uses a discrete action space of P+2 choices:

Equation (38):

$$
A_l=\{0,1,\ldots,P-1,PM,EndShift\}
$$

**Product execution actions.** A product action is valid only when the product is eligible on the line, a positive queue or demand-based feasible quantity exists, and the remaining capacity can cover the required setup plus at least one unit of production. When the action is executed, the simulator applies the sequence-dependent setup time and cost when the selected product differs from the previous line setup, produces as many units as capacity allows, reduces the queue, updates produced quantities, and increases line age by the active runtime.

**Preventive Maintenance (PM).** PM consumes maintenance time, adds maintenance cost, and resets the line age to zero. In the benchmark configuration, PM is controlled by a gated availability rule. Under this rule, PM is available only if enough capacity remains and either no productive work is available or the line has already produced during the period and lambda_l Age_l >= tau, where tau is the PM risk threshold. Alternative configurations can leave PM available whenever capacity permits or remove the PM action entirely.

**End-Shift.** End-Shift is available when no product action can be executed. If the line is already marked done, End-Shift remains the only valid action. This prevents machine agents from ending a period while producible work is still available.

**Action masking protocol.** The environment recomputes the action mask at every micro-step. Invalid product, PM, and End-Shift choices are masked before the policy samples or selects an action, so infeasible actions do not need to be learned through penalty rewards.


#### **4.2.1.6. Reward Design**

The benchmark training uses the Step 1 reward design. This mode is different from the older separated manager-machine reward description. It applies a shared team reward to all agents and adds a proportional unmet-demand kill-switch penalty.

At period close, the environment computes inventory, backlog, production, setup, PM, and expected CM costs. If backlog remains, every agent receives a penalty proportional to total unmet demand:

Equation (39):

$$
R^{kill}_t=-\kappa\sum_{p\in\mathcal{P}}B_{p,t}
$$

where kappa is the kill-switch penalty coefficient. For P >= 10, the benchmark setting uses kappa=20 per unmet unit. For smaller instances, it scales the P=5 base value proportionally with product count.

The team reward also penalises total active processing time and worker-side operational costs:

Equation (40):

$$
R^{team}_t=-\left(ProcTime_t+C^{setup}_t+C^{PM}_t+C^{CM}_t\right)
$$

Optional activation and load-balance penalties may be applied if their coefficients are non-zero, but the benchmark pipeline leaves both at zero. Alternative reward configurations can still use the older manager reward based on inventory, backlog, production, and weighted worker costs; however, that is not the active configuration used in the reported experiments.

Dense machine-step shaping remains active during production execution. Product actions receive reward proportional to productive capacity consumed, setup actions subtract a setup-cost penalty, and PM actions subtract a maintenance-cost penalty. Expected CM cost is accumulated from hazard rate and active runtime. The Step 1 reward design intentionally makes unmet demand expensive while still teaching the agents to avoid inefficient setup and maintenance behaviour.


#### **4.2.1.7. GAE and Normalization**

Advantages are estimated using Generalised Advantage Estimation (GAE), which interpolates between high-variance Monte Carlo returns and high-bias one-step temporal difference targets:

Equation (20):

$$
\widehat{A}_t=\sum_{n\geq 0}(\gamma\lambda_{GAE})^n\delta_{t+n}
$$

where the one-step TD error is:

Equation (21):

$$
\delta_t=r_t+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_t)
$$

with gamma the discount factor and lambda_GAE the GAE smoothing parameter controlling the bias-variance trade-off. When lambda_GAE = 1, the estimator recovers the full Monte Carlo return; when lambda_GAE = 0, it reduces to the one-step TD error. In practice, lambda_GAE = 0.95 is used, providing a low-variance estimate at modest bias cost.

Since agents have different activity patterns (the manager acts once per period while machine agents act up to k_max times), advantages are normalised separately per agent using only the active time steps:

Equation (22):

$$
\widetilde{A}_{i,t}=\frac{\widehat{A}_{i,t}-\mu_i^{active}}{\sigma_i^{active}+\epsilon}
$$

where mu_i^active and sigma_i^active are the mean and standard deviation of advantages computed exclusively over time steps where agent i was active, and epsilon = 10^-5 is a numerical stability constant.

#### **4.2.1.8. Actor Objective: Clipping and Entropy**

The actor is updated by maximising the PPO clipped surrogate objective, which constrains the policy update to a trust region without requiring second-order information:

Equation (23):

$$
L^{CLIP}(\theta_i)=\mathbb{E}_t\left[m_{i,t}\min\left(\rho_{i,t}\widetilde{A}_{i,t},\mathrm{clip}(\rho_{i,t},1-\epsilon_{clip},1+\epsilon_{clip})\widetilde{A}_{i,t}\right)\right]
$$

where the importance sampling ratio is:

Equation (24):

$$
\rho_{i,t}=\frac{\pi_{\theta_i}(a_{i,t}\mid o_{i,t},h_{i,t})}{\pi_{old,i}(a_{i,t}\mid o_{i,t},h_{i,t})}
$$

and m_i,t in 0,1 is the active mask for agent i at time step t, and epsilon_clip = 0.2 is the clipping threshold. The expectation is computed over time steps and environments in the mini-batch. The minimum operation and clipping together prevent both overly optimistic updates when advantages are positive and overly pessimistic updates when advantages are negative, bounding the effective KL divergence between successive policy iterates.

An entropy regularisation term is added to the actor objective to encourage exploration and prevent premature policy collapse:

Equation (28):

$$
\mathcal{H}[\pi_{\theta_i}]=-\mathbb{E}\left[\log\pi_{\theta_i}(a\mid o_{i,t},h_{i,t})\right]
$$

#### **4.2.1.9. Centralized Critic and Value Loss**

The centralised critic V_phi(s_t) is implemented as a separate recurrent network receiving the full global state s_t. To handle the large and non-stationary scale of the cost-based reward signal, the value targets are normalised using a running-mean-variance normaliser (ValueNorm) with momentum beta_vn = 0.99999:

Equation (25):

$$
\widehat{R}^{norm}_t=\frac{\widehat{R}_t-\mu_R}{\sqrt{\sigma_R^2+\epsilon}}
$$

where R_hat_t = r_t + gamma x R_hat_t+1 is the bootstrapped return and mu, sigma^2 are the running statistics of observed returns.

The value loss uses a clipped Huber formulation to simultaneously prevent value collapse and reduce sensitivity to large TD errors:

Equation (26):

$$
L^V(\phi)=\mathbb{E}_t\left[\max\left(\mathrm{Huber}(V_{\phi}(s_t)-\widehat{R}^{norm}_t),\mathrm{Huber}(V^{clip}_{\phi}(s_t)-\widehat{R}^{norm}_t)\right)\right]
$$

where V_phi^clip(s_t) = V_phi^old(s_t) + clip(V_phi(s_t) - V_phi^old(s_t), - epsilon_clip, epsilon_clip) is the clipped value prediction, and the Huber loss is:

Equation (27):

$$
\mathrm{Huber}(e)=\begin{cases}
0.5e^2, & |e|\leq\delta,\\
\delta(|e|-0.5\delta), & |e|>\delta.
\end{cases}
$$

with delta = 10.0 as the transition threshold.

#### **4.2.1.10. Combined Optimization Objective**

The combined optimisation objective for agent i is:

Equation (29):

$$
\mathcal{L}(\theta_i,\phi)=-L^{CLIP}(\theta_i)-\eta\mathcal{H}[\pi_{\theta_i}]+c_vL^V(\phi)
$$

where eta is the entropy coefficient and c_v is the value loss coefficient. Both actor and critic parameters are updated by minimising this combined objective via Adam optimisation with gradient clipping to the configured maximum gradient norm.

#### **4.2.1.11. Parameters and Training Configuration**

The benchmark pipeline uses a tuned P=5 hyperparameter set, with large-instance overrides for entropy and the backlog kill-switch. The default training budget is determined by product count: 500,000 environment steps for small instances (P <= 13), 1,000,000 for medium instances (14 <= P <= 22), and 3,000,000 for large instances (P >= 23).

###### Table 4.7: RMAPPO Hyperparameters

| Parameter | Value used in experiments | Description |
|----|:---|:---|
| Algorithm | RMAPPO | Recurrent multi-agent PPO implementation |
| Hidden size | 128 | Hidden dimension for actor/critic networks |
| Recurrent layers | 1 | Number of recurrent layers |
| Actor learning rate | 0.00026918605816868973 | Learning rate for the actor |
| Critic learning rate | 0.00009359933007344967 | Learning rate for the critic |
| Entropy coefficient | 0.0017196365 for P<10; 0.01 for P>=10 | Exploration bonus coefficient |
| PPO clip threshold | 0.3494944414 | PPO probability-ratio clipping threshold |
| PPO epochs | 6 | PPO update epochs per rollout |
| GAE lambda | 0.9624779422 | GAE smoothing parameter |
| Discount factor | 0.9626502159 | Future reward discount factor |
| Mini-batches | 1 | Mini-batches per PPO update |
| Rollout environments | 8 | Parallel rollout environments |
| Training worker threads | 1 | Training worker threads |


###### Table 4.8: Environment and Reward Configuration

| Parameter | Value used in benchmark pipeline | Description |
|:---|:---|:---|
| Demand lookahead | 4 | Demand forecast window in observation |
| Maximum micro-steps per period | 8 | Maximum machine actions per period |
| Reward design | Step 1 team reward | Team reward plus proportional unmet-demand penalty |
| Training allocator | Just-in-time allocator | Fast training allocator before post-hoc lot sizing |
| Observation encoding | Binary | Quantity-blind observation encoding |
| Dense production reward weight | 2.1194776779870743 | Productive-capacity reward weight |
| Setup penalty coefficient | 0.0 in benchmark training | Setup penalty coefficient passed during training |
| Kill-switch penalty | scaled P=5 value for P<10; 20 for P>=10 | Proportional unmet-demand penalty |
| PM action availability | Gated | Controls PM action availability |
| PM risk threshold | 1.0 by default | Risk threshold for gated PM |
| Result tag | optional | Adds suffix to model/result directories |


The saved model directory contains one actor for the manager and one shared actor for the machine agents. When available, the pipeline also stores the configuration so inference utilities can recover architecture and environment settings.

The system should therefore be described as a benchmark automation framework with reusable controls for training, comparison, RH2 skipping/backfilling, tagged result directories, and sensitivity sweeps, rather than as one fixed-budget experiment.

**Training procedure.** The following pseudocode summarises the RMAPPO training loop before the post-hoc evaluation stages.

```text
Algorithm 1: RMAPPO Training Procedure
1:  Initialise the production environment with Step 1 reward design, JIT training allocation,
    binary observations, a four-period demand lookahead, and eight machine micro-steps per period
2:  Initialise Process Agent actor pi_theta_0, shared Machine Agent actor pi_theta_m,
    centralised critic V_phi, recurrent hidden states, and rollout buffers
3:  for episode = 1, ..., N_episode do
4:      Reset environment, inventory, backlog, queues, machine ages, setup states, and hidden states
5:      for period t = 1, ..., T do
6:          Build manager observation and apply the eligible line-product mask
7:          Select Process Agent binary allocation mask a_0,t
8:          Use the JIT allocator to load temporary line queues from remaining demand
9:          for micro-step k = 1, ..., k_max do
10:             for each machine agent l = 1, ..., L do
11:                 Build local observation and action mask
12:                 Select product, PM, or End-Shift action
13:                 Execute action and update capacity, queue, setup state, age, cost, and buffer
14:             end for
15:             if all lines selected End-Shift or no valid productive action remains then break
16:         end for
17:         Update inventory/backlog and apply team reward plus kill-switch penalty
18:     end for
19:     Compute bootstrapped value, GAE returns, and normalized advantages
20:     for each agent and PPO epoch do
21:         Re-evaluate actions, compute clipped actor loss, value loss, entropy, and combined loss
22:         Update actor and critic parameters via Adam with gradient clipping
23:     end for
24:     Log metrics and save checkpoints at the configured interval
25: end for
```

*Figure 4.2: Deep Reinforcement Learning Architecture*

### **4.2.2. Step 2: Second-Level Optimization - Post-hoc LP Lot Sizing**

The post-hoc lot-sizing stage is the second-level optimiser. Its input is the fixed binary routing mask exported from Step 1. The allocation data may be period-level or period-by-micro-step. For the period-by-step format, the solver forms the union of active products in a period and restricts production to active, eligible line-product pairs.

Once the binary mask is fixed, the quantity problem contains continuous production, inventory, and backlog variables. Backlog is allowed with penalty so the model remains feasible even when the RL mask is restrictive. For each line, production time plus setup overhead must fit within available capacity. The setup overhead is conservative: the model uses the active-product set to reserve capacity for possible changeovers instead of assuming that the LP itself chooses the sequence.

The objective minimises inventory holding, backlog, direct production, and expected corrective-maintenance costs. Expected CM cost is deterministic and proportional to hazard rate, CM cost, processing time, and produced quantity. The second level therefore finds the quantity cost trade-off while leaving sequencing to the trained RL policy.

### **4.2.3. STEP 3: Resolve the First-Level Optimization**

After Step 2, the final inference simulation loads the post-hoc lot sizes back into the production environment. The pre-trained actors run with no weight updates. The centralised critic is not used during this evaluation; only the decentralised actor networks choose actions from local observations.

During final inference, the action-availability logic checks the loaded queues before enabling product actions. A machine can process a product only when the product is eligible, the queue is positive, and enough capacity remains after required setup. PM and End-Shift are also masked from the current line state. Invalid actions receive zero probability through the availability mask before action selection.

Machine agents drain the physical queues created from the post-hoc lot sizes and execute sequence-dependent setups and PM. The resulting inventory, backlog, setup, maintenance, and production costs form the final reported total cost.

# **CHAPTER 5: RESULT ANALYSIS**

## **5.1. Experimental Design**

To develop a generalizable scheduling policy, the system is trained across a diverse distribution of problem instances rather than a single fixed scenario. During training, a new random instance is generated at the start of each episode by sampling all cost, demand, and production parameters from predefined distributions. The fixed dimensions (L,P,T) are set once at initialisation and held constant throughout training to ensure observation and action space compatibility with the neural network. This approach - training across randomised instance parameters while keeping structural dimensions fixed - exposes the policy to a wide variety of supply chain conditions without requiring retraining for each new scenario.

### **5.1.1. Dataset for real-case evaluation**

To evaluate the system under realistic industrial conditions, a real-case dataset is constructed from a factory configuration. This configuration specifies a fixed set of 6 production lines and 7 products over a 30-period planning horizon, with domain-calibrated parameters for processing times, setup times, costs, capacity, and maintenance. Unlike the randomly generated training instances, the real-case dataset reflects an actual parallel-line manufacturing environment and is used to assess the generalization of the trained RMAPPO policy to a concrete deployment scenario. However, for confidentiality reasons, the specific dataset used in this study cannot be disclosed.

Furthermore, lines are not fully flexible. The eligibility matrix restricts which products each line may produce, introducing routing constraints that require the manager agent to make meaningful allocation decisions. Processing times are heterogeneous across eligible line -product pairs, reflecting real differences in line speed and product complexity.

Sequence-dependent setup times and costs apply when a line switches between eligible products. The demand profile spans 30 periods with intermittent zero-demand periods creating natural planning boundary conditions. Product 089 has the highest and most sustained demand, consistently exceeding 5,000 units in several periods. Products 083 and 111 have sparse or zero demand throughout, reducing their scheduling priority relative to the high-volume products.

Multiple independent evaluation runs are conducted by reseeding demand realizations around the fixed demand profile where stochasticity is introduced, or by treating the fixed demand profile as a deterministic benchmark with a single run. Results are compared against key reference algorithms under identical configuration loading.


### **5.1.2. Dataset for Benchmark Comparison (Small, Medium, Large)**

For benchmarking, the study uses structured generated datasets grouped into Small, Medium, and Large sets. Each benchmark size contains up to 100 fixed instances named by product count, line count, period count, and instance index. The comparison pipeline evaluates the first 10 instances by default because RH2 is expensive on medium and large sizes. Some earlier small-result files contain more completed RH2 entries because they were generated before this limit was standardised.

###### Table 5.1: Generated Instance Set

| Scale | Benchmark sizes used by scripts |
|:------|:-------------------------------|
| Small | 5×2×4, 6×2×4, 7×2×4, 8×3×4, 9×3×4, 10×3×4, 11×4×4, 12×4×4, 13×4×4 |
| Medium | 14×5×4, 15×5×4, 16×5×4, 17×6×4, 18×6×4, 19×6×4, 20×7×4, 21×7×4, 22×7×4 |
| Large | 23×8×4, 24×8×4, 25×8×4, 26×9×4, 27×9×4, 28×9×4, 29×10×4, 30×10×4, 31×10×4 |


###### Table 5.2: Parameter Settings of the Generated Instances

| Parameter | Value / Distribution |
|:---|:---|
| Daily capacity per line, C_l | 120.0 hours |
| Eligibility matrix, E_l,p | All ones for generated benchmarks |
| Processing time, pt[l,p] | Uniform [11/60, 15/60] hours/unit |
| Production cost, ProdCost[l,p] | Uniform [1.3, 1.6] per unit |
| Demand, D[p,t] | Integer uniform {0, ..., 149} units |
| Hazard rate, lambda_l | random choice from 2.0/120, 3.0/120 |
| CM time per event | Uniform [7/60, 11/60] hours |
| PM time | Uniform [2.5, 3.5] hours |
| PM cost | Uniform [10.0, 20.0] |
| CM cost | Uniform [2.0, 3.0] |
| Holding cost | 1.75 |
| Backlog cost | 5.25 |
| Flat backlog penalty | 12.25 per product with backlog |
| First setup time | Uniform [1.2, 2.4] hours |
| First setup cost | Uniform [25.0, 30.0] |
| Changeover setup time | Uniform [1.2, 2.4] hours for different products |
| Changeover setup cost | Uniform [25.0, 30.0] for different products |
| Same-product setup | 0 |


The comparison output for each benchmark is saved in the benchmark result directory. Despite the historical result-file naming convention, the reported comparison uses 10 evaluated instances by default.

## **5.2. Result Illustration and Explanation**

This section reports the benchmark and sensitivity results for the proposed framework. The comparison uses RH2 as the rolling-horizon benchmark and RMAPPO as the proposed three-step hybrid pipeline: Step 1 trains the RMAPPO routing and sequencing policies, Step 2 solves post-hoc LP/MILP lot sizing from the exported allocation mask, and Step 3 re-runs deterministic inference with the corrected quantities. The cost gap is defined as:

$$
\mathrm{Gap}=\frac{\mathrm{RH2\ Cost}-\mathrm{RMAPPO\ Cost}}{\mathrm{RH2\ Cost}}\times100\%
$$

A positive gap therefore means that RMAPPO produces a lower total cost than RH2.

### **5.2.1. Real-Case Dataset Result**

The real-case loopset dataset is retained as the industrial case-study result. It evaluates the proposed planning framework on the original production environment rather than on generated benchmark instances. The result provides a cost decomposition that shows how the final schedule distributes cost across inventory, backlog, production, setup, and maintenance components.

###### Table 5.3: Real-Case Performance Summary

| Cost component | Value ($) | Share of total cost |
|:---|---:|---:|
| Inventory holding cost | 8,392.21 | 3.2% |
| Backlog cost | 5,540.69 | 2.1% |
| Production cost | 247,552.81 | 94.5% |
| Setup cost | 282.28 | 0.1% |
| Preventive maintenance cost | 174.33 | 0.1% |
| **Total cost** | **261,942.31** | **100.0%** |


The real-case result is dominated by production cost, which is expected because production volume is the main driver of total expenditure in the industrial dataset. Inventory and backlog costs remain comparatively small, indicating that the schedule balances service fulfilment with stock accumulation. Setup and preventive-maintenance costs are low shares of total cost, but they remain operationally important because poor sequencing or poorly timed PM can reduce available capacity and indirectly increase backlog.

### **5.2.2. Benchmark Comparison Results (Small, Medium, Large)**

###### Table 5.4: Small Instance Results

| No. | Benchmark | RH2 Cost ($) | RMAPPO Cost ($) | RH2 Time (s) | RMAPPO Time (s) | Cost Gap (%) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | 5×2×4 | 2,540.5 | 2,721.3 | 1.4 | 1.18 | -7.1% |
| 2 | 6×2×4 | 2,774.5 | 3,031.8 | 3.1 | 1.25 | -9.3% |
| 3 | 7×2×4 | 3,565.9 | 3,843.6 | 1,327.5 | 2.7 | -7.8% |
| 4 | 8×3×4 | 3,934.1 | 4,296.0 | 1,531.1 | 3.2 | -9.2% |
| 5 | 9×3×4 | 4,463.4 | 4,787.1 | 908.1 | 3.3 | -7.3% |
| 6 | 10×3×4 | 4,943.8 | 5,267.5 | 912.2 | 2.9 | -6.5% |
| 7 | 11×4×4 | 6,444.2 | 5,791.4 | 4,000.0 | 2.9 | 10.1% |
| 8 | 12×4×4 | 6,479.0 | 6,200.9 | 4,000.0 | 3.4 | 4.3% |
| 9 | 13×4×4 | 10,885.8 | 7,489.6 | 4,000.0 | 4.0 | 31.2% |
| **Avg.** |  | **5,114.58** | **4,825.47** | **1,853.71** | **2.76** | **5.65%** |


Small problems do not fully benefit from learned approximation because RH2 still has enough time to find strong solutions on the easiest instances. RMAPPO underperforms RH2 on the first six small benchmarks, but begins to outperform once the routing space grows to 11×4×4 and above. Even in the small tier, RMAPPO remains much faster, producing schedules in a few seconds.

###### Table 5.5: Medium Instance Results

| No. | Benchmark | RH2 Cost ($) | RMAPPO Cost ($) | RH2 Time (s) | RMAPPO Time (s) | Cost Gap (%) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | 14×5×4 | 8,105.9 | 7,408.0 | 4,000.0 | 4.3 | 8.6% |
| 2 | 15×5×4 | 8,253.2 | 7,642.0 | 4,000.0 | 3.2 | 7.4% |
| 3 | 16×5×4 | 9,912.5 | 8,901.2 | 4,000.0 | 4.8 | 10.2% |
| 4 | 17×6×4 | 8,868.9 | 8,322.9 | 4,000.0 | 3.7 | 6.2% |
| 5 | 18×6×4 | 16,004.0 | 13,598.0 | 4,000.0 | 5.1 | 15.0% |
| 6 | 19×6×4 | 15,195.4 | 10,802.8 | 4,000.0 | 3.2 | 28.9% |
| 7 | 20×7×4 | 32,713.5 | 10,643.1 | 4,000.0 | 4.9 | 67.5% |
| 8 | 21×7×4 | 40,338.9 | 12,547.2 | 4,000.0 | 4.7 | 68.9% |
| 9 | 22×7×4 | 39,660.6 | 13,685.9 | 4,000.0 | 4.0 | 65.5% |
| **Avg.** |  | **19,894.8** | **10,394.6** | **4,000.0** | **4.21** | **47.8%** |


The medium tier shows the main crossover. RH2 reaches the 4,000-second time limit across the tier, while the RMAPPO pipeline produces decisions in roughly four seconds on average. RMAPPO outperforms RH2 on every medium benchmark, with especially large improvements from 20×7×4 onward.

###### Table 5.6: Large Instance Results

| No. | Benchmark | RH2 Cost ($) | RMAPPO Cost ($) | RH2 Time (s) | RMAPPO Time (s) | Cost Gap (%) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | 23×8×4 | 22,738.8 | 14,855.2 | 4,000.0 | 8.3 | 34.7% |
| 2 | 24×8×4 | 22,405.8 | 15,542.9 | 4,000.0 | 3.4 | 30.6% |
| 3 | 25×8×4 | 25,153.8 | 15,826.7 | 4,000.0 | 3.7 | 37.1% |
| 4 | 26×9×4 | 28,378.7 | 13,716.6 | 4,000.0 | 3.2 | 51.7% |
| 5 | 27×9×4 | 20,526.6 | 17,125.5 | 4,000.0 | 3.6 | 16.6% |
| 6 | 28×9×4 | 28,907.3 | 17,917.6 | 4,000.0 | 6.0 | 38.0% |
| 7 | 29×10×4 | 33,729.2 | 18,649.5 | 4,000.0 | 5.5 | 44.7% |
| 8 | 30×10×4 | 35,514.8 | 19,512.2 | 4,000.0 | 6.0 | 45.1% |
| 9 | 31×10×4 | 40,865.5 | 20,523.1 | 4,000.0 | 6.5 | 49.8% |
| **Avg.** |  | **28,691.2** | **17,074.4** | **4,000.0** | **5.1** | **40.5%** |


The large tier confirms the scalability advantage of the proposed method. RH2 is capped at 4,000 seconds for every large benchmark, while RMAPPO remains within seconds. The hybrid method scales better because RL handles high-level allocation quickly, and the post-hoc LP/MILP solves only the reduced lot-sizing problem conditioned on the RL routing mask.

###### Table 5.7: Summary Across Scales

| Size | Avg. RH2 Cost | Avg. RMAPPO Cost | Avg. Gap | Avg. RH2 Time (s) | Avg. RMAPPO Time (s) | Main Result |
|:---|---:|---:|---:|---:|---:|:---|
| Small | 5,114.6 | 4,825.5 | 5.7% | 1,853.7 | 2.8 | Mixed, RH2 better on smaller cases |
| Medium | 19,894.8 | 10,394.6 | 47.8% | 4,000.0 | 4.2 | RMAPPO better on all cases |
| Large | 28,691.2 | 17,074.4 | 40.5% | 4,000.0 | 5.1 | RMAPPO better on all cases |


Across benchmark sizes, RH2 performs well on small instances but becomes slow and less effective as the problem size increases. RMAPPO achieves much faster inference, typically within seconds, and outperforms RH2 on most medium and large instances.

## **5.3. Sensitivity Analysis**

The sensitivity analysis uses the representative 24×8×4 benchmark. Each setting reports the final RMAPPO cost after the three-step pipeline, the runtime in seconds, and the percentage change relative to the baseline:

$$
\mathrm{Change\ vs\ Baseline}=\frac{\mathrm{Cost}_{setting}-\mathrm{Cost}_{baseline}}{\mathrm{Cost}_{baseline}}\times100\%
$$

Negative values indicate an improvement over the baseline.

### **5.3.1. PM Gating Threshold**

This test varies when preventive maintenance becomes available under gated PM. Lower tau means PM is allowed earlier once line risk accumulates.

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: tau=1.0 | 15,542.9 | 3.4 | baseline |
| tau=0.5 | 12,805.4 | 5.3 | -17.61% |
| tau=0.75 | 14,598.5 | 6.9 | -6.08% |
| tau=1.0 | 15,472.4 | 4.7 | -0.45% |
| tau=1.25 | 15,459.6 | 4.7 | -0.54% |


The best tested setting is a PM threshold of 0.5. Allowing PM earlier prevents machine-risk accumulation while still preserving enough capacity for production. Higher thresholds behave close to the baseline.

### **5.3.2. Kill-Switch Penalty**

This test varies how strongly the policy is penalised for unmet demand and backlog during Step 1 training.

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: kill=20 | 15,542.9 | 3.4 | baseline |
| kill=10 | 15,104.5 | 9.2 | -2.82% |
| kill=40 | 15,479.6 | 5.2 | -0.41% |


A slightly lower penalty can reduce over-conservative behaviour and improve allocation. A penalty that is too high may force production choices that are costly or inflexible.

### **5.3.3. Entropy Coefficient**

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: entropy=0.01 | 15,542.9 | 3.4 | baseline |
| entropy=0.001 | 15,488.7 | 11.6 | -0.35% |
| entropy=0.005 | 12,643.6 | 4.0 | -18.65% |
| entropy=0.02 | 12,766.7 | 4.1 | -17.86% |


Very low entropy limits exploration, while higher entropy can help avoid early convergence in this benchmark. In the completed sensitivity run, the entropy coefficient of 0.005 gives the best entropy setting.

### **5.3.4. Learning Rate Scale**

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: lr scale=1.0 | 15,542.9 | 3.4 | baseline |
| lr scale=0.5 | 15,560.4 | 5.5 | 0.11% |
| lr scale=2.0 | 15,495.2 | 5.4 | -0.31% |


Learning-rate scaling has only a small effect in the completed run. The 2.0 scale gives a minor cost reduction, while the 0.5 scale is nearly identical to the baseline.

### **5.3.5. PPO Clip Parameter**

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: clip=0.349 | 15,542.9 | 3.4 | baseline |
| clip=0.2 | 13,040.6 | 3.2 | -16.10% |
| clip=0.5 | 15,514.8 | 3.3 | -0.18% |


Smaller clipping makes PPO updates more conservative and improves this benchmark run. Larger clipping remains close to the baseline but does not provide the same cost reduction as clip=0.2.

### **5.3.6. Discount Factor Gamma**

This test varies how strongly the policy values future rewards during training.

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: gamma=0.96265 | 15,542.9 | 3.4 | baseline |
| gamma=0.95 | 13,669.9 | 3.1 | -12.05% |
| gamma=0.99 | 12,844.9 | 3.1 | -17.36% |


A higher discount factor performs best in this completed run, indicating that the policy benefits from placing more weight on future inventory, backlog, and maintenance consequences.

## **5.4 Environmental, Social, and Economic Impacts**

### **5.4.1. Economic Impact**

The primary economic impact of the proposed RMAPPO scheduling system lies in its direct reduction of total supply chain costs across the planning horizon. By jointly optimising routing decisions, production quantity allocation, and maintenance scheduling, the system reduces three major categories of cost that are often managed independently in practice.

First, inventory holding costs are reduced because demand-aware RL routing and post-hoc lot sizing encourage the system to build inventory only where demand anticipation justifies the holding cost, rather than maintaining large standing inventories across all products. Second, backlog costs - which are disproportionately penalised in the objective due to the flat backlog penalty beta_p - are minimised by routing production to lines where demand urgency is highest, rather than defaulting to throughput-maximising greedy assignments. Third, maintenance costs are managed proactively: machine agents learn to schedule preventive maintenance in low-demand periods, reducing the expected frequency of corrective failures and the associated disruption costs.

Beyond direct cost reduction, the hybrid LP/MILP-MARL architecture offers economic value through scalability and adaptability. Unlike traditional rule-based schedulers that require manual re-tuning for each new product mix or demand regime, the trained RMAPPO policy generalises across randomised instance parameters, reducing the human effort required for operational scheduling. The post-hoc lot-sizing stage further improves quantity feasibility without requiring RH2 to solve the full rolling-horizon problem from scratch for every instance.

### **5.4.2. Social Impact**

From a social perspective, improved production scheduling has implications for workforce planning and job quality in manufacturing environments. Predictable, demand-aligned scheduling reduces the need for emergency overtime shifts and last-minute production changes, which are common sources of worker dissatisfaction and fatigue in high-variety manufacturing settings. By learning to balance production loads across lines and manage maintenance proactively, the system supports more stable shift patterns and reduces the frequency of unplanned downtime events that create uncertainty for workers.

Furthermore, the reduction in backlog - a consequence of better demand anticipation - translates directly into improved service reliability for downstream customers. In supply chains serving essential goods, consistent fulfilment of demand has meaningful social value beyond the economic penalty captured in the cost function.

The automation of scheduling decisions also raises considerations around the role of human operators. The proposed system is designed as a decision-support tool rather than a fully autonomous controller: the MILP outputs and policy recommendations can be reviewed and overridden by human planners, preserving operator agency while reducing cognitive burden. This positions the system as augmenting rather than replacing human expertise in production management.

### **5.4.3. Environmental Impact**

The environmental benefits of the proposed system are principally realised through improved capacity utilisation and reduced waste. By allocating production quantities more accurately to match demand, the system reduces overproduction - a significant source of material waste in manufacturing, particularly for perishable or time-sensitive products. The dense production reward structure encourages machine agents to fully utilise allocated capacity before ending a shift, minimising idle energy consumption relative to throughput.

Proactive maintenance scheduling also has environmental benefits. Lines that are well-maintained operate closer to their designed efficiency, consuming less energy per unit produced and generating fewer waste by-products from unplanned failures (e.g., material scrapped during uncontrolled line stoppages). By reducing the frequency of corrective maintenance events - which often involve emergency procedures with higher environmental footprints - the predictive maintenance component of the system contributes to more sustainable plant operations.

More broadly, the framework's applicability to parallel-line manufacturing environments makes it relevant to sectors with significant environmental exposure, including chemical processing, food production, and automotive assembly. Improvements in scheduling efficiency in these industries can contribute meaningfully to sector-level reductions in energy intensity and waste generation, aligning with broader sustainability objectives in industrial operations management.

# **CHAPTER 6: CONCLUSIONS**


## **6.1 Results Discussion and Implication**

This thesis proposed and evaluated a hybrid Recurrent Multi-Agent Proximal Policy Optimisation (RMAPPO) framework for parallel-line production scheduling with preventive maintenance. The system decomposes the problem into a trained multi-agent RL layer for routing, sequencing, and maintenance control, followed by a post-hoc LP/MILP lot-sizing layer that translates exported RL allocation masks into capacity-feasible production quantities. The experimental results provide several conclusions with implications for both theory and practice.

The benchmark results show that the proposed pipeline becomes more valuable as problem size increases. On small instances, the average gap is 5.65%, with RH2 still stronger on the easiest cases but RMAPPO outperforming at the upper end of the small tier. On medium instances, RMAPPO improves over RH2 by 47.8% on average and wins on every benchmark. On large instances, RMAPPO improves over RH2 by 40.5% on average while keeping runtime near seconds rather than the 4,000-second RH2 limit.

By delegating final quantity allocation to the post-hoc LP/MILP stage, the manager agent focuses its learning capacity on the combinatorial routing problem. The Step 1 policy learns useful activation masks, while Step 2 converts those masks into feasible lot sizes. This hybrid structure explains why the method scales: RL handles high-level allocation quickly, and the lot-sizing solver only solves the reduced problem conditioned on the learned routing mask.

The gated preventive-maintenance mechanism is also important. By making PM available only when no productive work remains or when accumulated risk exceeds the threshold after production, the policy avoids the older tendency to choose maintenance too early. The sensitivity results show that the PM gate threshold materially changes final cost, with a threshold of 0.5 giving the best tested result at 12,805.4.

The timing results show that RMAPPO generates decisions in seconds, while RH2 often requires hours on medium and large instances. This makes the proposed pipeline suitable as a decision-support approach where solution quality must be improved without running a full rolling-horizon solve from scratch for every planning instance.

## **6.2 Recommendations for Future Research**

Several directions are identified for extending this work, spanning algorithmic improvements, problem generalisations, and deployment considerations.

The current system treats demand as deterministic and known over the lookahead window. In practice, demand forecasts carry uncertainty that grows with horizon length. A natural extension is to replace the deterministic MILP with a stochastic programming variant - for example, sample average approximation or a robust MILP with demand uncertainty sets - allowing the allocator to hedge against forecast error. Alternatively, the RL observation vector could be augmented with demand forecast confidence intervals, training the manager policy to modulate production quantities based on uncertainty levels. Given that the benchmark demand profiles include zero-demand periods such as periods 1 and 6 in lines3.json, policies that learn to exploit known demand structure are already emergent; extending this to uncertain demand is a tractable next step.

The post-hoc LP/MILP lot-sizing module is currently non-differentiable and is applied after policy training. Future work could explore differentiable optimisation layers or imitation targets derived from the MILP solution so that quantity-refinement feedback can influence the manager policy during training.

The current architecture fixes (L,P,T) at training time to ensure observation and action space compatibility with the neural network\'s fixed input dimension. As a result, a new factory configuration with different numbers of lines or products requires a new training run. Future work could address this through graph neural network representations of the factory state, where lines and products are nodes and eligibility relationships are edges. Such an architecture would support zero-shot transfer to unseen factory configurations and enable a single trained policy to serve as a universal scheduler across a family of plants - directly extending the generalisation tested across the structured benchmark family.

The current model considers a single plant with parallel lines. Many industrial scheduling problems involve multiple plants, upstream suppliers with stochastic lead times, and downstream distribution networks. Extending the MARL framework to a multi-echelon setting - where plant-level manager agents coordinate with upstream procurement agents and downstream fulfilment agents - is a natural direction given the CTDE architecture already in place. The agent communication structure required for such coordination could be implemented as an extension of the shared machine policy, with inter-plant routing treated as a higher-level action space analogous to the current manager\'s binary mask.

Incorporation of real-time machine condition data. The current degradation model uses cumulative runtime as a proxy for failure risk via the hazard rate. In practice, manufacturing lines are increasingly instrumented with sensors providing real-time health signals - vibration, temperature, acoustic emission - that are more predictive of imminent failure than runtime alone. Replacing or augmenting the age variable in the machine agent\'s observation with such sensor streams would constitute a true condition-based maintenance system and is the natural next step for industrial deployment of the framework.

Deployment of RL-based scheduling systems in real factories requires operators to understand and trust the system\'s decisions. The current framework produces routing masks and maintenance actions that are not natively interpretable. Future work could incorporate attention visualisation over the observation vector to highlight which demand signals or machine ages drove a particular routing decision, counterfactual explanation modules showing what demand conditions would have changed the decision, or constraint-satisfaction certificates confirming that the MILP allocation is capacity-feasible. These tools would reduce the barrier to adoption in regulated industries where scheduling decisions must be auditable.

Large language model-based planning approaches have recently been explored for combinatorial scheduling problems, using chain-of-thought prompting or fine-tuned models as heuristic solvers. A systematic comparison between the RMAPPO framework and LLM-based scheduling on the structured benchmark family would contribute to the emerging literature on the boundary between classical optimisation, reinforcement learning, and foundation model approaches to industrial planning. Given the strong RMAPPO advantages on medium and large benchmarks, such a comparison would establish a clear quantitative baseline for future work in this direction.

# **References**

\[1\] Lalida Deeratanasrikul and S. Mizuno, "Multiple-stage multiple-machine capacitated lot-sizing and scheduling with sequence-dependent setup: A case study in the wheel industry," Journal of Industrial and Management Optimization, vol. 13, no. 1, pp. 413--428, Mar. 2016, doi: <https://doi.org/10.3934/jimo.2016024>.

\[2\] Jésica de Armas and M. Laguna, "Parallel machine, capacitated lot-sizing and scheduling for the pipe-insulation industry," International Journal of Production Research, vol. 58, no. 3, pp. 800--817, Apr. 2019, doi: <https://doi.org/10.1080/00207543.2019.1600763>.

\[3\] Avilés Martínez N, Renato Maynard Etchepare, M. M. Aguayo, and M. Aníbal Valenzuela, "A mixed-integer programming model for an integrated production planning problem with preventive maintenance in the pulp and paper industry," Engineering Optimization, vol. 55, no. 8, pp. 1352--1369, Jun. 2022, doi: <https://doi.org/10.1080/0305215x.2022.2086237>.

\[4\] M. Alimian, V. Ghezavati, R. Tavakkoli-Moghaddam, and R. Ramezanian, "Solving a parallel-line capacitated lot-sizing and scheduling problem with sequence-dependent setup time/cost and preventive maintenance by a rolling horizon method," Computers & Industrial Engineering, vol. 168, p. 108041, Jun. 2022, doi: <https://doi.org/10.1016/j.cie.2022.108041>.

\[5\] Desiree Maldonado Carvalho and V. Nascimento, "Hybrid matheuristics to solve the integrated lot sizing and scheduling problem on parallel machines with sequence-dependent and non-triangular setup," European Journal of Operational Research, vol. 296, no. 1, pp. 158--173, Jan. 2022, doi: <https://doi.org/10.1016/j.ejor.2021.03.050>.

\[6\] H. Dansou, M. Gruson, F. Lamothe, and J. Legavre, "CIRRELT-2025-11 An Integrated Lot Sizing and Maintenance Planning Problem for a Single Machine," 2025. Accessed: Feb. 27, 2026. \[Online\]. Available: <https://www.cirrelt.ca/documentstravail/cirrelt-2025-11.pdf>

\[7\] M. Babaei, M. Mohammadi, and S. M. T. F. Ghomi, "A genetic algorithm for the simultaneous lot sizing and scheduling problem in capacitated flow shop with complex setups and backlogging," The International Journal of Advanced Manufacturing Technology, vol. 70, no. 1--4, pp. 125--134, Aug. 2013, doi: <https://doi.org/10.1007/s00170-013-5252-y>.

\[8\] L. Yue, Z. Guan, L. Zhang, S. Ullah, and Y. Cui, "Multi objective lotsizing and scheduling with material constraints in flexible parallel lines using a Pareto based guided artificial bee colony algorithm," Computers & Industrial Engineering, vol. 128, pp. 659--680, Feb. 2019, doi: <https://doi.org/10.1016/j.cie.2018.12.065>.

\[9\] H. Feng, L. Xi, L. Xiao, T. Xia, and E. Pan, "Imperfect preventive maintenance optimization for flexible flowshop manufacturing cells considering sequence-dependent group scheduling," Reliability Engineering & System Safety, vol. 176, pp. 218--229, Aug. 2018, doi: <https://doi.org/10.1016/j.ress.2018.04.004>.

\[10\] J. Xiao, H. Yang, C. Zhang, L. Zheng, and J. N. D. Gupta, "A hybrid Lagrangian-simulated annealing-based heuristic for the parallel-machine capacitated lot-sizing and scheduling problem with sequence-dependent setup times," Computers & Operations Research, vol. 63, pp. 72--82, Nov. 2015, doi: <https://doi.org/10.1016/j.cor.2015.04.010>.

\[11\] R. Ramezanian and M. Saidi-Mehrabad, "Hybrid simulated annealing and MIP-based heuristics for stochastic lot-sizing and scheduling problem in capacitated multi-stage production system," Applied Mathematical Modelling, vol. 37, no. 7, pp. 5134--5147, Apr. 2013, doi: <https://doi.org/10.1016/j.apm.2012.10.024>.

\[12\] Y. An, X. Chen, J. Hu, L. Zhang, Y. Li, and J. Jiang, "Joint optimization of preventive maintenance and production rescheduling with new machine insertion and processing speed selection," Reliability Engineering & System Safety, vol. 220, p. 108269, Apr. 2022, doi: <https://doi.org/10.1016/j.ress.2021.108269>.

\[13\] E. Azab, M. Nafea, L. A. Shihata, and M. Mashaly, "A Machine-Learning-Assisted Simulation Approach for Incorporating Predictive Maintenance in Dynamic Flow-Shop Scheduling," Applied Sciences, vol. 11, no. 24, p. 11725, Dec. 2021, doi: <https://doi.org/10.3390/app112411725>.

\[14\] Q. Tang and Y. Wang, "A Model Predictive Control for Lot Sizing and Scheduling Optimization in the Process Industry under Bidirectional Uncertainty of Production Ability and Market Demand," Computational Intelligence and Neuroscience, vol. 2022, pp. 1--23, Sep. 2022, doi: <https://doi.org/10.1155/2022/2676545>.

\[15\] I. Nyamayaro and O. Adetunji, "Application Of Artificial Intelligence And Machine Learning In Lot Sizing," Proceedings of the International Conference on Industrial Engineering and Operations Management, Apr. 2024, doi: <https://doi.org/10.46254/af05.20240148>.

\[16\] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, and X. Chi, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," Advances in Neural Information Processing Systems, vol. 33, pp. 1621--1632, 2020. \[Online\]. Available: <https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html>

\[17\] S. Luo, "Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning," Applied Soft Computing, vol. 91, p. 106208, Jun. 2020, doi: <https://doi.org/10.1016/j.asoc.2020.106208>.

\[18\] L. Wang, X. Hu, Y. Wang, S. Xu, S. Ma, K. Yang, Z. Liu, and W. Wang, "Dynamic job-shop scheduling in smart manufacturing using deep reinforcement learning," Computer Networks, vol. 190, p. 107969, May 2021, doi: <https://doi.org/10.1016/j.comnet.2021.107969>.

\[19\] C.-F. Chien and Y.-B. Lan, "Agent-based approach integrating deep reinforcement learning and hybrid genetic algorithm for dynamic scheduling for Industry 3.5 smart production," Computers & Industrial Engineering, vol. 162, p. 107782, Dec. 2021, doi: <https://doi.org/10.1016/j.cie.2021.107782>.

\[20\] Y. He, G. Wu, Y. Chen, and W. Pedrycz, "A Two-stage Framework and Reinforcement Learning-based Optimization Algorithms for Complex Scheduling Problems," arXiv:2103.05847, 2021. \[Online\]. Available: <https://arxiv.org/abs/2103.05847>

\[21\] C. Yu, C. Zhang, J. Fan, and W. Shen, "An Integrated Mathematical Programming and Reinforcement Learning Algorithm for the Flexible Job Shop Scheduling with Variable Lot-sizing," Journal of Manufacturing Systems, vol. 82, pp. 210--223, Oct. 2025, doi: <https://doi.org/10.1016/j.jmsy.2025.05.002>.
