**VIETNAM NATIONAL UNIVERSITY -- HO CHI MINH CITY**

**INTERNATIONAL UNIVERSITY**

**SCHOOL OF INDUSTRIAL ENGINEERING AND MANAGEMENT**

![](media/image1.emf){width="1.5763888888888888in" height="1.5486111111111112in"}

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

![](media/image1.emf){width="1.5763888888888888in" height="1.5486111111111112in"}

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


> **Updated code alignment note:** This version was revised to match the current repository pipeline: Step 1 RMAPPO training with JIT allocation and binary observations, gated PM action masking, Step 2 RL allocation export plus post-hoc LP/MILP lot sizing, Step 3 deterministic inference with MILP lot sizes, RH2 backfilling, and the active sensitivity grids in `run_sensitivity_experiments.py`.

# **Abstract**

Production scheduling in parallel-line manufacturing becomes especially difficult when sequence-dependent setups and preventive maintenance are included. Traditional exact methods and rolling-horizon heuristics scale poorly, while single-agent reinforcement learning struggles with heterogeneous decision structures. This study introduces a hybrid Recurrent Multi-Agent Proximal Policy Optimisation (RMAPPO) framework for the Parallel-Line Capacitated Lot-Sizing Problem with Sequence-Dependent Setup Times and Preventive Maintenance (PL-CLSP-SDST-PM). The implemented approach trains a multi-agent reinforcement learning layer for routing, sequencing, and maintenance control, then applies a post-hoc LP/MILP lot-sizing module that converts exported RL allocation masks into capacity-feasible production quantities for final evaluation. Using a CTDE training scheme, the method is tested on an industrial case and a benchmark set across multiple problem scales. RMAPPO performs comparably on small instances, improves medium-scale results, and delivers substantial cost reductions on large instances while cutting computation time to near real-time. Sensitivity analyses highlight robust parameter choices and confirm the framework's scalability for complex production environments.

> ***Keywords*:** multi-agent reinforcement learning, production scheduling, lot-sizing, preventive maintenance, proximal policy optimization. 

# **Acknowledgements**

I would like to express my sincere and deepest gratitude to my thesis advisor Assoc. Prof. Dr. Nguyen Van Hop for the generous guidance, patience, and dedication provided throughout the entire development of this work. The insightful feedback, thorough review of each iteration, and consistent encouragement during moments of uncertainty were instrumental in shaping both the direction and quality of this thesis. The genuine investment in my progress and the warmth shown in every interaction have been a profound source of motivation that I will carry well beyond my academic years.

I would also like to extend my appreciation to the School of Industrial Engineering and Management for the opportunity to pursue this programme over the past four years. The education I have received, guided by knowledgeable and passionate lecturers who bring both academic rigour and industry relevance to their teaching, has equipped me with a foundation I am confident will serve me throughout my career in supply chain and operations management.

I am deeply grateful to my family, whose quiet and unwavering support sustained me through the most demanding periods of this journey. To my managers and colleagues at my workplace, I thank you for your understanding of the time commitments this research demanded and for the practical support you extended along the way. To my peers, who engaged in countless hours of discussion, challenged my thinking, and shared in both the difficulties and the small victories of this process - your companionship and encouragement made this journey far more meaningful than it would have been alone.

# **Table of Contents** {#table-of-contents .TOC-Heading}

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

[2.3.4. Predictive approach (Predict-then-Optimize): [17](#predictive-approach-predict-then-optimize)](#predictive-approach-predict-then-optimize)

[2.3.5. Prescriptive approach (Deep Reinforcement Learning): [18](#prescriptive-approach-deep-reinforcement-learning)](#prescriptive-approach-deep-reinforcement-learning)

[Chapter 3: METHODOLOGY [19](#chapter-3-methodology)](#chapter-3-methodology)

[3.1. Approaches Comparison and Selection: [19](#approaches-comparison-and-selection)](#approaches-comparison-and-selection)

[3.1.1. Comparative Analysis: [19](#comparative-analysis)](#comparative-analysis)

[3.1.2. Selection Rationale and Alignment with Objective: [22](#selection-rationale-and-alignment-with-objective)](#selection-rationale-and-alignment-with-objective)

[3.2. Proposed System Design or Proposed Solution Approach: [23](#proposed-system-design-or-proposed-solution-approach)](#proposed-system-design-or-proposed-solution-approach)

[Chapter 4: SOLUTION DEVELOPMENT [29](#chapter-4-solution-development-1)](#chapter-4-solution-development-1)

[4.1. Prototype Solution [29](#prototype-solution)](#prototype-solution)

[4.1.1. Problem Formulation [29](#problem-formulation)](#problem-formulation)

[4.1.2. Mathematical Model [30](#mathematical-model-1)](#mathematical-model-1)

[4.2. Solution Development [34](#solution-development)](#solution-development)

[4.2.1. End-to-End System Description [34](#end-to-end-system-description)](#end-to-end-system-description)

[4.2.2. Post-hoc LP/MILP for Production Quantity Allocation [36](#relaxed-milp-for-production-quantity-allocation)](#relaxed-milp-for-production-quantity-allocation)

[4.2.3. Integration with the RL Control Loop [39](#integration-with-the-rl-control-loop)](#integration-with-the-rl-control-loop)

[4.2.4. RMAPPO Algorithm [40](#rmappo-algorithm)](#rmappo-algorithm)

[4.2.5. State and Observation Spaces [48](#state-and-observation-spaces)](#state-and-observation-spaces)

[4.2.6. Action Pool: [51](#action-pool)](#action-pool)

[4.2.7. Reward Design [54](#reward-design)](#reward-design)

[4.2.8. Parameters and Training Configuration [56](#parameters-and-training-configuration)](#parameters-and-training-configuration)

[CHAPTER 5: RESULT ANALYSIS [60](#chapter-5-result-analysis)](#chapter-5-result-analysis)

[5.1. Experimental Design [60](#experimental-design)](#experimental-design)

[5.1.1. Dataset for real-case evaluation [60](#dataset-for-real-case-evaluation)](#dataset-for-real-case-evaluation)

[5.1.2. Dataset for Benchmark Comparison (Small, Medium, Large) [61](#dataset-for-benchmark-comparison-small-medium-large)](#dataset-for-benchmark-comparison-small-medium-large)

[5.2. Result Illustration and Explanation [64](#result-illustration-and-explanation)](#result-illustration-and-explanation)

[5.2.1. Benchmark Comparison Results (Small, Medium, Large) [64](#benchmark-comparison-results-small-medium-large)](#benchmark-comparison-results-small-medium-large)

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

[Table 4.1: Sets and Indices [30](#table-4.1-sets-and-indices)](#table-4.1-sets-and-indices)

[Table 4.2: System Parameters [30](#table-4.2-system-parameters)](#table-4.2-system-parameters)

[Table 4.3: Cost Parameters [31](#table-4.3-cost-parameters)](#table-4.3-cost-parameters)

[Table 4.4: Decision Variables [31](#table-4.4-decision-variables)](#table-4.4-decision-variables)

[Table 4.5: Algorithm 1 Current RMAPPO + Post-hoc LP/MILP Benchmark Pipeline [44](#table-4.5-algorithm-1-rmappo-with-relaxed-milp-for-production-scheduling)](#table-4.5-algorithm-1-rmappo-with-relaxed-milp-for-production-scheduling)

[Table 4.6: Observation Vector Layout [49](#table-4.6-observation-vector-layout)](#table-4.6-observation-vector-layout)

[Table 4.7: RMAPPO Hyperparameters [56](#table-4.7-rmappo-hyperparameters)](#table-4.7-rmappo-hyperparameters)

[Table 4.8: Environment and Reward Configuration [57](#table-4.8-environment-and-reward-configuration)](#table-4.8-environment-and-reward-configuration)

[Table 5.1: Generated Instance Set [61](#table-5.1-generated-instance-set)](#table-5.1-generated-instance-set)

[Table 5.2: Parameter Settings of the Generated Instances [62](#table-5.2-parameter-settings-of-the-generated-instances)](#table-5.2-parameter-settings-of-the-generated-instances)

[Table 5.3: Small Instance Results [64](#table-5.3-real-case-performance-summary)](#table-5.3-real-case-performance-summary)

[Table 5.4: Medium Instance Results [67](#table-5.4-small-instance-results)](#table-5.4-small-instance-results)

[Table 5.5: Large Instance Results [68](#table-5.5-medium-instance-results)](#table-5.5-medium-instance-results)

[Table 5.6: Summary Across Scales [69](#table-5.6-large-instance-results)](#table-5.6-large-instance-results)

**\**
 {#section}
=====

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

#  {#section-1}

#  {#section-2}

#  {#section-3}

#  {#section-4}

#  {#section-5}

#  {#section-6}

**\**

# **List of Abbreviations**

#  {#section-7}

| CM | Corrective Maintenance |
|----|----|
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
|  |  |

#  {#section-8}

# Chapter 1: INTRODUCTION {#chapter-1-introduction .Style2}

## **1.1. Background: ** {#background}

The automotive industry is under constant pressure to improve vehicle fuel efficiency and reduce emissions. Continuously Variable Transmissions (CVT) push belts are a key technology in achieving these goals due to their ability to maintain engines at optimal operating points. The push belt operates under compressive force and is composed of hundreds of precisely manufactured steel elements assembled together with high-strength steel bands. The loopset is a core sub-component of the push belt. It forms the structural backbone that holds the elements together and ensures stable force transmission. Because loopsets feed directly into push belt assembly, any instability or shortage in loopset output immediately impacts downstream assembly operations.

<figure>
<img src="media/image2.png" style="width:4.3901in;height:3.08in" alt="C:\Users\Bao Tran\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\38641638.tmp" />
<figcaption><p>: CVT Push belt</p></figcaption>
</figure>

Loopset manufacturing represents a complex production environment with parallel production lines (LL5, LL6, LL7, LL9, LL10, LL11) that process multiple product families including Conventional, GU Light (GUL) and GU in discrete production lots. These production lines are heterogeneous. Due to differences in equipment configuration and technical capability, the same product may have materially different cycle times and output rates on different lines. In addition, not all product families can be freely assigned to all lines. Product line eligibility constraints result in certain lines being dedicated to specific product families. Taken together, these characteristics, including parallel unrelated lines, product-line eligibility restrictions, and time-varying demand, define loopset production as a practical instance of the Capacitated Lot Sizing and Scheduling Problem (CLSP) on parallel unrelated machines. 

###### Table 1.1: Product release matrix. {#table-1.1-product-release-matrix. .Table2}

<table style="width:97%;">
<colgroup>
<col style="width: 12%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
</colgroup>
<thead>
<tr>
<th></th>
<th colspan="2" style="text-align: center;">Conventional</th>
<th colspan="3" style="text-align: center;">GUL</th>
<th colspan="3" style="text-align: center;">GU</th>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td style="text-align: center;">067</td>
<td style="text-align: center;">072</td>
<td style="text-align: center;">094</td>
<td style="text-align: center;">100</td>
<td style="text-align: center;">097</td>
<td style="text-align: center;">089</td>
<td style="text-align: center;">083</td>
<td style="text-align: center;">111</td>
</tr>
<tr>
<td>LL5</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td>LL6</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td>LL7</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td>LL9</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td>LL10</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td>LL11</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
<td style="text-align: center;"></td>
<td style="text-align: center;">✓</td>
</tr>
</tbody>
</table>

In addition to line heterogeneity, loopset production is characterized by sequence-dependent setup times and costs. When a line switches from one product family to another, the resulting changeover time and resource cost depend specifically on which product is produced next and which product was produced immediately before it, as well as on the particular line performing the changeover. As a result, the sequence in which lots are scheduled directly influences both the effective production capacity and the total operational cost. Because setup losses can be substantial, the ordering of production lots becomes a critical planning decision that interacts closely with lot sizing and line assignment.

Equipment reliability further complicates planning. Loopset lines require periodic preventive maintenance (PM) to sustain their performance levels and prevent costly unplanned failures. Unlike stochastic breakdowns, which are difficult to anticipate, preventive maintenance activities can be anticipated and incorporated into the production plan. When PM is explicitly considered in the scheduling model, planners can coordinate maintenance windows with production sequencing to minimize their impact on output. At the same time, unplanned corrective maintenance (CM) events impose additional cost and capacity loss that must be accounted for when evaluating production feasibility and total cost.

Despite this complexity, current planning practice relies heavily on manual adjustments and spreadsheet-based tools. Planners typically assign products to lines based on experience and rules of thumb. These methods fail to effectively optimize for the complexities of the systems. Additionally, manual planning struggles to look ahead far enough to minimize sequence-dependent setups, resulting in significant underestimates setup losses caused by poor sequencing, and cannot proactively integrate maintenance planning with production decisions. As a result, the system experiences unnecessary changeover time, avoidable tardiness, and inefficient use of available capacity.

This gap between industrial practice and planning potential motivates the need for present study. By formulating loopset production as a Parallel-Line Capacitated Lot-Sizing and Scheduling Problem with Sequence-Dependent Setups and Preventive Maintenance (PL-CLSP-SDST-PM), this research develops a structured, data-driven planning model that reflects the full complexities of the real production environment. The model integrates demand-driven lot sizing, product-line eligibility, unrelated machine speeds, sequence-dependent setup costs, and maintenance considerations into a unified optimization framework. In doing so, it replaces subjective manual scheduling with a systematic approach grounded in operational reality.

## **1.2. Problem Statement:**  {#problem-statement}

The loopset production system presents a planning environment of considerable practical and scientific complexity. On the demand side, the production system must satisfy daily customer requirements for multiple product families simultaneously. On the supply side, six parallel but unrelated production lines, each with different cycle times and eligibility restrictions, must collectively absorb this demand within a finite planning horizon. The interaction between product mix, line capability, and demand timing creates a combinatorially difficult planning problem that cannot be solved effectively by manual methods.

Each production line operates under a finite capacity limit per planning period. This capacity is not unlimited; the total quantity that can be produced on a given line in a given period is bounded by the line\'s throughput rate and the available production time in that period. In a lot-sizing formulation, this constraint is fundamental because production lots must be sized and sequenced such that the total capacity consumed across all lots assigned to a line in any period does not exceed the line\'s available capacity. Since lines have different throughput rates for the same product, the capacity required by a given lot depends jointly on the product being manufactured and the specific line to which it is assigned.

Sequence-dependent setup times and costs add a further layer of difficulty. Because the changeover penalty depends on both the outgoing and incoming product on a given line, the cost incurred at any point in the schedule is a function of the entire sequence of lots, not just the current assignment. Minimizing total setup cost therefore requires reasoning over sequences, a task that is computationally intensive and highly sensitive to how far ahead planners look. Current spreadsheet-based methods lack the algorithmic capability to optimize sequences over a multi-period horizon, consistently leaving substantial setup savings unrealized.

Preventive maintenance introduces an additional source of capacity constraint that is largely absent from classical lot-sizing models. In practice, PM activities must be performed at regular intervals to maintain line performance and prevent unplanned failures that result in corrective maintenance and production losses. If maintenance windows are not coordinated with production lot assignments, they can cause avoidable disruptions, force costly sequence changes, and erode the capacity buffer available for meeting demand. A planning model that treats maintenance as an external event, rather than integrating it as a decision or constraint, will consistently overestimate available capacity and underestimate the true cost of the production plan.

From a scientific perspective, existing literature on Capacitated Lot-Sizing Problems has focused primarily on single-machine or identical-parallel-machine settings, where sequence-dependent setups are treated as an extension of moderate complexity. The combination of unrelated parallel machines, product-line eligibility, sequence-dependent setup costs, and integrated preventive maintenance in a demand-driven, multi-product environment has received comparatively limited attention. Loopset production therefore represents a valuable and practically grounded test case for extending CLSP theory toward realistic industrial conditions.

## **1.3. Objectives of Study: ** {#objectives-of-study}

The objective of this study is to develop an integrated planning framework for the loopset production environment by formulating the problem as a PL-CLSP-SDST-PM. The model determines, for each line and each planning period, the appropriate production lot sizes, the sequencing of lots on each line, and the timing of preventive maintenance activities, while ensuring that all product demands with their due dates are met at minimal total cost. By doing so, the study generalizes the classical single machine lot sizing and scheduling framework to a more realistic multi line context that includes unrelated machine speeds, product line eligibility constraints, sequence dependent setups, and maintenance integration. The interaction between production and maintenance decisions is central to this problem, since maintenance directly reduces available production capacity, while production sequencing influences how often maintenance is required to avoid failures.

Beyond methodological development, the study aims to generate both industrial and academic value. Production planners and industrial engineers will gain a structured decision-support framework that makes production lot assignments, line scheduling, and changeover sequencing more transparent, consistent, and cost-effective, thereby reducing the reliance on experience-based judgment that is difficult to standardize or transfer. Operations managers will benefit from improved visibility into the trade-offs between inventory holding, backlogging, setup frequency, and maintenance timing, which support more informed and defensible planning decisions. From a scientific perspective, researchers working on lot-sizing, production scheduling, and maintenance integration will find in this study a rigorously formulated extension of the CLSP that brings together several underexplored complexities in a single, industrially validated context. 

## **Scope and Limitation:**

This study focuses specifically on six parallel loop lines (LL5, LL6, LL7, LL9, LL10, LL11) which are the key bottleneck in the loopset manufacturing process. The upstream pipe line stages are treated as external inputs where semi-finished pipe material is assumed to be available at the loop line buffers at or before its specified release time. This boundary separates the lot-sizing and scheduling problem studied here from the operational variability inherent in upstream processes. 

Production capacity on each line in each planning period is treated as a given finite parameter, reflecting the line\'s throughput rate and the total available production time for that period. This capacity parameter consolidates all practical sources of capacity loss, including equipment reliability issues, minor stoppages, and performance variability, into a single bound that limits the total production quantity that can be produced on each line. Corrective maintenance is incorporated through its impact on the reduction of available capacity in the periods in which it occurs and through an explicit stochastic representation of failure events. 

The model operates at the production lot level rather than tracking individual pieces. A production lot corresponds to a batch of product of a single family processed continuously on a single line within a single planning period. This granularity aligns with the existing inventory control and reporting structure at the plant. Material handling and inter-stage transfer times are assumed to be negligible or absorbed within setup and processing durations, consistent with the scope of planning-level decisions rather than detailed operational control.

Several advanced features are intentionally excluded from the present study and identified as directions for future work. These include blocking constraints between production stages and dynamic re-planning triggered by real-time events. These exclusions ensure that the proposed model remains tractable and closely aligned with the planning-level decision-making context for which it is designed, while still capturing the essential complexities of loopset production. This scope and limitation ensure that the proposed model remains solvable and closely aligned with the real industrial practice while still capturing the essential complexity of loopset production.

# Chapter 2: RELATED WORKS {#chapter-2-related-works .Style2}

## Overview: {#overview .Style3}

This study investigates the PL-CLSP-SDST-PM for the loop-set production lines of a major automotive enterprise in Vietnam. The PL-CLSD problem represents one of the most practically relevant and computationally challenging classes of production planning problems. When production is organised across parallel machines or lines, complexity increases due to machine assignment decisions, load balancing, and interactions across lines. This complexity is further amplified when setup times and costs vary according to the production sequence. An additional but often underexplored factor is the role of preventive maintenance. Real manufacturing systems experience equipment deterioration, and PM activities consume capacity that would otherwise be available for production. Neglecting PM can lead to overly optimistic capacity assumptions and infeasible schedules, while excessively conservative maintenance strategies raise operational costs. Integrating PM into lot-sizing and scheduling (LSS) models is therefore crucial for achieving realistic, cost-efficient, and industrially relevant solutions. A review of existing studies offers an understanding of the current research landscape on PL-CLSP-SDST-PM. It identifies the methodologies, findings, and gaps that inform the present research. By examining prior work, this chapter establishes the foundation for developing an optimisation approach tailored to the specific requirements and operational constraints of loop-set manufacturing, particularly in balancing lot-sizing, scheduling, sequencing, and maintenance decisions.

## Literature Review: {#literature-review .Style3}

### Heuristic and Matheuristic Solution Methods: {#heuristic-and-matheuristic-solution-methods .Style5}

Exact MIP approaches rarely scale for CLSSP. Therefore, heuristic and matheuristic methods dominate recent research. Deeratanasrikul & Mizuno (2016) model a multi-stage wheel-manufacturing CLSSP with sequence-dependent setups and test on real data. They use relax-and-fix MIP heuristics per period and iteratively fix variables. These heuristics find feasible solutions efficiently and improve the company's plan. Similarly, de Armas & Laguna (2016) study a parallel‐machine lot-sizing problem for the pipe insulation industry using a two-stage approach. They first solve a big-MIP model for lot sizes, then sequence lots with a constructive heuristic and Monte Carlo for stochastic rates. Avilés et al. (2022) developed a MILP model integrating PM for a multi-line production planning problem. Their formulation accommodates heterogeneous parallel lines with sequence-independent setups and periodic PM windows, offering a practical template for capital-intensive process industries. The model was validated against real mill data, demonstrating cost savings relative to manual scheduling. Alimian et al. (2022) formulate a parallel-line CLSP with sequence-dependent setups, due dates, and preventive maintenance as a mixed-integer model. To handle its complexity, they develop two MILP-based rolling-horizon (RH) heuristics (RH1 and RH2) that solve truncated time windows sequentially. Computational results show these RH methods significantly outperform a straight CPLEX solve in solution quality. In particular, RH2 provides reasonable production plans, better-adjusted schedules with respect to due dates, and a proper maintenance plan compared to alternatives. This demonstrates how mathematics can exploit problem structure to scale PL-CLSD models.

Matheuristics, hybrid algorithms that embed MILP solvers within heuristic frameworks, have grown in prominence because they combine the modelling precision of MILP with the scalability of heuristics. The relax-and-fix (R&F) and fix-and-optimize (F&O) paradigm is particularly widespread. Carvalho & Nascimento (2022) consider the integrated lot sizing and scheduling problem (ILSSP) on non-identical parallel machines with non-triangular sequence setups. They propose a R&F decomposition combined with path-relinking and kernel search heuristics. Their hybrid methods vastly outperform CPLEX solvers on most benchmark instances. Dansou et al. (2025) address a single-machine capacitated lot-sizing problem (CLSP) with preventive maintenance. They model machine deterioration using capacity reduction functions or age-based decay and solve by a two-stage R&F plus F&O heuristic. In this formulation, PM decisions reduce failure-induced downtime by preserving capacity. The proposed heuristics effectively balance lot-sizing and maintenance scheduling.

### Metaheuristic algorithms: {#metaheuristic-algorithms .Style5}

The practical necessity of near-optimal solutions for large instances has driven extensive development of metaheuristic algorithms for PL-CLSD problems. Babaei et al. (2014) applied GA to a multi-level CLSD with setup carry-over and backlogging in a capacitated flow shop. Yue et al. (2019) addressed a multi-objective CLSP with sequence-dependent setup times for flexible parallel lines using a Pareto-based Guided Artificial Bee Colony (PGABC) algorithm, demonstrating that population-based metaheuristics can effectively navigate the Pareto frontier of cost versus makespan on parallel-line problems. 

For integrated production-PM models, Feng et al. (2018) proposed an imperfect PM optimisation framework for flexible flowshop manufacturing cells (FFMCs) with sequence-dependent group scheduling. To solve the model, a simulated annealing embedded genetic algorithm (SAGA) is developed. Experiments show that jointly optimising lot sequences and PM plans simultaneously results in considerable cost reductions of 12-18% over sequential approaches.

Hybrid metaheuristics are increasingly preferred over standalone algorithms. Xiao et al. (2015) hybridised Lagrangian relaxation with Simulated Annealing (SA) for parallel-machine CLSD with sequence-dependent setups. Ramezanian and Saidi-Mehrabad (2013) combined hybrid SA, Firefly algorithms, and MIP-based heuristics for stochastic CLSD with uncertain demand and processing times. An et al. (2022) addressed joint optimisation of PM and flexible job-shop rescheduling with new machine insertion and processing speed selection via a metaheuristic approach, reporting effective navigation of the combined scheduling-maintenance decision space in dynamic manufacturing environments. Ruiz-Rodríguez et al. (2024) compared GA-simheuristics (GA combined with Monte Carlo simulation), reinforcement learning, and classical dispatching rules for dynamic maintenance scheduling under uncertainty. The GA-simheuristic was most robust on high-uncertainty instances, highlighting that stochastic maintenance durations significantly decrease the performance of deterministic algorithms.

### Machine Learning Integration: {#machine-learning-integration .Style5}

Although AI/ML applications in lot-sizing and scheduling are still in their early stages, some recent studies show their potential, especially when problems include dynamic parameters such as machine degradation, unpredictable demand, or the need for predictive maintenance. Azab et al. (2021) develop a data-driven, ML-integrated scheduling framework for a pharmaceutical flow-shop where preventive maintenance windows must be predicted in advance. By training ML models on IoT sensor data, the system predicts the remaining useful life (RUL) of equipment and proposes proactive maintenance periods. These predictions feed into a discrete-event simulator that then schedules jobs and maintenance jointly. The results show substantial reductions in makespan and unplanned downtime compared to reactive maintenance approaches, demonstrating the value of integrating ML predictions into lot-sizing and scheduling decisions. Tang et al. 2022 uses an Elman neural network to predict both demand and machine failure rates, feeding these forecasts into a lot-sizing and scheduling model solved by Particle Swarm Optimization (PSO). Such techniques show that incorporating demand learning can improve schedule quality under uncertainty. However, a recent survey by Adetunji (2024) noted that AI/ML use in lot-sizing is "still in its infancy," with limited adoption compared to routing, forecasting, or predictive maintenance. 

Existing AI-driven CLSSP research generally falls into two categories: predictive approaches, which apply machine learning to forecast maintenance requirements or demand, and prescriptive approaches, which use learning-based methods to derive heuristic scheduling rules. Application of AI/ML to integrate learning in scheduling and decision-making processes remains largely unexplored and represents an opportunity for future work to explore training agents to make lot-sizing decisions on-the-fly. 

### Similarity and Differences Between the Key reference and the Thesis: {#similarity-and-differences-between-the-key-reference-and-the-thesis .Style5}

Among the studies reviewed, Alimian et al. (2022) stands as the most directly relevant work to the formulation adopted in this thesis. It presents a deterministic Mixed-Integer Nonlinear Programming (MINLP) model for a parallel-line Capacitated Lot-Sizing Problem with sequence-dependent setup time/cost, due dates, and integrated Preventive Maintenance (PM) planning. The model is solved using a MIP-based RH heuristic, and its findings are validated against a set of generated benchmark instances. This study shares the same core problem class and builds upon the same foundational structure, while introducing several extensions motivated by the industrial context of loopset production. 

On the side of similarity, the present study inherits from Alimian et al. (2022) the integrated MILP formulation in which lot-sizing, sequencing, and PM planning are solved simultaneously, the sequence-dependent setup cost structure, the parallel unrelated machine architecture, the binary PM decision variable, and the comprehensive six-component cost objective. These elements are adopted without modification because they accurately represent the operational structure of loopset production and have been validated as effective modeling choices in prior work.

The primary structural extension is the addition of product-line eligibility constraints. In the loopset environment, not all product families can be freely assigned to all lines as certain lines are technically restricted to specific product families, as described in Chapter I. This restriction is absent from Alimian et al. (2022), where any product can be assigned to any line.

The more fundamental difference lies in the treatment of failure risk and maintenance planning. Alimian et al. (2022) use fixed statistical parameters, specifically a constant hazard rate and an assumed exponential lifetime distribution, to compute the expected number of failures analytically. These parameters remain unchanged across planning periods and the model does not adjust in response to recent operating conditions, such as heavy utilization, early signs of wear, or unusually stable performance. As a result, the RH procedure functions in a reactive manner. It optimizes based on the conditions observed at the beginning of each cycle, but it cannot anticipate deteriorating equipment conditions before it materializes as capacity loss or unplanned downtime. 

A second limitation concerns the separation of decision-making functions. In the rolling-horizon framework, lot sizing, line assignment, sequencing, and preventive maintenance scheduling are optimised jointly within each window, but the policy that determines which action is selected in any given state is reconstructed from the solver at every cycle. As a result, the system does not retain any accumulated knowledge regarding which action sequences are effective under specific combinations of machine age, inventory positions, or demand conditions. Each planning cycle begins independently of previous ones, and any experiential insight that could have been learned from historical planning outcomes is effectively lost.

The research gap addressed in the present study can therefore be stated as follows: the absence of a forward-looking, experience-based mechanism that can learn effective production planning and maintenance scheduling policies directly from interactions with the production environment, without relying on fixed statistical parameters or requiring a mathematical solver to be invoked at every decision point. Addressing this gap requires a prescriptive framework that does not simply forecast uncertain inputs and feed them into an optimisation model. Instead, it must learn a complete decision policy that maps observable system states to cost-minimising actions across the full joint decision space of lot sizing, sequencing, and preventive maintenance scheduling.

### Design Concepts Consideration: {#design-concepts-consideration .Style5}

The present study adopts the following design concepts, each justified by its alignment with the loopset production environment and the prescriptive modelling approach chosen.

First, the problem is framed as a sequential decision process rather than a static mathematical programme. Loopset production unfolds over discrete planning periods, and decisions made in one period directly affect the feasibility and cost of decisions in subsequent periods through inventory levels, machine ages, and outstanding demand. This inter-temporal dependency structure is naturally captured by a Markov Decision Process (MDP) formulation, in which a state vector encodes all information relevant to current and future decisions, and an agent selects actions to maximise cumulative reward over the full planning horizon. The MDP framework is therefore adopted as the theoretical foundation for the prescriptive planning model.

Second, lot sizing, production sequencing, and preventive maintenance scheduling are treated as a joint action space rather than as separate sub-problems. In the loopset environment, these decisions are tightly coupled: the lot size assigned to a product family on a given line determines the capacity consumed and the inventory position achieved, the production sequence determines the changeover cost incurred, and the preventive maintenance decision determines both the capacity available for production and the failure risk carried into the next period. Approaches that sêparte these decisions into sequential layers, as is common in hierarchical planning models, tend to produce suboptimal results because decisions made in an upstream layer restrict the feasible choices available to downstream layers, and the latter cannot reverse or compensate for these imposed constraints. By formulating all three decisions as components of a single composite action vector, the prescriptive agent is able to learn and exploit the trade-offs among them through direct interaction with the production environment.

Third, the state representation is designed to provide the agent with sufficient information to infer both immediate costs and future risks. The state vector includes the accumulated machine runtime since the last preventive maintenance action on each line, which serves as a proxy for the current failure risk, the inventory level of each product family, the outstanding demand backlog, and the number of remaining periods in the planning horizon. This information set mirrors the inputs that a human planner would consult when making scheduling decisions, ensuring that the agent has access to operationally grounded signals rather than abstract statistical parameters.

Fourth, the reward function is defined as the negative of the total period cost, encompassing all six cost components present in Alimian et al. (2022): inventory holding, backlogging, production, sequence-dependent setup, preventive maintenance, and corrective maintenance. This formulation ensures that the agent\'s learning objective is directly aligned with the cost-minimisation goal of the study and that all cost trade-offs relevant to the loopset planning problem are reflected in the training signal. In particular, the inclusion of corrective maintenance cost in the reward creates an implicit incentive for the agent to schedule preventive maintenance proactively, since actions that defer maintenance and lead to equipment failure are penalised through the resulting corrective maintenance cost in a subsequent period.

Fifth, product-line eligibility constraints are enforced through the action masking mechanism of the prescriptive agent. In the loopset environment, not all product families can be freely assigned to all lines, as described in Chapter I. Rather than incorporating these restrictions as hard constraints in a mathematical programme, the prescriptive approach enforces them by zeroing out the probability mass assigned to ineligible product-line combinations in the policy network\'s output layer. This guarantees that the agent never selects an infeasible action, preserving constraint satisfaction without requiring an external solver. 

## Key references/ Candidate Solution Methods: {#key-references-candidate-solution-methods .Style3}

### Matheuristic Approaches: {#matheuristic-approaches .Style5}

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

Machine learning methods offer a fundamentally different capability: rather than optimizing a fixed mathematical model, they learn patterns from historical data and use those patterns to make predictions or guide decisions. In the context of production planning and maintenance scheduling, ML methods are relevant in two broad roles: as predictive components that forecast uncertain inputs to the optimization model, and as prescriptive components that learn solution policies or dispatching rules directly from data.

### **Predictive approach (Predict-then-Optimize):**

In the predictive role, a trained model such as a recurrent neural network, a feedforward neural network, a random forest, or a gradient boosting model takes as input a set of features describing the current system state and outputs forecasts for future periods. Relevant forecasts for the present problem include predicted demand per product family per period, and predicted failure risk or remaining useful life per production line per period. These forecasts can then be passed to an optimization solver as updated input parameters, replacing the fixed statistical estimates that a purely deterministic model would use. This Predict-then-Optimize architecture keeps the prediction and optimization modules separate, allowing each to be developed, validated, and replaced independently.

### Prescriptive approach (Deep Reinforcement Learning):

In the prescriptive role, methods such as deep reinforcement learning (DRL) learn a policy that maps system states directly to actions. For example, deciding at each period whether to schedule a PM block or continue production on a given line. DRL agents are trained through repeated interaction with a simulated environment, updating their policy to maximize cumulative reward over time. While DRL can in principle handle complex, non-stationary environments without requiring an explicit mathematical model, it typically requires large amounts of training data and long training times, and the resulting policies are difficult to interpret or audit. This limits its suitability for industrial settings where planners need to understand and trust the decisions generated by the planning tool.

# **Chapter 3: METHODOLOGY**

## **3.1. Approaches Comparison and Selection:** {#approaches-comparison-and-selection}

### **3.1.1. Comparative Analysis:** {#comparative-analysis}

#### **3.1.1.1. Matheuristics (R&F, F&O, Rolling Horizon)** {#matheuristics-rf-fo-rolling-horizon}

**Strengths:**

Matheuristics are fully compatible with the MILP formulation and inherit the solver\'s constraint handling without requiring a custom solution encoding. They also provide computable optimality gaps for each subproblem, which allows solution quality to be evaluated rigorously, and they scale effectively to industrial instance sizes through structured problem decomposition. The rolling-horizon variant, in particular, preserves inter-period dependencies through a freezing mechanism and enables the solver to re-optimise future periods at each step as conditions evolve.

**Weaknesses:**

All matheuristic approaches share the same fundamental limitation as metaheuristics with respect to learning. They optimize a fixed model at each invocation without accumulating knowledge across planning cycles. In the rolling-horizon heuristic, even as the planning window advances, the solver treats each new cycle as an independent parametric optimization problem rather than as an opportunity to refine its understanding of the underlying system. As a result, the policy implicit in the solver\'s decisions does not improve over time, and any experience that could reduce long-run cost through better maintenance timing or more efficient sequencing is never captured.

Furthermore, matheuristics require a solver call at every planning period, incurring computational cost that scales with instance size and potentially limiting their applicability in real-time or high-frequency planning contexts.

#### **3.1.1.2. Metaheuristics (GA, SA, TS, PSO, ABC):** {#metaheuristics-ga-sa-ts-pso-abc}

**Strengths:**

Metaheuristic algorithms offer strong scalability to large problem instances and do not require a linearized or convex formulation, allowing them to work directly on the original nonlinear problem structure. Their flexibility makes it relatively straightforward to incorporate new constraints through adjustments to the solution encoding, and multi-objective variants can approximate Pareto frontiers without requiring objectives to be aggregated into a single cost function.

**Weaknesses:**

Metaheuristics provide no optimality guarantees and cannot generate computable bounds on solution quality unless an external benchmark is available. Applying them effectively to the present problem requires highly customized solution encodings that simultaneously represent binary PM decisions, sequence dependent setup transitions, and product line eligibility constraints, which is a technically demanding design task. More fundamentally, metaheuristics operate on a fixed mathematical model with static parameters. They do not learn from experience across planning cycles and cannot adapt their search strategy in response to changing equipment conditions or demand patterns. Each invocation of the algorithm treats the problem as if it were being encountered for the first time, discarding any information that could be extracted from the outcomes of past planning decisions.

#### **3.1.1.3. Machine learning Integration - Predictive (Predict-then-Optimize)** {#machine-learning-integration---predictive-predict-then-optimize}

**Strengths:**

The predictive approach addresses the static parameterisation limitation of matheuristics by training machine learning models to forecast uncertain inputs, most notably demand fluctuations and equipment failure risk, and feeding these forecasts into the optimisation solver as updated parameters. The Predict-then-Optimise architecture maintains a clear separation between the prediction and optimisation modules, allowing each component to be developed, validated, and replaced independently. When historical data is sufficiently rich, this approach can substantially improve the quality of the solver\'s inputs and thereby produce better plans than a purely deterministic baseline.

**Weaknesses:**

Despite its advantages over purely reactive baselines, the predictive approach has important structural limitations. The prediction and optimisation modules are trained independently, which means the prediction model is not aware of how its forecasts will be used by the solver, and the solver does not account for the uncertainty inherent in the forecasts. This misalignment can cause the solver to over-fit to point forecasts that are systematically biased in certain operating regimes. Additionally, the Predict-then-Optimise architecture still requires a full solver invocation at every planning cycle, retaining the computational cost of the matheuristic methods.

More importantly, the approach does not learn a decision policy. It focuses on predicting inputs, while the mapping from predicted inputs to actions is determined by the solver at run time and does not improve with accumulated experience. As a result, the predictive module addresses the parameterisation limitation described above, but it does not resolve the deeper policy-learning limitation, namely the inability to accumulate and exploit knowledge about which action sequences yield favourable outcomes under varying system conditions.

#### **3.1.1.4. Machine learning Integration - Prescriptive (Deep Reinforcement Learning)** {#machine-learning-integration---prescriptive-deep-reinforcement-learning}

**Strengths:**

Deep Reinforcement Learning (DRL) addresses both the parameterisation gap and the policy-learning gap simultaneously. Instead of optimising a fixed mathematical model at each planning cycle, a DRL agent learns a policy that is encoded in a neural network and maps observable system states directly to planning actions. The policy is trained through repeated interaction with a simulated production environment, accumulating experience across many planning episodes and improving over time as the agent discovers which action sequences minimize cumulative cost. Once training is completed, the policy can be deployed without invoking an optimisation solver, enabling real-time decision making at negligible computational cost per period. The end-to-end learning paradigm allows the agent to internalise complex and non-linear relationships among machine age, inventory level, demand pressure, and setup cost that would be difficult to capture in a parametric model. Furthermore, the policy can be periodically fine-tuned on new operational data, enabling it to adapt to changes in equipment ageing trajectories, product mix, or demand patterns without requiring a full retraining cycle. 

**Weaknesses:**

DRL approaches require a well-designed simulation environment for training, and the quality of the learned policy depends on the fidelity of this environment to the real production system. Training is computationally intensive and requires careful hyper-parameter tuning to achieve stable convergence. The resulting policy is also less transparent than a mathematical programming solution, as the mapping from state to action is encoded in the weights of a neural network rather than in an explicit objective function with interpretable constraints. This reduced interpretability represents a trade-off that must be managed through careful policy analysis and validation against the baseline solver.

### **3.1.2. Selection Rationale and Alignment with Objective:** {#selection-rationale-and-alignment-with-objective}

Examining the three approaches across the dimensions most relevant to the present problem, including constraint constraint satisfaction, scalability, adaptability to dynamic conditions and alignment with the research gap, yields a clear ordering of suitability. 

Metaheuristics offer scalability and formulation flexibility, but their inability to bound solution quality and their complete absence of learning across planning cycles make them poorly suited as a primary method for a problem where the key research contribution is proactive, experience-based planning. They are best reserved for future work where the problem is scaled beyond the current instance size or where the simulation environment needed to train a DRL agent cannot be constructed.

Matheuristics address the most critical scalability limitation of exact solvers while maintaining the ability to handle complex operational constraints. However, these methods are unable to accumulate decision-making experience across planning cycles, and their static parameterisation prevents them from anticipating deteriorating equipment conditions or shifting demand patterns before such changes manifest as capacity losses or service failures.

The predictive Predict-then-Optimise approach partially addresses the parameterisation gap by updating the inputs to the solver with machine-learned forecasts. However, it retains the solver invocation overhead of the matheuristic baseline and does not learn a decision policy. It therefore represents an incremental improvement over the reactive baseline rather than a structural solution to the policy-learning gap.

The prescriptive DRL approach is selected as the primary method because it is the only candidate that directly addresses both dimensions of the research gap identified above. A DRL agent learns to make proactive planning decisions that account for future consequences, such as scheduling preventive maintenance before failure risk becomes critical, building inventory buffers in anticipation of upcoming demand peaks, and sequencing production lots in a manner that minimises cumulative changeover cost over the planning horizon. These capabilities are structural properties of the DRL framework that emerge from policy learning rather than from external predictions or manually specified look-ahead rules. Furthermore, the deployment of a trained policy eliminates the need for a solver invocation at each planning period, reducing the computational cost of live planning to a single forward pass through the policy network.



## **3.2. Proposed System Design or Proposed Solution Approach:** {#proposed-system-design-or-proposed-solution-approach}

Following the slide structure, the proposed system is organised as a three-step hybrid pipeline. The pipeline separates the discrete scheduling problem from the continuous lot-sizing problem: RMAPPO learns the high-level assignment and sequencing policy, while the post-hoc lot-sizing model computes production quantities from the learned routing masks. This structure matches the current code and avoids the old monolithic design in which the RL policy would have to learn both routing and exact quantities directly.

**Inputs.** Each benchmark instance provides the number of products, lines, and periods; demand profile; capacity; eligibility matrix; processing times; setup times and costs; production costs; preventive-maintenance parameters; corrective-maintenance cost and hazard rates; and training parameters. These are loaded from JSON benchmark files and passed into `BoschEnv`.

**Step 1 - Multi-agent RL training (first-level optimisation).** The first level solves the routing and sequencing problem. The Process Agent $A_0$ learns product-line activation decisions as a binary $L \times P$ mask. Machine agents $A_1,\ldots,A_L$ learn local production, preventive-maintenance, and End-Shift actions. In the current code, Step 1 uses RMAPPO with `allocator_mode=jit`, `reward_mode=step1`, `obs_mode=binary`, `lookahead_days=4`, and `max_actions_per_period=8`. The JIT allocator is used only to create temporary queues during training; it is not the final lot-sizing method.

**Step 2 - Post-hoc LP/MILP lot sizing (second-level optimisation).** After training, `rl_export_allocation.py` runs the trained actors in deterministic inference mode and exports the RL allocation masks. `milp_posthoc.py` then takes the frozen binary routing masks as fixed input and solves a post-hoc lot-sizing model. Because routing is fixed, the difficult binary assignment part is removed from the quantity model; the remaining problem is a tractable continuous lot-sizing problem with inventory, backlog, production, expected CM, eligibility, and capacity constraints. The output is a feasible production quantity plan by line, product, and period.

**Step 3 - Resolve the first-level schedule with corrected quantities.** The trained RMAPPO policies are run again with the post-hoc lot sizes loaded into the environment. The decentralised actors execute sequence-dependent setups, production, gated PM, and End-Shift actions without further weight updates. This final simulation evaluates production, setup, maintenance, backlog, and inventory costs and produces the final per-instance total cost.

This design is a two-level optimisation architecture: the multi-agent RL system solves assignment and sequencing, while post-hoc lot sizing finds the best quantity trade-off under the RL routing mask. It is computationally tractable because the expensive rolling-horizon baseline is replaced by fast learned inference plus a reduced lot-sizing solve.

**Centralised training with decentralised execution.** During training, a centralised critic uses the full shared state to compute advantages. During execution, only the decentralised actors are used: each agent acts from its local observation and recurrent hidden state. The manager has a dedicated policy, while all machine agents share one policy differentiated by a one-hot line-identity feature.

**Gated preventive maintenance.** The current benchmark and sensitivity runs use the gated PM mechanism. Under `pm_action_mode=gated`, PM is available only when capacity permits and either no productive work is available, or the line has produced during the period and its risk score $\lambda_l \cdot Age_l$ reaches `pm_gate_risk_threshold`. This keeps the slide's maintenance-aware planning concept while matching the current implementation.

The proposed system is illustrated in Figure 3.1.

<figure>
<img src="media/image3.png" style="width:5.90753in;height:3.69231in" alt="C:\Users\Bao Tran\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\7FCFEB04.tmp" />
<figcaption><p><span id="_Toc229103289" class="anchor"></span>Figure 3.1: Proposed solution approach - RMAPPO system with post-hoc LP/MILP lot-sizing integration</p></figcaption>
</figure>

**\**

# **Chapter 4: SOLUTION DEVELOPMENT** {#chapter-4-solution-development-1}

## **4.1. Prototype Solution** {#prototype-solution}



### **4.1.1. Problem Formulation** {#problem-formulation}

The implemented problem is a two-level optimisation problem for parallel-line capacitated lot sizing and scheduling with sequence-dependent setups and preventive maintenance. The first level is a multi-agent RL scheduling problem: decide which products should be assigned to which lines and how each line should sequence production, PM, and shift termination over micro-steps. The second level is a post-hoc lot-sizing problem: once the learned binary routing masks are fixed, compute feasible quantities that minimise operating cost.

This formulation follows the slide structure. The multi-agent RL system solves the assignment and sequencing problem. The post-hoc LP/MILP lot-sizing stage finds the cost trade-off among inventory, backlog, production, and expected corrective maintenance under the fixed routing mask. The final Step 3 simulation then resolves the first-level schedule using the corrected lot sizes.

The current code therefore implements a three-step benchmark pipeline rather than a single end-to-end mathematical programme. `step1_only` measures the trained policy with the Step 1 JIT training allocator. `step1_2_3` measures the full pipeline after allocation export and post-hoc lot sizing. Both are compared with RH2 using the same benchmark instance files.

### **4.1.2. Mathematical Model** {#mathematical-model-1}

###### Table 4.1: Sets and Indices {#table-4.1-sets-and-indices .Table2}

| Symbol | Definition |
|:---|:---|
| $$\mathcal{L = \{}1,\ldots,L\}$$ | Set of parallel production lines |
| $$\mathcal{P = \{}1,\ldots,P\}$$ | Set of distinct products |
| $$\mathcal{T = \{}0,\ldots,T - 1\}$$ | Set of discrete planning periods |
| $$l\mathcal{\in L}$$ | Index for a production line |
| $$p\mathcal{\in P}$$ | Index for a product |
| $$t\mathcal{\in T}$$ | Index for a planning period |
| $$k$$ | Index for a machine micro-step within a period |

###### Table 4.2: System Parameters {#table-4.2-system-parameters .Table2}

| Symbol | Definition |
|:---|----|
| $$D_{p,t}$$ | Customer demand for product $p$ in period $t$ (units) |
| $$C_{l}$$ | Maximum daily operational capacity of line $l$ (hours) |
| $$pt_{l,p}$$ | Processing time per unit of product $p$ on line $l$ (hours/unit) |
| $$st_{l,i,j}$$ | Sequence-dependent setup time on line $l$ when switching from product $i$ to $j$ (hours) |
| $$E_{l,p} \in \{ 0,1\}$$ | Eligibility indicator: 1 if line $l$ can produce product $p$, else 0 |
| $$k_{\max}$$ | Maximum number of machine micro-steps per planning period |
| $$k_{w}$$ | Demand lookahead horizon (number of future periods) |
| $$w$$ | Post-hoc lot-sizing rolling window (number of lookahead periods for quantity allocation) |

###### Table 4.3: Cost Parameters {#table-4.3-cost-parameters .Table2}

| Symbol | Definition |
|:---|:---|
| $$h$$ | Holding cost per unit of inventory per period |
| $$b$$ | Backlog cost per unit of unmet demand per period |
| $$\beta_{p}$$ | Flat backlog penalty per period when product $p$ has any outstanding backlog |
| $$sc_{l,i,j}$$ | Setup cost incurred on line $l$ when switching from product $i$ to $j$ |
| $$\text{ProdCost}_{l,p}$$ | Variable production cost per unit of product $p$ on line $l$ |
| $$pm_{l}$$ | Cost of performing preventive maintenance on line $l$ |
| $$cm_{l}$$ | Cost per expected corrective maintenance event on line $l$ |
| $$\lambda_{l}$$ | Hazard rate of line $l$ (expected failures per unit of active runtime) |
| $$M^{RL}_{l,p,t}$$ | Fixed binary routing mask exported from the trained RL policy |

###### Table 4.4: Decision Variables {#table-4.4-decision-variables .Table2}

| Symbol | Type | Definition |
|:---|:---|:---|
| $$a_{0,t} \in \{ 0,1\}^{L \times P}$$ | Binary matrix | Manager activation mask: $a_{0,t}\lbrack l,p\rbrack = 1$ if line $l$ is activated for product $p$ in period $t$ |
| $$Q_{l,p,t} \geq 0$$ | Continuous | Production quantity allocated to line $l$'s queue for product $p$ in period $t$ by the post-hoc lot-sizing stage |
| $$x_{l,p,k} \geq 0$$ | Continuous | Post-hoc lot-sizing quantity of product $p$ allocated to line $l$ at lookahead step $k$ |
| $$M^{RL}_{l,p,k} \in \{0,1\}$$ | Fixed input | Exported RL activation mask; production is allowed only when this value is 1 |
| $$\text{inv}_{k,p} \geq 0$$ | Continuous | Post-hoc lot-sizing inventory state for product $p$ at lookahead step $k$ |
| $$\text{back}_{k,p} \geq 0$$ | Continuous | Post-hoc lot-sizing backlog state for product $p$ at lookahead step $k$ |
| $$I_{p,t} \geq 0$$ | Continuous | Actual inventory of product $p$ at end of period $t$ |
| $$B_{p,t} \geq 0$$ | Continuous | Actual backlog of product $p$ at end of period $t$ |
| $$\text{Age}_{l,t} \geq 0$$ | Continuous | Cumulative degradation runtime of line $l$ at period $t$ |

**Objective Function**

The overall optimisation objective is to minimise total expected cost over the planning horizon $\mathcal{T}$, encompassing inventory holding, unmet demand (backlog), production, setup, and maintenance costs:

$$\begin{array}{r}
\min\sum_{t\mathcal{\in T}}^{}\left\lbrack \begin{array}{r}
\sum_{p\mathcal{\in P}}^{}\left( h \cdot I_{p,t} + b \cdot B_{p,t} + \beta_{p} \cdot \mathbf{1}\left\lbrack B_{p,t} > 0 \right\rbrack \right) + \\
\sum_{l\mathcal{\in L}}^{}{\sum_{p\mathcal{\in P}}^{}\text{ProdCost}_{l,p}} \cdot x_{l,p,t} + \sum_{l\mathcal{\in L}}^{}\left( C_{l,t}^{\text{setup}} + C_{l,t}^{\text{PM}} + C_{l,t}^{\text{CM}} \right)
\end{array} \right\rbrack\#(1)
\end{array}$$

where the per-period cost components are defined as:

$$\begin{array}{r}
C_{l,t}^{\text{setup}} = \sum_{i \neq j}^{}sc_{l,i,j} \cdot \mathbf{1}\left\lbrack \text{changeover }i \rightarrow j\text{ occurs on line }l\text{ in period }t \right\rbrack\#(2)
\end{array}$$

$$\begin{array}{r}
C_{l,t}^{\text{PM}} = pm_{l} \cdot \mathbf{1}\left\lbrack \text{PM performed on line }l\text{ in period }t \right\rbrack\#(3)
\end{array}$$

$$\begin{array}{r}
C_{l,t}^{\text{CM}} = \lambda_{l} \cdot cm_{l} \cdot \sum_{p\mathcal{\in P}}^{}x_{l,p,t} \cdot pt_{l,p}\ \#(4)
\end{array}$$

The CM cost formulation uses a deterministic expectation based on the hazard rate and active processing time, yielding a differentiable and predictable penalty within the RL reward signal.

**Constraints:**

**Inventory balance:** At each period end, inventory and backlog are updated based on actual production and demand:

$$\begin{array}{r}
I_{p,t} = \max\left( 0,\mspace{6mu} I_{p,t - 1} + \sum_{l\mathcal{\in L}}^{}x_{l,p,t} - D_{p,t} - B_{p,t - 1} \right)\#(5)
\end{array}$$

$$\begin{array}{r}
B_{p,t} = \max\left( 0,\mspace{6mu} D_{p,t} + B_{p,t - 1} - I_{p,t - 1} - \sum_{l\mathcal{\in L}}^{}x_{l,p,t} \right)\ \#(6)
\end{array}$$

**Capacity constraint:** Total time consumed on each line must not exceed available daily capacity:

$$\begin{array}{r}
\sum_{p\mathcal{\in P}}^{}x_{l,p,t} \cdot pt_{l,p} + \sum_{\text{setups}}^{}st_{l, \cdot , \cdot} + \text{MaintenanceTime}_{l,t} \leq C_{l}\quad\forall l\mathcal{\in L,}t\mathcal{\in T}\#(7)
\end{array}$$

**Eligibility constraint:** Production must respect hard line-product compatibility:

$$\begin{array}{r}
x_{l,p,t} = 0\quad\text{if }E_{l,p} = 0,\quad\forall l\mathcal{\in L,}p\mathcal{\in P,}t\mathcal{\in T}\#(8)
\end{array}$$

**Machine age dynamics:**

$$\begin{array}{r}
\text{Age}_{l,t} = \left\{ \begin{matrix}
0 & \text{if PM is performed on line }l\text{ in period }t \\
\text{Age}_{l,t - 1} + \sum_{p\mathcal{\in P}}^{}x_{l,p,t} \cdot pt_{l,p} & \text{otherwise}
\end{matrix} \right.\ \#(9)
\end{array}\ $$

## **4.2. Solution Development** {#solution-development}



### **4.2.1. End-to-End System Description** {#end-to-end-system-description}

The production system contains $1+L$ agents: one Process Agent and one Machine Agent per production line. The current implementation follows the three-step flow shown in the slides.

**Step 1 - first-level optimisation: multi-agent RL training.** `run_benchmark_pipeline.py` trains RMAPPO policies for the selected benchmark size. The Process Agent learns product-line allocation as a binary mask over $(l,p)$ pairs. Machine agents learn product execution, gated PM, and End-Shift actions. The training environment uses `allocator_mode=jit`, so activating a pair $[l,p]=1$ causes the JIT allocator to load the remaining-demand quota for that product into the line queue during training. This is a training-time queueing mechanism, not the final lot-sizing solver.

**Step 2 - second-level optimisation: post-hoc lot sizing.** After training, `rl_export_allocation.py` runs deterministic inference and exports the RL allocation masks. `milp_posthoc.py` receives those masks and solves the quantity model. Since the binary routing mask is fixed, the post-hoc model only decides continuous production quantities, inventory, and backlog. It respects eligibility and capacity limits, subtracts conservative setup overhead from capacity, and minimises holding, backlog, production, and expected corrective-maintenance costs.

**Step 3 - resolve first-level optimisation.** The final inference simulation loads the post-hoc lot sizes back into `BoschEnv`. The pre-trained actors run with no weight updates. The centralised critic is not used during this evaluation; only the decentralised actor networks choose actions from local observations. Machine agents drain the physical queues created from the post-hoc lot sizes and execute sequence-dependent setups and PM. The resulting inventory, backlog, setup, maintenance, and production costs form the final total cost.

### **4.2.2. Post-hoc LP/MILP for Production Quantity Allocation** {#relaxed-milp-for-production-quantity-allocation}

The post-hoc lot-sizing stage is the second-level optimiser. Its input is the fixed binary routing mask exported from Step 1. In the current code, the allocation JSON may be period-level or period-by-micro-step. For the period-by-step format, the solver forms the union of active products in a period and restricts production to active, eligible line-product pairs.

Once the binary mask is fixed, the quantity problem contains continuous production, inventory, and backlog variables. Backlog is allowed with penalty so the model remains feasible even when the RL mask is restrictive. For each line, production time plus setup overhead must fit within available capacity. The setup overhead is conservative: the code uses the active-product set to reserve capacity for possible changeovers instead of assuming that the LP itself chooses the sequence.

The objective minimises inventory holding, backlog, direct production, and expected corrective-maintenance costs. Expected CM cost is deterministic and proportional to hazard rate, CM cost, processing time, and produced quantity. This matches the slide claim that the second level finds the quantity cost trade-off while leaving sequencing to the trained RL policy.

### **4.2.3. Integration with the RL Control Loop** {#integration-with-the-rl-control-loop}

The integration between RL and lot sizing occurs after training. Step 1 learns policies and exports masks. Step 2 converts those masks into quantities. Step 3 executes the trained policies with the corrected quantities loaded as queues.

During final inference, `_build_available_actions()` checks the loaded queues before enabling product actions. A machine can process a product only when the product is eligible, the queue is positive, and enough capacity remains after required setup. PM and End-Shift are also masked from the current line state. Invalid actions receive zero probability through the availability mask before action selection.

### **4.2.4. RMAPPO Algorithm** {#rmappo-algorithm}

#### **4.2.4.1. POMDP Formulation** {#pomdp-formulation}

The multi-agent production scheduling problem is formalised as a cooperative Partially Observable Markov Decision Process (POMDP), defined by the tuple $\mathcal{\langle S,\{}\mathcal{O}_{i}\},\{\mathcal{A}_{i}\mathcal{\},T,\{}R_{i}\},\gamma\rangle$, where:

- $\mathcal{S}$ is the global state space, consisting of inventory levels, backlogs, machine ages, production queues, setup states, and remaining periods.
- $\mathcal{O}_{i}$ is the local observation space for agent $i$; each agent receives a partial view of the global state.
- $\mathcal{A}_{i}$ is the action space of agent $i$.
- $\mathcal{T:S \times}\prod_{i}^{}\mathcal{A}_{i}\mathcal{\rightarrow S}$ is the environment transition function.
- $R_{i}\mathcal{:S \times}\prod_{i}^{}\mathcal{A}_{i}\mathbb{\rightarrow R}$ is the reward function for agent $i$.
- $\gamma \in (0,1)$ is the discount factor.

At each time step $t$, agent $i$ observes $o_{i,t} \in \mathcal{O}_{i}$ (a partial view of $s_{t}$) and selects an action $a_{i,t} \sim \pi_{\theta_{i}}( \cdot \mid o_{i,t},h_{i,t})$, where $h_{i,t}$ denotes the GRU hidden state encoding the agent's memory of past observations within the episode.

The partial observability arises structurally: machine agents cannot observe queue states on other lines, and agents do not observe the internal post-hoc lot-sizing solve during Step 1 training. This makes recurrent policies necessary, as agents must infer latent state from the history of their own observations.

#### **4.2.4.2. Centralised Training with Decentralised Execution** {#centralised-training-with-decentralised-execution}

RMAPPO adopts the Centralised Training with Decentralised Execution (CTDE) paradigm. During training, a centralised critic $V_{\phi}(s_{t})$ has access to the full global state $s_{t}$ and produces value estimates used to compute advantages. During execution, each agent's actor $\pi_{\theta_{i}}$ uses only its local observation $o_{i,t}$ and hidden state $h_{i,t}$, making the policy implementable in a fully decentralised manner.

The global state fed to the centralised critic is the concatenation of all agents' observations:

$$\begin{array}{r}
s_{t} = \left\lbrack o_{0,t}\mspace{6mu} \parallel \mspace{6mu} o_{1,t}\mspace{6mu} \parallel \mspace{6mu}\cdots\mspace{6mu} \parallel \mspace{6mu} o_{L,t} \right\rbrack \in \mathbb{R}^{d \cdot N}\#(16)
\end{array}$$

where $d$ is the per-agent observation dimension and $N = 1 + L$ is the total number of agents.

#### **4.2.4.3 Joint Policy and Recurrent Actor** {#joint-policy-and-recurrent-actor}

Under the CTDE assumption of conditional independence given local observations, the joint policy factorises as:

$$\begin{array}{r}
\pi_{\theta}\left( \mathbf{a}_{t}\mid\mathbf{o}_{t} \right) = \prod_{i = 0}^{L}\pi_{\theta_{i}}\left( a_{i,t}\mid o_{i,t},h_{i,t} \right)\#(17)
\end{array}$$

where $\mathbf{a}_{t} = (a_{0,t},a_{1,t},\ldots,a_{L,t})$ is the joint action vector and $\mathbf{o}_{t} = (o_{0,t},\ldots,o_{L,t})$ is the joint observation. In this work, the manager maintains a dedicated policy $\pi_{\theta_{0}}$, while all machine agents share a single policy $\pi_{\theta_{m}}$ differentiated by a line-identity one-hot embedding in the observation. The joint policy thus involves two distinct parameter sets $\theta = \{\theta_{0},\theta_{m}\}$.

Each recurrent actor is implemented as an MLP feature extractor followed by a single-layer Gated Recurrent Unit (GRU):

$$\begin{array}{r}
\mathbf{f}_{i,t} = \text{MLP}_{\theta_{i}}\left( o_{i,t} \right),\quad\quad h_{i,t} = \text{GRU}_{\theta_{i}}\left( \mathbf{f}_{i,t},\mspace{6mu} h_{i,t - 1} \right)\#(18)
\end{array}$$

$$\begin{array}{r}
a_{i,t} \sim \pi_{\theta_{i}}\left( \cdot \mid h_{i,t} \right)\#(19)
\end{array}$$

The GRU hidden state $h_{i,t} \in \mathbb{R}^{256}$ is reset to zero at episode boundaries, allowing the agent to maintain a within-episode belief state across the sequential micro-steps of each production period.

#### **4.2.4.4. Advantage Estimation with GAE** {#advantage-estimation-with-gae}

Advantages are estimated using Generalised Advantage Estimation (GAE), which interpolates between high-variance Monte Carlo returns and high-bias one-step temporal difference targets:

$$\begin{array}{r}
{\widehat{A}}_{t} = \sum_{n = 0}^{\infty}(\gamma\lambda_{\text{GAE}})^{n}\delta_{t + n}\#(20)
\end{array}$$

where the one-step TD error is:

$$\begin{array}{r}
\delta_{t} = r_{t} + \gamma V_{\phi}\left( s_{t + 1} \right) - V_{\phi}\left( s_{t} \right)\#(21)
\end{array}$$

with $\gamma$ the discount factor and $\lambda_{\text{GAE}}$ the GAE smoothing parameter controlling the bias-variance trade-off. When $\lambda_{\text{GAE}} = 1$, the estimator recovers the full Monte Carlo return; when $\lambda_{\text{GAE}} = 0$, it reduces to the one-step TD error. In practice, $\lambda_{\text{GAE}} = 0.95$ is used, providing a low-variance estimate at modest bias cost.

Since agents have different activity patterns (the manager acts once per period while machine agents act up to $k_{\max}$ times), advantages are normalised separately per agent using only the active time steps:

$$\begin{array}{r}
{\widetilde{A}}_{i,t} = \frac{{\widehat{A}}_{i,t} - \mu_{i}^{\text{active}}}{\sigma_{i}^{\text{active}} + \varepsilon}\#(22)
\end{array}$$

where $\mu_{i}^{\text{active}}$ and $\sigma_{i}^{\text{active}}$ are the mean and standard deviation of advantages computed exclusively over time steps where agent $i$ was active (i.e., `active_mask = 1`), and $\varepsilon = 10^{- 5}$ is a numerical stability constant.

#### **4.2.4.5. Clipped Surrogate Objective** {#clipped-surrogate-objective}

The actor is updated by maximising the PPO clipped surrogate objective, which constrains the policy update to a trust region without requiring second-order information:

$$\begin{array}{r}
\mathcal{L}^{\text{CLIP}}\left( \theta_{i} \right) = \mathbb{E}_{t}\left\lbrack \min\left( \rho_{i,t}\,{\widetilde{A}}_{i,t},\mspace{6mu}\mspace{6mu}\text{clip}\left( \rho_{i,t},\, 1 - \varepsilon_{\text{clip}},\, 1 + \varepsilon_{\text{clip}} \right){\widetilde{A}}_{i,t} \right) \cdot m_{i,t} \right\rbrack\#(23)
\end{array}$$

where the importance sampling ratio is:

$$\begin{array}{r}
\rho_{i,t} = \frac{\pi_{\theta_{i}}\left( a_{i,t}\mid o_{i,t},h_{i,t} \right)}{\pi_{\theta_{i}^{\text{old}}}\left( a_{i,t}\mid o_{i,t},h_{i,t} \right)}\#(24)
\end{array}$$

and $m_{i,t} \in \{ 0,1\}$ is the active mask for agent $i$ at time step $t$, and $\varepsilon_{\text{clip}} = 0.2$ is the clipping threshold. The expectation is computed over time steps and environments in the mini-batch. The minimum operation and clipping together prevent both overly optimistic updates when advantages are positive and overly pessimistic updates when advantages are negative, bounding the effective KL divergence between successive policy iterates.

#### **4.2.4.6. Centralised Critic and Value Loss** {#centralised-critic-and-value-loss}

The centralised critic $V_{\phi}(s_{t})$ is implemented as a separate recurrent network receiving the full global state $s_{t}$. To handle the large and non-stationary scale of the cost-based reward signal, the value targets are normalised using a running-mean-variance normaliser (ValueNorm) with momentum $\beta_{\text{vn}} = 0.99999$:

$$\begin{array}{r}
{\widehat{R}}_{t}^{\text{norm}} = \frac{{\widehat{R}}_{t} - \widehat{\mu}}{\sqrt{{\widehat{\sigma}}^{2} + \varepsilon}}\#(25)
\end{array}$$

where ${\widehat{R}}_{t} = r_{t} + \gamma{\widehat{R}}_{t + 1}$ is the bootstrapped return and $\widehat{\mu}$, ${\widehat{\sigma}}^{2}$ are the running statistics of observed returns.

The value loss uses a clipped Huber formulation to simultaneously prevent value collapse and reduce sensitivity to large TD errors:

$$\begin{array}{r}
\mathcal{L}^{\text{V}}(\phi) = \mathbb{E}_{t}\left\lbrack \max\left( \mathcal{L}_{\text{Huber}}\left( V_{\phi}\left( s_{t} \right) - {\widehat{R}}_{t}^{\text{norm}} \right),\mspace{6mu}\mspace{6mu}\mathcal{L}_{\text{Huber}}\left( V_{\phi}^{\text{clip}}\left( s_{t} \right) - {\widehat{R}}_{t}^{\text{norm}} \right) \right) \right\rbrack\#(26)
\end{array}$$

where $V_{\phi}^{\text{clip}}(s_{t}) = V_{\phi^{\text{old}}}(s_{t}) + \text{clip}(V_{\phi}(s_{t}) - V_{\phi^{\text{old}}}(s_{t}),\, - \varepsilon_{\text{clip}},\,\varepsilon_{\text{clip}})$ is the clipped value prediction, and the Huber loss is:

$$\begin{array}{r}
\mathcal{L}_{\text{Huber}}(e) = \left\{ \begin{matrix}
\frac{1}{2}e^{2} & |e| \leq \delta \\
\delta\left( |e| - \frac{\delta}{2} \right) & |e| > \delta
\end{matrix} \right.\ \ \#(27)
\end{array}$$

with $\delta = 10.0$ as the transition threshold.

#### **4.2.4.7. Entropy Bonus and Combined Objective** {#entropy-bonus-and-combined-objective}

An entropy regularisation term is added to the actor objective to encourage exploration and prevent premature policy collapse:

$$\begin{array}{r}
\mathcal{H}\left\lbrack \pi_{\theta_{i}} \right\rbrack = - \mathbb{E}_{a \sim \pi_{\theta_{i}}}\left\lbrack \log\pi_{\theta_{i}}\left( a\mid o_{i,t},h_{i,t} \right) \right\rbrack\#(28)
\end{array}$$

The combined optimisation objective for agent $i$ is:

$$\begin{array}{r}
\mathcal{L}\left( \theta_{i},\phi \right) = - \mathcal{L}^{\text{CLIP}}\left( \theta_{i} \right) - \eta\mathcal{\cdot H}\left\lbrack \pi_{\theta_{i}} \right\rbrack + c_{v} \cdot \mathcal{L}^{\text{V}}(\phi)\#(29)
\end{array}$$

where $\eta$ is the entropy coefficient and $c_{v}$ is the value loss coefficient. Both actor and critic parameters are updated by minimising this combined objective via Adam optimisation with gradient clipping to a maximum norm of $\parallel \nabla \parallel_{\max}$.


#### **4.2.4.8. RMAPPO Pseudo-code** {#rmappo-pseudo-code}

The following table describes the current training and evaluation workflow implemented by the benchmark scripts.

###### Table 4.5: Algorithm 1 Current RMAPPO + Post-hoc LP/MILP Benchmark Pipeline {#table-4.5-algorithm-1-rmappo-with-relaxed-milp-for-production-scheduling .Table2}

|  |
|----|
| Algorithm 1 Current RMAPPO + Post-hoc LP/MILP Benchmark Pipeline |
| 1: Select benchmark dimensions $(P,L,T)$ and create the result directory `benchmark_results/{P}_{L}_{T}` or tagged equivalent |
| 2: Train RMAPPO with `reward_mode=step1`, `allocator_mode=jit`, `obs_mode=binary`, `lookahead_days=4`, and `max_actions_per_period=8` |
| 3: During each rollout, sample the manager's binary $L \times P$ activation mask and use the JIT allocator to create temporary queues for demanded products |
| 4: Let each machine agent choose product, gated PM, or End-Shift actions under the current availability mask |
| 5: At period close, apply the proportional unmet-demand kill-switch penalty and team efficiency reward; compute GAE and update actors/critics with PPO clipping |
| 6: Save `actor_agent0.pt`, `actor_agent1.pt`, and `config.yaml` to the benchmark model directory |
| 7: For each comparison instance, run deterministic inference and export period/micro-step manager masks with `rl_export_allocation.py` |
| 8: Solve post-hoc lot sizing with `milp_posthoc.py`, using the exported active masks as feasibility restrictions |
| 9: Re-run deterministic inference with the post-hoc lot-size file loaded into `BoschEnv`; store this cost as `step1_2_3` |
| 10: Also run direct deterministic inference without post-hoc lot sizes; store this cost as `step1_only` |
| 11: Run `rh2_baseline.py` unless RH2 comparison is skipped; store this cost as `rh2` |
| 12: Write per-instance costs and timing fields to `comparison_100instances.json` |
| 13: For sensitivity experiments, repeat the same pipeline with tagged hyperparameter overrides and summarise paired differences against the baseline |

#### **4.2.4.9. Deep Reinforcement Learning Architecture** {#deep-reinforcement-learning-architecture}

The provided figure below illustrates a Centralized Training with Decentralized Execution (CTDE) architecture, tailored for reinforcement learning in partially observable environments using a PPO actor-critic framework. During the decentralized execution phase, the agent interacts with the environment by mapping local, partial observations to actions via its actor network, relying purely on its limited view without access to the global state. The trajectories generated from these interactions, comprising states, observations, actions, and rewards, are stored in a shared experience buffer to bridge the gap to the offline training phase. During centralized training, a privileged critic network evaluates the collected data by processing the full global state to compute precise state-value estimates and GAE. These advantage signals subsequently guide the optimization of the actor\'s policy weights using a clipped surrogate objective function, while the critic\'s parameters are simultaneously updated by minimizing the mean squared error value loss; finally, the newly optimized weights are deployed back to the decentralized actor for the next execution cycle.

<figure>
<img src="media/image5.jpg" style="width:5.58427in;height:4.22123in" />
<figcaption><p><span id="_Toc229106955" class="anchor"></span>Figure 4.2: Deep Reinforcement Learning Archtecture</p></figcaption>
</figure>

### **4.2.5. State and Observation Spaces** {#state-and-observation-spaces}

All agents receive an observation vector of identical dimension $d$, enabling the shared machine policy architecture without requiring separate network definitions per agent. Segments outside an agent's operational scope are zero-filled.

The observation dimension is:

$$d = \underset{\text{inv, back}}{\underbrace{2P}} + \underset{\text{queue}}{\underbrace{LP}} + \underset{\text{coverage, q\_total, shortfall}}{\underbrace{3P}} + \underset{\text{demand window}}{\underbrace{k_{w}P}} + \underset{\text{rem. periods}}{\underbrace{1}} + \underset{\text{line avail.}}{\underbrace{L}} + \underset{\text{line setup}}{\underbrace{LP}} + \underset{\text{ages, line\_id}}{\underbrace{2L}} + \underset{\text{contention}}{\underbrace{L}} + \underset{\text{urgency}}{\underbrace{P}} + \underset{\text{eligibility}}{\underbrace{LP}} + \underset{\text{scarcity, lines\_needed}}{\underbrace{2P}}$$

###### Table 4.6: Observation Vector Layout {#table-4.6-observation-vector-layout .Table2}

<table>
<colgroup>
<col style="width: 24%" />
<col style="width: 9%" />
<col style="width: 29%" />
<col style="width: 18%" />
<col style="width: 2%" />
<col style="width: 15%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;">Segment</th>
<th style="text-align: left;">Dim</th>
<th style="text-align: left;">Description</th>
<th style="text-align: left;"><span class="math display"><em>A</em><sub>0</sub></span></th>
<th colspan="2" style="text-align: center;"><span class="math display"><em>A</em><sub><em>l</em></sub></span></th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="math display">inv</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;"><span class="math display">ln (1 + <em>I</em><sub><em>p</em>, <em>t</em></sub>)</span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">back</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;"><span class="math display">ln (1 + <em>B</em><sub><em>p</em>, <em>t</em></sub>)</span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">queue</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em><em>P</em></span></td>
<td style="text-align: center;">Log-scaled production queue</td>
<td colspan="2" style="text-align: center;">Full matrix</td>
<td style="text-align: left;">Own row only</td>
</tr>
<tr>
<td><span class="math display">coverage</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;">Fraction of lines holding any queue per product</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">queue_total</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;"><span class="math display">$$\ln(1 + \sum_{l}^{}Q_{l,p,t})$$</span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">shortfall</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;">Log-scaled net unmet demand signal</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">demand_window</span></td>
<td style="text-align: left;"><span class="math display"><em>k</em><sub><em>w</em></sub><em>P</em></span></td>
<td style="text-align: center;"><span class="math inline">ln (1 + <em>D</em><sub><em>t</em> + <em>d</em>, <em>p</em></sub>)</span> for <span class="math inline"><em>d</em> ∈ {0, …, <em>k</em><sub><em>w</em></sub> − 1}</span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">remaining_periods</span></td>
<td style="text-align: left;"><span class="math display">1</span></td>
<td style="text-align: center;"><span class="math display"><em>T</em> − <em>t</em></span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">line_avail</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em></span></td>
<td style="text-align: center;">Line availability flags</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">line_setup</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em><em>P</em></span></td>
<td style="text-align: center;">One-hot last-product state per line</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">ages</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em></span></td>
<td style="text-align: center;">Machine degradation ages <span class="math inline">Age<sub><em>l</em>, <em>t</em></sub></span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">line_id</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em></span></td>
<td style="text-align: center;">One-hot identity: <span class="math inline"><em>e</em><sub><em>l</em></sub></span> for agent <span class="math inline"><em>A</em><sub><em>l</em></sub></span></td>
<td colspan="2">Zero</td>
<td style="text-align: left;">Yes</td>
</tr>
<tr>
<td><span class="math display">contention</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em></span></td>
<td style="text-align: center;">Per-line product contention score</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">urgency</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;">Per-product demand urgency</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">eligibility</span></td>
<td style="text-align: left;"><span class="math display"><em>L</em><em>P</em></span></td>
<td style="text-align: center;">Static eligibility matrix <span class="math inline"><em>E</em><sub><em>l</em>, <em>p</em></sub></span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">scarcity</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;"><span class="math display">$$1/\sum_{l}^{}E_{l,p}$$</span></td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
<tr>
<td><span class="math display">lines_needed</span></td>
<td style="text-align: left;"><span class="math display"><em>P</em></span></td>
<td style="text-align: center;">Periods of capacity needed to clear backlog</td>
<td colspan="2">Yes</td>
<td style="text-align: left;">Zero</td>
</tr>
</tbody>
</table>

**Observation scaling.** The environment supports log-scaling for continuous observations, but the active benchmark pipeline uses `obs_mode=binary`. Under binary mode, inventory, backlog, queue, demand-window, and shortfall features are represented as present/absent signals; the formulas below describe the continuous/log-scaled counterpart used when binary mode is disabled:

$\begin{array}{r}
\text{inv}_{p} = \ln\left( 1 + I_{p,t} \right),\quad\quad\text{back}_{p} = \ln\left( 1 + B_{p,t} \right)\ \#(30)
\end{array}$

**Shortfall** provides the Process Agent with an urgency signal that accounts for existing queue allocations. In binary mode this becomes a positive/zero indicator; in continuous mode it is:

$$\begin{array}{r}
\text{shortfall}_{p} = \ln\left( 1 + \max\left( 0,\mspace{6mu}\sum_{d = 0}^{k_{w} - 1}D_{t + d,\, p} + B_{p,t} - I_{p,t} - \sum_{l}^{}Q_{l,p,t} \right) \right)\#(31)
\end{array}$$

**Product scarcity** encodes structural supply risk arising from limited line eligibility:

$$\begin{array}{r}
\text{scarcity}_{p} = \frac{1}{\sum_{l\mathcal{\in L}}^{}E_{l,p}}\#(32)
\end{array}$$

**Lines needed** quantifies how many periods of full-factory capacity would be required to eliminate the current backlog:

$$\begin{array}{r}
\text{lines\_needed}_{p} = \min\left( \frac{B_{p,t}}{\text{CapUnits}_{p}},\mspace{6mu} T \right),\quad\quad\text{CapUnits}_{p} = \sum_{\substack{l:\, E_{l,p} = 1 \\ pt_{l,p} > 0}}^{}\frac{C_{l}}{pt_{l,p}}\#(33)
\end{array}$$

**Product urgency** encodes the proximity of the next non-zero demand event within the lookahead window:

$$\begin{array}{r}
\text{urgency}_{p} = \frac{1}{d_{p}^{*} + 1},\quad d_{p}^{*} = \min\left\{ d \in \left\{ 0,\ldots,k_{w} - 1 \right\}:D_{t + d,\, p} > 0 \right\}\#(34)
\end{array}$$

**Line contention** measures how many of a line's eligible products currently have active demand, normalised by the total number of products:

$$\begin{array}{r}
\text{contention}_{l} = \frac{1}{P}\sum_{p\mathcal{\in P}}^{}\mathbf{1}\left\lbrack E_{l,p} = 1\mspace{6mu}\text{and}\mspace{6mu} D_{t,p} > 0 \right\rbrack\#(35)
\end{array}$$

The **one-hot line identity** $\text{line\_id} = e_{l} \in \mathbb{R}^{L}$ for machine agent $A_{l}$ is the key feature enabling the shared machine policy to learn heterogeneous per-line behaviour while sharing a single set of network weights. Because setup times, hazard rates, and production costs differ per line, the policy must condition its decisions on which line it is controlling; the identity embedding provides this information explicitly.

### **4.2.6. Action Pool:** {#action-pool}

#### **4.2.6.1. Process Agent Action Space (**$\mathbf{A}_{\mathbf{0}}$**)** {#process-agent-action-space-mathbfa_mathbf0}

The Process Agent's action space is a `MultiDiscrete` space of binary dimensions, one per $(l,p)$ pair:

$$\mathcal{A}_{0} = \{ 0,1\}^{L \times P}$$

where $a_{0,t}\lbrack l,p\rbrack = 1$ signals that line $l$ should be activated for product $p$ in period $t$. The action is encoded as a concatenation of $L \times P$ two-element one-hot vectors, giving a total action representation of length $2LP$.

**Action pool.** The Process Agent selects one binary decision for every $(l,p)$ pair simultaneously, forming an activation mask of shape $L \times P$. There are two possible values per entry:

Activate (1). Line $l$ is assigned to produce product $p$ in period $t$. During Step 1, the JIT allocator queues demand for active slots; during Step 2, the post-hoc LP/MILP allocates quantity $Q_{l,p,t} \geq 0$ to active slots subject to capacity, inventory balance, and eligibility constraints. Setting this to 1 on an ineligible pair ($E_{l,p} = 0$) is structurally prevented by the action mask.

Deactivate (0). Line $l$ is not assigned to product $p$ in period $t$. During Step 1, no JIT queue is created from that activation slot. During Step 2, the post-hoc lot-sizing model fixes the corresponding quantity to zero unless the pair appears in the exported active mask. This is the default state for all ineligible pairs and for eligible pairs the manager chooses not to activate.

Eligibility constraints are enforced via a hard availability mask applied before sampling: any $(l,p)$ pair with $E_{l,p} = 0$ is constrained to the deactivate state by setting its logit to $- 10^{10}$, guaranteeing zero sampling probability without requiring explicit penalty rewards. The design motivation is the separation of concerns between routing and quantity decisions: the Process Agent determines where production effort should be directed, while the JIT allocator supports fast training and the post-hoc LP/MILP determines final production quantities given that routing. This reduces the effective action space from $O(H^{LP})$ - if the manager specified quantities directly - to $O(2^{LP})$, enabling the RL policy to focus on the combinatorial routing problem.


#### **4.2.6.2. Machine Agent Action Space (**$\mathbf{A}_{\mathbf{1}}\mathbf{,\ldots,}\mathbf{A}_{\mathbf{L}}$**)** {#machine-agent-action-space-mathbfa_mathbf1mathbfldotsmathbfa_mathbfl}

Each machine agent controls one production line and uses a discrete action space of $P+2$ choices:

$$\mathcal{A}_{l} = \{0,1,\ldots,P-1\} \cup \{\text{PM}\} \cup \{\text{End-Shift}\}$$

**Product execution actions.** A product action is valid only when the product is eligible on the line, a positive queue or demand-based feasible quantity exists, and the remaining capacity can cover the required setup plus at least one unit of production. When the action is executed, the simulator applies the sequence-dependent setup time and cost when the selected product differs from the previous line setup, produces as many units as capacity allows, reduces the queue, updates produced quantities, and increases line age by the active runtime.

**Preventive Maintenance (PM).** PM consumes `pm_time`, adds `pm_cost`, and resets the line age to zero. In the current benchmark configuration, PM is usually controlled by `pm_action_mode=gated`. In this mode, PM is available only if enough capacity remains and either no productive work is available or the line has already produced during the period and $\lambda_l Age_l \geq \tau$, where $\tau$ is `pm_gate_risk_threshold`. The `normal` mode leaves PM available whenever capacity permits, while `no_pm` removes the PM action.

**End-Shift.** End-Shift is available when no product action can be executed. If the line is already marked done, End-Shift remains the only valid action. This prevents machine agents from ending a period while producible work is still available.

**Action masking protocol.** The environment recomputes the action mask at every micro-step. Invalid product, PM, and End-Shift choices are masked before the policy samples or selects an action, so infeasible actions do not need to be learned through penalty rewards.


### **4.2.7. Reward Design** {#reward-design}

The current benchmark training uses `reward_mode=step1`. This mode is different from the older separated manager-machine reward description. It applies a shared team reward to all agents and adds a proportional unmet-demand kill-switch penalty.

At period close, the environment computes inventory, backlog, production, setup, PM, and expected CM costs. In `step1` mode, if backlog remains, every agent receives a penalty proportional to total unmet demand:

$$R^{\text{kill}}_t = -\kappa \sum_p B_{p,t}$$

where $\kappa$ is `kill_switch_penalty`. For $P \geq 10$, the current pipeline uses $\kappa=20$ per unmet unit. For smaller instances, it scales the P=5 base value proportionally with product count.

The team reward also penalises total active processing time and worker-side operational costs:

$$R^{\text{team}}_t = -\left(\text{ProcTime}_t + C^{setup}_t + C^{PM}_t + C^{CM}_t\right)$$

Optional activation and load-balance penalties may be applied if their coefficients are non-zero, but the benchmark pipeline leaves both at zero. In non-`step1` modes, the environment can still use the older manager reward based on inventory, backlog, production, and weighted worker costs; however, that is not the active configuration used by `run_benchmark_pipeline.py`.

Dense machine-step shaping remains active during production execution. Product actions receive reward proportional to productive capacity consumed through `dense_production_reward`, setup actions subtract `dense_setup_penalty * setup_cost`, and PM actions subtract `dense_pm_penalty * pm_cost`. Expected CM cost is accumulated from hazard rate and active runtime. The Step 1 reward design intentionally makes unmet demand expensive while still teaching the agents to avoid inefficient setup and maintenance behaviour.


### **4.2.8. Parameters and Training Configuration** {#parameters-and-training-configuration}

The current benchmark pipeline uses the tuned P=5 hyperparameter set defined in `run_benchmark_pipeline.py`, with large-instance overrides for entropy and the backlog kill-switch. The default training budget is determined by product count: 500,000 environment steps for small instances ($P \leq 13$), 1,000,000 for medium instances ($14 \leq P \leq 22$), and 3,000,000 for large instances ($P \geq 23$). The budget can be overridden with `--num_env_steps`.

###### Table 4.7: RMAPPO Hyperparameters {#table-4.7-rmappo-hyperparameters .Table2}

| Parameter | Current value in code | Description |
|----|:---|:---|
| `algorithm_name` | `rmappo` | Recurrent multi-agent PPO implementation |
| `hidden_size` | 128 | Hidden dimension for actor/critic networks |
| `recurrent_N` | 1 | Number of recurrent layers |
| `lr` | 0.00026918605816868973 | Actor learning rate |
| `critic_lr` | 0.00009359933007344967 | Critic learning rate |
| `entropy_coef` | 0.0017196365 for P<10; 0.01 for P>=10 | Exploration bonus coefficient |
| `clip_param` | 0.3494944414 | PPO probability-ratio clipping threshold |
| `ppo_epoch` | 6 | PPO update epochs per rollout |
| `gae_lambda` | 0.9624779422 | GAE smoothing parameter |
| `gamma` | 0.9626502159 | Discount factor |
| `num_mini_batch` | 1 | Mini-batches per PPO update |
| `n_rollout_threads` | 8 | Parallel rollout environments |
| `n_training_threads` | 1 | Training worker threads |

###### Table 4.8: Environment and Reward Configuration {#table-4.8-environment-and-reward-configuration .Table2}

| Parameter | Current value in benchmark pipeline | Description |
|:---|:---|:---|
| `lookahead_days` | 4 | Demand forecast window in observation |
| `max_actions_per_period` | 8 | Maximum machine micro-steps per period |
| `reward_mode` | `step1` | Team reward plus proportional unmet-demand penalty |
| `allocator_mode` | `jit` during Step 1 training | Fast training allocator before post-hoc lot sizing |
| `obs_mode` | `binary` | Quantity-blind observation encoding |
| `dense_production_reward` | 2.1194776779870743 | Productive-capacity reward weight |
| `dense_setup_penalty` | 0.0 in benchmark training command | Setup penalty coefficient passed during training |
| `kill_switch_penalty` | scaled P=5 value for P<10; 20 for P>=10 | Proportional unmet-demand penalty |
| `pm_action_mode` | `gated` in the main benchmark pipeline default | Controls PM action availability |
| `pm_gate_risk_threshold` | 1.0 by default | Risk threshold for gated PM |
| `result_tag` | optional | Adds suffix to model/result directories |

The saved model directory contains `actor_agent0.pt` for the manager and `actor_agent1.pt` for the shared machine policy. When available, the pipeline also copies `config.yaml` so inference utilities can recover architecture and environment settings. Training uses W&B logging with the experiment name `benchmark_{P}_{L}_{T}` plus any result tag.

The current code should therefore be described as a benchmark automation system with reusable flags for training, comparison, RH2 skipping/backfilling, tagged result directories, and sensitivity sweeps, rather than as one fixed-budget experiment.

# **CHAPTER 5: RESULT ANALYSIS**

## **5.1. Experimental Design** {#experimental-design}

To develop a generalizable scheduling policy, the system is trained across a diverse distribution of problem instances rather than a single fixed scenario. During training, a new random instance is generated at the start of each episode by sampling all cost, demand, and production parameters from predefined distributions. The fixed dimensions $(L,P,T)$ are set once at initialisation and held constant throughout training to ensure observation and action space compatibility with the neural network. This approach - training across randomised instance parameters while keeping structural dimensions fixed - exposes the policy to a wide variety of supply chain conditions without requiring retraining for each new scenario.

### **5.1.1. Dataset for real-case evaluation** {#dataset-for-real-case-evaluation}

To evaluate the system under realistic industrial conditions, a real-case dataset is constructed from a factory configuration. This configuration specifies a fixed set of 6 production lines and 7 products over a 30-period planning horizon, with domain-calibrated parameters for processing times, setup times, costs, capacity, and maintenance. Unlike the randomly generated training instances, the real-case dataset reflects an actual parallel-line manufacturing environment and is used to assess the generalization of the trained RMAPPO policy to a concrete deployment scenario. However, for confidentiality reasons, the specific dataset used in this study cannot be disclosed.

Furthermore, lines are not fully flexible. The eligibility matrix restricts which products each line may produce, introducing routing constraints that require the manager agent to make meaningful allocation decisions. Processing times are heterogeneous across eligible line -product pairs, reflecting real differences in line speed and product complexity.

Sequence-dependent setup times and costs apply when a line switches between eligible products. The demand profile spans 30 periods with intermittent zero-demand periods creating natural planning boundary conditions. Product 089 has the highest and most sustained demand, consistently exceeding 5,000 units in several periods. Products 083 and 111 have sparse or zero demand throughout, reducing their scheduling priority relative to the high-volume products.

Multiple independent evaluation runs are conducted by reseeding demand realizations around the fixed demand profile where stochasticity is introduced, or by treating the fixed demand profile as a deterministic benchmark with a single run. Results are compared against key reference algorithms under identical configuration loading.


### **5.1.2. Dataset for Benchmark Comparison (Small, Medium, Large)** {#dataset-for-benchmark-comparison-small-medium-large}

For benchmarking, the repository contains structured generated datasets under `dataset/Small`, `dataset/Medium`, and `dataset/Large`. Each benchmark size contains up to 100 fixed JSON instances named by product count, line count, period count, and instance index. The comparison pipeline currently evaluates the first 10 instances by default because RH2 is expensive on medium and large sizes. Some earlier small-result files contain more completed RH2 entries because they were generated before this limit was standardised.

###### Table 5.1: Generated Instance Set {#table-5.1-generated-instance-set .Table2}

| Scale | Benchmark sizes used by scripts |
|:------|:-------------------------------|
| Small | 5×2×4, 6×2×4, 7×2×4, 8×3×4, 9×3×4, 10×3×4, 11×4×4, 12×4×4, 13×4×4 |
| Medium | 14×5×4, 15×5×4, 16×5×4, 17×6×4, 18×6×4, 19×6×4, 20×7×4, 21×7×4, 22×7×4 |
| Large | 23×8×4, 24×8×4, 25×8×4, 26×9×4, 27×9×4, 28×9×4, 29×10×4, 30×10×4, 31×10×4 |

###### Table 5.2: Parameter Settings of the Generated Instances {#table-5.2-parameter-settings-of-the-generated-instances .Table2}

| Parameter | Value / Distribution in `generate_test_benchmark.py` and `BoschEnv` |
|:---|:---|
| Daily capacity per line, $C_l$ | 120.0 hours |
| Eligibility matrix, $E_{l,p}$ | All ones for generated benchmarks |
| Processing time, $pt_{l,p}$ | $\mathcal{U}[11/60, 15/60]$ hours/unit |
| Production cost, $ProdCost_{l,p}$ | $\mathcal{U}[1.3, 1.6]$ per unit |
| Demand, $D_{p,t}$ | Integer $\mathcal{U}\{0,149\}$ units |
| Hazard rate, $\lambda_l$ | random choice from $\{2.0/120, 3.0/120\}$ |
| CM time per event | $\mathcal{U}[7/60, 11/60]$ hours |
| PM time | $\mathcal{U}[2.5, 3.5]$ hours |
| PM cost | $\mathcal{U}[10.0, 20.0]$ |
| CM cost | $\mathcal{U}[2.0, 3.0]$ |
| Holding cost | 1.75 |
| Backlog cost | 5.25 |
| Flat backlog penalty | 12.25 per product with backlog |
| First setup time | $\mathcal{U}[1.2, 2.4]$ hours |
| First setup cost | $\mathcal{U}[25.0, 30.0]$ |
| Changeover setup time | $\mathcal{U}[1.2, 2.4]$ hours for different products |
| Changeover setup cost | $\mathcal{U}[25.0, 30.0]$ for different products |
| Same-product setup | 0 |

The comparison output for each benchmark is saved to `benchmark_results/{P}_{L}_{T}/comparison_100instances.json`, or to a tagged directory such as `benchmark_results/24_8_4_gated_pm/`. Despite the filename, the current comparison loop uses `N_COMPARE = 10` by default.

## **5.2. Result Illustration and Explanation** {#result-illustration-and-explanation}

This section follows the result tables in the presentation slides. The comparison uses RH2 as the rolling-horizon benchmark and RMAPPO as the proposed three-step hybrid pipeline: Step 1 trains the RMAPPO routing and sequencing policies, Step 2 solves post-hoc LP/MILP lot sizing from the exported allocation mask, and Step 3 re-runs deterministic inference with the corrected quantities. The cost gap is defined as:

$$\text{Gap}=\frac{\text{RH2 Cost}-\text{RMAPPO Cost}}{\text{RH2 Cost}}\times 100\%$$

A positive gap therefore means that RMAPPO produces a lower total cost than RH2.

### **5.2.1. Benchmark Comparison Results (Small, Medium, Large)** {#benchmark-comparison-results-small-medium-large}

###### Table 5.3: Small Instance Results {#table-5.3-real-case-performance-summary .Table2}

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

###### Table 5.4: Medium Instance Results {#table-5.4-small-instance-results .Table2}

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

The medium tier shows the main crossover. RH2 reaches the 4,000-second time limit across the tier, while the RMAPPO pipeline produces decisions in roughly four seconds on average. RMAPPO outperforms RH2 on every medium benchmark in the slide results, with especially large improvements from 20×7×4 onward.

###### Table 5.5: Large Instance Results {#table-5.5-medium-instance-results .Table2}

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

###### Table 5.6: Summary Across Scales {#table-5.6-large-instance-results .Table2}

| Size | Avg. RH2 Cost | Avg. RMAPPO Cost | Avg. Gap | Avg. RH2 Time (s) | Avg. RMAPPO Time (s) | Main Result |
|:---|---:|---:|---:|---:|---:|:---|
| Small | 5,114.6 | 4,825.5 | 5.7% | 1,853.7 | 2.8 | Mixed, RH2 better on smaller cases |
| Medium | 19,894.8 | 10,394.6 | 47.8% | 4,000.0 | 4.2 | RMAPPO better on all cases |
| Large | 28,691.2 | 17,074.4 | 40.5% | 4,000.0 | 5.1 | RMAPPO better on all cases |

Across benchmark sizes, RH2 performs well on small instances but becomes slow and less effective as the problem size increases. RMAPPO achieves much faster inference, typically within seconds, and outperforms RH2 on most medium and large instances.

## **5.3. Sensitivity Analysis** {#sensitivity-analysis}

The sensitivity analysis follows the slide results for the representative 24×8×4 benchmark. Each setting reports the final RMAPPO cost after the three-step pipeline, the runtime in seconds, and the percentage change relative to the baseline:

$$\text{Change vs Baseline}=\frac{\text{Cost}_{setting}-\text{Cost}_{baseline}}{\text{Cost}_{baseline}}\times 100\%$$

Negative values indicate an improvement over the baseline.

### **5.3.1. PM Gating Threshold** {#setup-time-estimation-mode}

This test varies when preventive maintenance becomes available under gated PM. Lower $\tau$ means PM is allowed earlier once line risk accumulates.

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: tau=1.0 | 15,542.9 | 3.4 | baseline |
| tau=0.5 | 12,805.4 | 5.3 | -17.60% |
| tau=0.75 | 14,598.5 | 6.9 | -6.10% |
| tau=1.0 | 15,472.4 | 4.7 | -0.50% |
| tau=1.25 | 15,459.6 | 4.7 | -0.50% |

The best tested setting is $\tau=0.5$. Allowing PM earlier prevents machine-risk accumulation while still preserving enough capacity for production. Higher thresholds behave close to the baseline.

### **5.3.2. Kill-Switch Penalty** {#capacity-safety-factor}

This test varies how strongly the policy is penalised for unmet demand and backlog during Step 1 training.

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: kill=20 | 15,542.9 | 3.4 | 0 |
| kill=10 | 15,104.5 | 9.2 | -2.82% |
| kill=40 | 15,254.3 | 9.0 | -1.86% |

A slightly lower penalty can reduce over-conservative behaviour and improve allocation. A penalty that is too high may force production choices that are costly or inflexible.

### **5.3.3. Entropy Coefficient** {#manager-operational-cost-weight}

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: entropy=0.01 | 15,542.9 | 3.4 | 0 |
| entropy=0.001 | 16,401.3 | 5.2 | 5.52% |
| entropy=0.005 | 14,956.1 | 4.9 | -3.78% |
| entropy=0.02 | 17,031.2 | 5.4 | 9.58% |

Very low entropy limits exploration, while very high entropy makes the policy less stable. A moderate value can improve learning by balancing exploration and exploitation; in the slide results, `entropy=0.005` gives the best entropy setting.

### **5.3.4. Learning Rate Scale** {#learning-rate}

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: lr scale=1.0 | 15,542.9 | 3.4 | 0 |
| lr scale=0.5 | 15,831.7 | 5.0 | 1.86% |
| lr scale=2.0 | 16,253.3 | 5.1 | 4.57% |

Lower learning rate may learn too slowly, while higher learning rate can make policy updates unstable. The baseline learning rate scale appears closest to a good balance in the slide results.

### **5.3.5. PPO Clip Parameter** {#ppo-clip-parameter}

| Setting | Cost ($) | Runtime (s) | Change vs Baseline |
|:---|---:|---:|---:|
| Baseline: clip=0.3 | 15,542.9 | 3.4 | 0 |
| clip=0.2 | 15,053.4 | 4.8 | -3.15% |
| clip=0.5 | 16,133.2 | 5.0 | 3.80% |

Smaller clipping makes PPO updates more conservative and can improve stability. Larger clipping allows bigger updates, but in the slide results it degrades the learned policy.

## **5.4 Environmental, Social, and Economic Impacts** {#environmental-social-and-economic-impacts}

### **5.4.1. Economic Impact** {#economic-impact}

The primary economic impact of the proposed RMAPPO scheduling system lies in its direct reduction of total supply chain costs across the planning horizon. By jointly optimising routing decisions, production quantity allocation, and maintenance scheduling, the system reduces three major categories of cost that are often managed independently in practice.

First, inventory holding costs are reduced because demand-aware RL routing and post-hoc lot sizing encourage the system to build inventory only where demand anticipation justifies the holding cost, rather than maintaining large standing inventories across all products. Second, backlog costs - which are disproportionately penalised in the objective due to the flat backlog penalty $\beta_{p}$ - are minimised by routing production to lines where demand urgency is highest, rather than defaulting to throughput-maximising greedy assignments. Third, maintenance costs are managed proactively: machine agents learn to schedule preventive maintenance in low-demand periods, reducing the expected frequency of corrective failures and the associated disruption costs.

Beyond direct cost reduction, the hybrid LP/MILP-MARL architecture offers economic value through scalability and adaptability. Unlike traditional rule-based schedulers that require manual re-tuning for each new product mix or demand regime, the trained RMAPPO policy generalises across randomised instance parameters, reducing the human effort required for operational scheduling. The post-hoc lot-sizing stage further improves quantity feasibility without requiring RH2 to solve the full rolling-horizon problem from scratch for every instance.

### **5.4.2. Social Impact** {#social-impact}

From a social perspective, improved production scheduling has implications for workforce planning and job quality in manufacturing environments. Predictable, demand-aligned scheduling reduces the need for emergency overtime shifts and last-minute production changes, which are common sources of worker dissatisfaction and fatigue in high-variety manufacturing settings. By learning to balance production loads across lines and manage maintenance proactively, the system supports more stable shift patterns and reduces the frequency of unplanned downtime events that create uncertainty for workers.

Furthermore, the reduction in backlog - a consequence of better demand anticipation - translates directly into improved service reliability for downstream customers. In supply chains serving essential goods, consistent fulfilment of demand has meaningful social value beyond the economic penalty captured in the cost function.

The automation of scheduling decisions also raises considerations around the role of human operators. The proposed system is designed as a decision-support tool rather than a fully autonomous controller: the MILP outputs and policy recommendations can be reviewed and overridden by human planners, preserving operator agency while reducing cognitive burden. This positions the system as augmenting rather than replacing human expertise in production management.

### **5.4.3. Environmental Impact** {#environmental-impact}

The environmental benefits of the proposed system are principally realised through improved capacity utilisation and reduced waste. By allocating production quantities more accurately to match demand, the system reduces overproduction - a significant source of material waste in manufacturing, particularly for perishable or time-sensitive products. The dense production reward structure encourages machine agents to fully utilise allocated capacity before ending a shift, minimising idle energy consumption relative to throughput.

Proactive maintenance scheduling also has environmental benefits. Lines that are well-maintained operate closer to their designed efficiency, consuming less energy per unit produced and generating fewer waste by-products from unplanned failures (e.g., material scrapped during uncontrolled line stoppages). By reducing the frequency of corrective maintenance events - which often involve emergency procedures with higher environmental footprints - the predictive maintenance component of the system contributes to more sustainable plant operations.

More broadly, the framework's applicability to parallel-line manufacturing environments makes it relevant to sectors with significant environmental exposure, including chemical processing, food production, and automotive assembly. Improvements in scheduling efficiency in these industries can contribute meaningfully to sector-level reductions in energy intensity and waste generation, aligning with broader sustainability objectives in industrial operations management.

# **CHAPTER 6: CONCLUSIONS**


## **6.1 Results Discussion and Implication** {#results-discussion-and-implication}

This thesis proposed and evaluated a hybrid Recurrent Multi-Agent Proximal Policy Optimisation (RMAPPO) framework for parallel-line production scheduling with preventive maintenance. In the current implementation, the system decomposes the problem into a trained multi-agent RL layer for routing, sequencing, and maintenance control, followed by a post-hoc LP/MILP lot-sizing layer that translates exported RL allocation masks into capacity-feasible production quantities. The experimental results in the presentation slides provide several conclusions with implications for both theory and practice.

The benchmark results show that the proposed pipeline becomes more valuable as problem size increases. On small instances, the average gap is 5.65%, with RH2 still stronger on the easiest cases but RMAPPO outperforming at the upper end of the small tier. On medium instances, RMAPPO improves over RH2 by 47.8% on average and wins on every benchmark in the slide results. On large instances, RMAPPO improves over RH2 by 40.5% on average while keeping runtime near seconds rather than the 4,000-second RH2 limit.

By delegating final quantity allocation to the post-hoc LP/MILP stage, the manager agent focuses its learning capacity on the combinatorial routing problem. The Step 1 policy learns useful activation masks, while Step 2 converts those masks into feasible lot sizes. This hybrid structure explains why the method scales: RL handles high-level allocation quickly, and the lot-sizing solver only solves the reduced problem conditioned on the learned routing mask.

The gated preventive-maintenance mechanism is also important. By making PM available only when no productive work remains or when accumulated risk exceeds the threshold after production, the policy avoids the older tendency to choose maintenance too early. The slide sensitivity results show that the PM gate threshold materially changes final cost, with `tau=0.5` giving the best tested result at 12,805.4.

The timing results show that RMAPPO generates decisions in seconds, while RH2 often requires hours on medium and large instances. This makes the current pipeline suitable as a decision-support approach where solution quality must be improved without running a full rolling-horizon solve from scratch for every planning instance.

## **6.2 Recommendations for Future Research** {#recommendations-for-future-research}

Several directions are identified for extending this work, spanning algorithmic improvements, problem generalisations, and deployment considerations.

The current system treats demand as deterministic and known over the lookahead window. In practice, demand forecasts carry uncertainty that grows with horizon length. A natural extension is to replace the deterministic MILP with a stochastic programming variant - for example, sample average approximation or a robust MILP with demand uncertainty sets - allowing the allocator to hedge against forecast error. Alternatively, the RL observation vector could be augmented with demand forecast confidence intervals, training the manager policy to modulate production quantities based on uncertainty levels. Given that the benchmark demand profiles include zero-demand periods such as periods 1 and 6 in lines3.json, policies that learn to exploit known demand structure are already emergent; extending this to uncertain demand is a tractable next step.

The post-hoc LP/MILP lot-sizing module is currently non-differentiable and is applied after policy training. Future work could explore differentiable optimisation layers or imitation targets derived from the MILP solution so that quantity-refinement feedback can influence the manager policy during training.

The current architecture fixes (L,P,T) at training time to ensure observation and action space compatibility with the neural network\'s fixed input dimension. As a result, a new factory configuration with different numbers of lines or products requires a new training run. Future work could address this through graph neural network representations of the factory state, where lines and products are nodes and eligibility relationships are edges. Such an architecture would support zero-shot transfer to unseen factory configurations and enable a single trained policy to serve as a universal scheduler across a family of plants - directly extending the generalisation tested across the structured benchmark family.

The current model considers a single plant with parallel lines. Many industrial scheduling problems involve multiple plants, upstream suppliers with stochastic lead times, and downstream distribution networks. Extending the MARL framework to a multi-echelon setting - where plant-level manager agents coordinate with upstream procurement agents and downstream fulfilment agents - is a natural direction given the CTDE architecture already in place. The agent communication structure required for such coordination could be implemented as an extension of the shared machine policy, with inter-plant routing treated as a higher-level action space analogous to the current manager\'s binary mask.

Incorporation of real-time machine condition data. The current degradation model uses cumulative runtime as a proxy for failure risk via the hazard rate. In practice, manufacturing lines are increasingly instrumented with sensors providing real-time health signals - vibration, temperature, acoustic emission - that are more predictive of imminent failure than runtime alone. Replacing or augmenting the age variable in the machine agent\'s observation with such sensor streams would constitute a true condition-based maintenance system and is the natural next step for industrial deployment of the framework.

Deployment of RL-based scheduling systems in real factories requires operators to understand and trust the system\'s decisions. The current framework produces routing masks and maintenance actions that are not natively interpretable. Future work could incorporate attention visualisation over the observation vector to highlight which demand signals or machine ages drove a particular routing decision, counterfactual explanation modules showing what demand conditions would have changed the decision, or constraint-satisfaction certificates confirming that the MILP allocation is capacity-feasible. These tools would reduce the barrier to adoption in regulated industries where scheduling decisions must be auditable.

Large language model-based planning approaches have recently been explored for combinatorial scheduling problems, using chain-of-thought prompting or fine-tuned models as heuristic solvers. A systematic comparison between the RMAPPO framework and LLM-based scheduling on the structured benchmark family would contribute to the emerging literature on the boundary between classical optimisation, reinforcement learning, and foundation model approaches to industrial planning. Given the slide results showing strong RMAPPO advantages on medium and large benchmarks, such a comparison would establish a clear quantitative baseline for future work in this direction.

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
