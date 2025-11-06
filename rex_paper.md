## REX: CAUSAL DISCOVERY BASED ON MACHINE LEARNING AND EXPLAINABILITY TECHNIQUES

Jes´ us Renero 1,3 , Roberto Maestre 1,3 , and Idoia Ochoa 2,3

1 BBVA Madrid, Spain

2 Tecnun, University of Navarra, San Sebasti´ an, Spain

3 Instituto de Ciencia de Datos e Inteligencia Artificil (DATAI), University of Navarra, Pamplona, Spain

October 20, 2025

## ABSTRACT

Explainable Artificial Intelligence (XAI) techniques hold significant potential for enhancing the causal discovery process, which is crucial for understanding complex systems in areas like healthcare, economics, and artificial intelligence. However, no causal discovery methods currently incorporate explainability into their models to derive the causal graphs. Thus, in this paper we explore this innovative approach, as it offers substantial potential and represents a promising new direction worth investigating. Specifically, we introduce REX, a causal discovery method that leverages machine learning (ML) models coupled with explainability techniques, specifically Shapley values, to identify and interpret significant causal relationships among variables.

Comparative evaluations on synthetic datasets comprising continuous tabular data reveal that REX outperforms state-of-the-art causal discovery methods across diverse data generation processes, including non-linear and additive noise models. Moreover, REX was tested on the Sachs single-cell protein-signaling dataset, achieving a precision of 0.952 and recovering key causal relationships with no incorrect edges. Taking together, these results showcase REX's effectiveness in accurately recovering true causal structures while minimizing false positive predictions, its robustness across diverse datasets, and its applicability to real-world problems. By combining ML and explainability techniques with causal discovery, REX bridges the gap between predictive modeling and causal inference, offering an effective tool for understanding complex causal structures. REX is publicly available at https://github.com/renero/causalgraph .

Keywords: Causal Discovery, Explainability, Shapley Values

## 1 Introduction

Causal discovery -the process of identifying cause-and-effect relationships from observational data- is a pivotal challenge in artificial intelligence (AI) and machine learning (ML). Unveiling causal structures enables robust predictions, facilitates counterfactual reasoning, and enhances decision-making processes in complex systems [1]. Traditional methods for causal discovery often rely on statistical tests for independence and structural equation modeling, which may not scale efficiently with high-dimensional data or effectively capture intricate non-linear relationships [2, 3].

In recent years, ML models, particularly deep learning architectures, have achieved remarkable success in predictive tasks. However, these models are typically considered 'black boxes' due to their lack of interpretability. This opacity has led to a growing interest in explainable AI (XAI) techniques, with Shapley values emerging as a prominent method for interpreting model predictions [4]. Shapley values, grounded in cooperative game theory, provide a principled approach to attributing the contribution of each feature to the output of a model by quantifying the average marginal contribution of a feature across all possible subsets of features [5].

Explainable Artificial Intelligence (XAI) refers to a broad set of techniques aimed at improving the transparency of ML models, commonly divided into two categories: (i) models that are interpretable by design, and (ii) post-

hoc explanation methods that analyze complex, often opaque, predictive models after training. This work focuses specifically on post-hoc feature attribution. We rely on Shapley values [5] for this purpose, due to their firm theoretical foundation -particularly their satisfaction of axioms such as efficiency, symmetry, and additivity [5, 6]- as well as the availability of efficient, model-specific implementations [4, 7]. It is important to clarify that feature attributions derived from XAI methods, including Shapley values, should not be interpreted as causal explanations [8, 9].

While Shapley values offer valuable insights into feature importance within a model's predictive framework, the link between feature importance and causal influence is non-trivial. A high Shapley value for a feature indicates a strong impact on the model's predictions but does not necessarily imply a direct causal relationship with the target variable [10, 11]. Moreover, correlations captured by the model may be confounded by hidden variables or represent spurious associations [1]. Therefore, leveraging Shapley values for causal discovery requires careful consideration to avoid misleading inferences.

Despite the advancements in both causal discovery and explainable AI, there is a notable absence of methods that integrate explainability techniques-specifically Shapley values-into the causal discovery process to extract causal graphs. This highlights a significant gap in the intersection of these two fields, offering a promising new direction.

In this paper, we propose a novel method coined REX that integrates Shapley values into the causal discovery process, aiming to uncover causal structures by interpreting feature attributions from ML models. By cautiously exploiting the conceptual connection between Shapley values and causal relationships, REX helps narrow down the set of features that play significant roles in the causal graph. This targeted approach allows us to focus on the most influential variables for further causal analysis, improving the efficiency and accuracy of causal discovery in complex datasets. REX (see Sections 2.2 and 6.1) makes use of Shapley values under the assumptions of Causal Markov Condition (Definition 1) and faithfulness (Definition 2) as a quantitative approximation of conditional dependence. This allows REX to identify candidate causal parents from observational data, not by treating the explanations as causal per se , but by integrating XAI tools into a broader causal discovery framework.

Wedemonstrate REX's competitive performance on synthetic datasets (AppendixG) traditionally employed to evaluate new models, encompassing diverse generation processes, as well as on two real-world complex system datasets. This performance underscores its practical applicability and the advantages over existing methods in addressing real-world causal discovery challenges.

By integrating explainability techniques into the causal discovery workflow REX aims to generate more robust causal graphs, which are fundamental for advancing from associative predictive modeling towards causal inference (i.e., elucidating cause-effect mechanisms and predicting interventional outcomes). The benefits of this specific integration within REX are twofold: first, it enhances the transparency of the causal discovery process itself, as the selection of potential causal links is guided by interpretable Shapley values that quantify feature contributions. Second, by focusing the search on these significant and interpretable contributions, REX can more accurately identify genuine causal relationships from observational data, leading to more reliable causal models.

## 1.1 Related work

Classical approaches to causal discovery are primarily divided into three families: constraint-based methods, scorebased methods, and structural causal models (SCMs).

Constraint-based methods, such as the PC algorithm [2] and Fast Causal Inference (FCI) [12], traditionally build causal structures by iteratively performing statistical conditional independence (CI) tests on the observational data to remove edges from an initially complete graph. These methods explore statistical dependencies in the data to determine causal relationships between variables. Their reliance on direct CI tests applied to the data for edge pruning differs from REX's approach. REX first employs machine learning models to predict each variable (as detailed in Section 3.1) and then uses Shapley values (Section 2.2) derived from these models as a more holistic proxy for conditional dependence, assessing feature importance across numerous predictive contexts rather than through specific CI tests for initial graph construction.

Score-based methods, like Greedy Equivalence Search (GES) [13], search for the model that best fits the data according to a scoring criterion. This approach involves evaluating different graph structures and selecting the one that optimizes an objective function, balancing model fit and complexity. This contrasts with REX 's methodology of constructing the graph from locally significant parent-child relationships identified via Shapley-based feature contributions (Algorithm 2), rather than optimizing a global graph score.

SCMs provide a framework for representing and estimating causal relationships through equations that describe how variables influence one another [1]. Within this category, the LiNGAM algorithm [14] exploits non-Gaussianity in data to identify causal structures in linear models, assuming linear, non-Gaussian, acyclic relationships. REX, however,

uses flexible ML models (Section 3.1) such as DFNs and GBTs that inherently capture non-linearities and does not depend on non-Gaussianity for its core Shapley-based parent selection, reserving additive noise model assumptions primarily for a later edge orientation stage (Section 3.3). Causal Additive Models (CAM) [15] extend these methods by modeling nonlinear relationships using additive noise models, often utilizing techniques like penalized regression for graph estimation. While REX also accommodates non-linearities through its underlying ML regressors and employs an additive noise assumption for orientation, its parent identification process (Algorithm 2) is driven by Shapley-based feature importance derived from these general regressors, not by fitting a specific global additive model structure with penalization for graph selection. However, these classical SCM approaches often require strong assumptions about the data-generating process and may struggle with complex nonlinearities inherent in real-world data [3].

Hybrid methods, such as Max-Min Hill-Climbing (MMHC) [16], combine elements of both constraint-based and score-based approaches, for instance, by using CI tests to restrict the search space for a subsequent score-based optimization phase. These methods aim to leverage the strengths of each approach. REX's integration of ML and explainability is distinct, focusing on using XAI-derived feature importances from predictive models as the primary input for parent discovery before specific graph optimization or CI testing steps.

Recent advancements have led to the application of machine learning techniques in causal discovery. Algorithms like NOTEARS [17] formulate causal discovery as a continuous optimization problem by defining an algebraic acyclicity constraint, enabling gradient-based learning of the entire DAG structure simultaneously. This unified optimization differs from REX 's sequential workflow, which involves distinct stages for predictive modeling, Shapley-based parent discovery (Algorithm 1), edge orientation (Section 3.3), and cycle resolution (Section 3.4). Structural Agnostic Modeling (SAM) [18], in particular, employs adversarial learning to model causal structures without making strong assumptions about the data distribution. These methods improve scalability and can capture nonlinear relationships but may still face challenges related to the interpretability of the discovery process itself [19], a domain REX specifically addresses through its inherent use of XAI techniques to guide graph construction.

## 2 Preliminaries

Let X = { X 1 , X 2 , . . . , X p } represent a vector of p continuous variables, with unknown joint probability distribution P ( X ) . The observational causal discovery setting considers m independent and identically distributed (i.i.d.) samples drawn from P ( X ) . The goal is then to infer the causal directed acyclic graph (DAG) G = ( V, E ) , where V is the set of variables and E is the set of directed edges ( X j → X i ) representing the causal relationship 'X j causes X i '.

Causal relation vs. statistical dependence Throughout the paper we distinguish two notions:

- (i) Statistical dependence Two variables X and Y are (marginally or conditionally) dependent if their joint distribution under passive observation does not factorize, e.g., X ̸ ⊥ ⊥ Y | Z . Dependence is a purely observational property.
- (ii) Causal relationship We say X is a cause of Y (denoted X → Y in the causal graph) when intervening on X changes the distribution of Y : P ( Y | do( X = x )) depends on x [1]. Causation implies a directed edge in the underlying interventional DAG, but does not require X and Y to be statistically dependent (collider structures are the standard counter-example).

The Causal Markov Condition (CM) states that, given the causal graph G , the joint probability distribution P ( X ) factorizes as:

<!-- formula-not-decoded -->

where Pa ( i, G ) represents the parents of X i in the graph. In other words, the CM states that, given a causal graph, each variable is conditionally independent of its non-descendants, given its parents in the graph. This condition is crucial for causal inference as it ensures that the relationships in the graph can be used to explain the dependencies in the data.

Definition 1 (Causal Markov property) . P is Markov with respect to G if each variable is independent of its nondescendants, conditioned on its parents:

<!-- formula-not-decoded -->

Faithfulness refers to the assumption that all conditional independencies present in the observed data are reflected in the structure of the causal graph. Formally, the distribution P in Eq. (1) is said to be faithful to G if the only conditional independencies in P are those implied by the d-separation in G [2, 1]. This assumption ensures that there are no accidental independencies in the data due to parameter values (e.g., when causal effects perfectly cancel each other out).

Definition 2 (Faithfulness) . P is faithful to G if every conditional independence that holds in P is entailed by the DAG via d -separation:

<!-- formula-not-decoded -->

Equivalently, the set of conditional independences of P coincides with the set implied by G .

Faithfulness, combined with the Causal Markov Condition, is essential for inferring the correct causal structure from observational data, as it guarantees that the observed statistical relationships correspond directly to the structure of the causal graph. Additionally, in the causal discovery setting, the underlying model that generates the data is assumed to be a general Structural Equation Model (SEM) as follows:

<!-- formula-not-decoded -->

where f i is a function from R | Pa ( i, G ) | +1 → R , and ε i is a unit centered Gaussian noise. SEM is a framework for modeling causal relationships where each variable is expressed as a function of its direct causes (parents) and an error term, typically assumed to be independent noise. SEMs are also essential for causal discovery because they provide a formal way to represent and quantify the effects of interventions and establish the relationships between variables.

Causal discovery aims to infer cause-effect relationships between variables in observational data, typically represented by a causal graph. A causal graph, often modeled as a DAG, consists of nodes representing variables and directed edges indicating causal influences. The primary challenge in causal discovery then, is to identify these causal relationships without direct evidence from interventions, relying instead on specific assumptions about the data and its underlying generative processes.

Definition 3 (Causal-discovery problem) . Given m i.i.d. samples from an unknown P that is Markov and faithful to some (unknown) DAG G , causal discovery asks for an estimator that, as n →∞ , identifies with high probability the Markov-equivalence class

<!-- formula-not-decoded -->

or a representative DAG in that class.

Throughout the paper we work under these standard assumptions ([1, 2]); Section 4 empirically evaluates how the proposed REX algorithm performs when the assumptions are met or mildly violated.

While causal discovery leverages patterns of statistical (in)dependence under the Markov and Faithfulness assumptions, the terms causal relation and statistical dependence are not interchangeable. The synthetic experiments in Appendix A (confounder, chain, collider) illustrate cases where high marginal dependence hides a non-causal variable and vice-versa.

## 2.1 SHAP values and causal relationships

When the predictor is sufficiently expressive, adding a variable X j to an existing feature set S improves the outof-sample prediction of the target Y only if X j contributes information about Y that is not already contained in S . Hence, a near-zero marginal improvement suggests the conditional independence X j ⊥ ⊥ Y | S . This link between lack of predictive gain and conditional independence underlies many causal-discovery algorithms, which use such independencies to distinguish direct causal links from indirect (mediated) or spurious (confounded) associations when building a causal graph.

Related SHAP-causality studies Several recent papers have examined how Shapley attributions might interface with causal reasoning [20]. introduces causal SHAP , which assumes a known DAG and re-weights coalitions accordingly; [21] applies standard SHAP values to a domain-specific regression task (coal-bed-methane wells) and discuss causal plausibility qualitatively; and [22] surveys ways in which explainers may hint at causality, but stops short of a data-driven discovery algorithm. In contrast, the present work derives an explicit link between aggregated SHAP values and conditional dependence (Section 2.2, Eq. 11) and embeds that link in a full pipeline (Sections 3-4) that discovers the causal graph directly from observational data without prior causal knowledge.

## 2.2 Mathematical Foundation

In this section, we establish the theoretical connection between Shapley values and conditional independence within the context of causal discovery. This foundation supports the use of Shapley values in identifying causal relationships by relating feature importance measures to probabilistic dependence structures.

Let F be the set of all features (variables), excluding the target variable X i . Each feature X j ∈ F is considered a 'player' in a cooperative game where the objective is to predict X i . We define a function ˆ f ( S ) : 2 | F | → R that maps a subset of features S ⊆ F to a real number representing the contribution of S to predicting X i , with | F | being the cardinality of F . The classical definition of the Shapley value ϕ j for feature X j is given by:

<!-- formula-not-decoded -->

where ˆ f ( S ∪{ X j } ) -ˆ f ( S ) represents the marginal contribution of feature X j to the prediction of X i given subset S .

## 2.2.1 Marginal Contributions and Conditional Independence

In [23], a connection between Shapley value summands and conditional independence is demonstrated by relating conditional independence in a faithful Bayesian network with the summands of the Shapley value. Specifically, within this framework, if a variable X j provides additional information about the target variable given a subset S of other variables, this relationship is captured by the Shapley value summands for X j given S . This interpretation aligns with the concept of conditional independence, where the Shapley summand reflects the additional predictive contribution of X j in the context of S . In causal discovery, assuming causal sufficiency and faithfulness, SHAP values can indicate conditional independence relationships within a causal DAG, providing an interpretable approach to uncovering causal relationships rather than mere associations.

To connect Shapley values with conditional independence, consider the marginal contribution ∆ j,S of feature X j for a specific subset S ⊆ F \{ X j } , given by ∆ j,S = ˆ f ( S ∪ { X j } ) -ˆ f ( S ) . Assuming that ˆ f ( S ) accurately reflects the predictive power of S for X i , the marginal contribution ∆ j,S relates to the conditional dependence between X i and X j given S . Specifically:

- If X j provides no additional information about X i given S , then X i is conditionally independent of X j given S , denoted as X i ⊥ ⊥ X j | S . In this case p ( X i | X j , S ) = p ( X i | S ) , and the marginal contribution is zero ( ∆ j,S = 0 ).
- If X j provides additional predictive information about X i given S , then X i is conditionally dependent on X j given S , denoted as X i ̸ ⊥ ⊥ X j | S . In this case:

̸

<!-- formula-not-decoded -->

and the marginal contribution is positive ( ∆ j,S &gt; 0 ).

## 2.2.2 Aggregating over all subsets

The Shapley value ϕ j aggregates the marginal contributions ∆ j,S over all subsets S of F \{ X j } , weighted by the Shapley weights:

<!-- formula-not-decoded -->

These weights satisfy ∑ S w ( S ) = 1 and reflect the importance of each subset in the calculation.

Remark. Equations (4) and (6) are reproduced verbatim from the original work of Shapley (1953) [5]; we include them only to fix notation.

## 2.2.3 Shapley-weighted Conditional Dependence Indicator

To formalize the relationship between Shapley values and conditional independence, we introduce a new indicator function I j,S :

̸

<!-- formula-not-decoded -->

Using this indicator function, the Shapley value ϕ j can be expressed as:

<!-- formula-not-decoded -->

Assuming that the average marginal contribution ∆ j is approximately constant across subsets where X j is conditionally dependent on X i , we can approximate ϕ j as:

<!-- formula-not-decoded -->

## 2.2.4 Weighted probability of conditional dependence

The term ∑ S w ( S ) · I j,S represents the weighted probability that X j is conditionally dependent on X i across all subsets S , under the probability distribution defined by the Shapley weights w ( S ) , i.e.,:

<!-- formula-not-decoded -->

Thus, the Shapley value ϕ j can be interpreted as:

<!-- formula-not-decoded -->

where P weighted ( · ) abbreviates the Shapley weighting over coalitions. Equation (11) states that a feature's Shapley value ϕ j is proportional to both the average dependence strength ∆ j and the Shapley-weighted probability that X j is conditionally dependent on Y . [23] showed that a single Shapley summand f ( S ∪{ X j } ) -f ( S ) vanishes if and only if X j ⊥ ⊥ Y | S in a faithful Bayesian network. In contrast, Eqs. (6)-(10) aggregate all summands under the original Shapley weights, yielding the Shapley-weighted probability of conditional dependence in Eq. (10)-a quantity that did not appear in [23] and underpins the causal-discovery criterion of the present work.

Controlled synthetic experiments (Appendix A, Table 5) support this claim across confounder, chain, collider, and collinear structures (see Section 2.2.6).

## 2.2.5 Implications for causal discovery

The relationship between Shapley values and conditional dependence suggests that features with higher Shapley values are more likely to be conditionally dependent on the target variable X i , while features with lower Shapley values are more likely to be conditionally independent. This connection provides a theoretical justification for using Shapley values in causal discovery.

More formally, a high Shapley Value ( ϕ j large) indicates that X j frequently contributes significant predictive information about X i across various subsets S , suggesting a potential causal relationship or strong association. On the contrary, a low Shapley Value ( ϕ j small) implies that X j rarely provides additional predictive information about X i , indicating possible conditional independence and a lower likelihood of a direct causal link.

Leveraging this theoretical foundation, Shapley values can be utilized to estimate the conditional dependence structure among variables. However, it is important to acknowledge the assumptions underlying this connection, particularly for the approximation in Eq. (11). This approximation presumes that the average marginal contribution ∆ j is relatively consistent across subsets where X j is conditionally dependent on X i , though in practice, individual marginal contributions ∆ j,S can vary significantly, making the constant ∆ j a simplification. Furthermore, the predictive function ˆ f ( S ) is assumed to accurately capture true conditional probabilities p ( X i | S ) . Finally, for a causal interpretation of the SHAP-derived insights within a discovery context, the core assumptions of faithfulness and causal sufficiency are necessary to ensure that inferred relationships represent causal connections rather than mere statistical dependencies.

Despite these assumptions, the established relationship provides valuable insights into how Shapley values can reflect the underlying conditional independence structures, supporting their use in causal inference tasks.

## 2.2.6 Divergence between dependence and Shapley

The theoretical connection articulated above, particularly Equation (11), is supported by controlled synthetic experiments (see Appendix A). These experiments, conducted across canonical causal structures (confounders, chains, colliders, and collinear parents), highlight several key insights:

Divergence from single-set conditional independence (CI) tests . Shapley values differ notably from traditional single-set CI tests (e.g., Fisher-Z tests). For example, in the confounder ( Z → X , Z → Y ) and chain ( X → Z → Y ) structures predicting Y , a feature X may appear conditionally independent of Y given Z (yielding a non-significant Fisher-Z p-value, i.e., X ⊥ ⊥ Y | Z ), yet still receive substantial non-zero Shapley values. This occurs because Shapley values aggregate contributions across multiple feature coalitions, including subsets not conditioning on Z , thereby capturing broader predictive relevance.

Robustness under high correlation scenarios : In scenarios involving highly collinear predictors (e.g., X 1 ≈ X 2 → Y ), Shapley values exhibit greater stability and interpretability compared to traditional CI tests or regression coefficients, which often become unstable or ambiguous. Empirical results consistently show stable Shapley values

for collinear features, reflecting intuitive allocation of feature importance, consistent with properties of interventional explainers like TreeExplainer (discussed in Section 6.2).

Sensitivity to true causal drivers: Features that act as direct causes or strong mediators consistently receive high Shapley values, accurately reflecting high conditional dependence probabilities. For instance, mediators and direct causal features in confounder, chain, or collider structures consistently display elevated Shapley values, highlighting their predictive importance.

Overall, these empirical findings underscore that Shapley values, despite being model-dependent, provide comprehensive and robust assessments of feature contributions by aggregating across all feature subsets, making them valuable for causal discovery methods like REX under assumptions of faithfulness and causal sufficiency.

## 3 Proposed method

The proposed method REX approaches the causal discovery problem in a series of steps (see Fig. 1). First, we finetune and train a series of regressors to predict each variable X i , for i = 1 , . . . , p , using the remaining p -1 variables in the dataset, denoted as X \ i = { X 1 , . . . , X i -1 , X i +1 , . . . , X p } . Once the regressors are fine-tuned and trained, we compute Shapley values to estimate the contribution of each variable to the prediction, applying a bootstrapping mechanism, i.e., iteratively using different samples of the dataset. Through this repeated sampling, REX identifies a robust set of features that can be considered the potential causes (i.e., Pa ( X i ) ) for each target variable X i . As a final step, results from the different regressors are combined to obtain a plausible causal graph, which is reviewed to direct edges and remove eventual cycles. Next, we describe these steps in detail. It is worth noting that while the overall structure of the REX pipeline-training predictive models, assessing feature contributions, and constructing a graph-is a general approach, the core novelty of REX lies in its specific instantiation using Shapley values for robust feature impact assessment and parent selection, guided by the theoretical connections discussed in Section 2.2, and its subsequent steps for graph refinement.

Figure 1: Overview of the REX workflow. The process begins with training regressors (Section 3.1), followed by a bootstrapping procedure (Algorithm 1) that includes data sampling, computation of SHAP values ( ϕ ) to assess feature impact (Section 3.2.1), selection of candidate parents based on these values (Algorithm 2, Section 3.2.2), and updating the adjacency matrix (Section 3.2.3). Edges are then oriented (Section 3.3), and outputs from multiple regressors are combined and pruned to produce the final DAG (Section 3.4). While parts of the pipeline are general, REX is specifically designed to leverage Shapley values for assessing feature contributions, as motivated in Section 2.2.

<!-- image -->

## 3.1 Models training

The first step in deriving the causal model is to train a separate model to predict each variable in the dataset, with hyperparameter optimization (HPO). Before model training, the data is normalized by removing the mean and scaling to unit variance. To ensure robustness, we employ two complementary regressors: a deep feed-forward neural network (DFN) and gradient boosting trees (GBT). DFNs are highly flexible models capable of approximating any function, leveraging their capacity as universal approximators, as demonstrated by [24]. GBTs, known for their effectiveness in handling tabular continuous data and offering more interpretable outputs as shown in [25], serve to validate and complement the results obtained from the DFN models, with the XGBoost implementation used in this study. Although the REX framework allows for the integration of additional regressors, the results from DFN and GBT have proven to be sufficiently robust for our purposes (see Appendix D).

Atotal of p DFN and p GBT models are trained to predict each feature X i ( i = 1 , . . . , p ) from the remaining variables X \ i , using mean squared error (MSE) as the loss function. To enable DFNs, which are deterministic architectures, to effectively model stochastic relationships inherent in the data-generating process (Eq. 2), we incorporate an additional i.i.d. Gaussian noise variable ν ∼ N (0 , σ 2 ) as an input feature. This technique, inspired by generative causal modeling [26] and with ν re-generated at each training step, allows the DFN to use this input noise to model the influence of the exogenous SEM noise term ε i , rendering the learned predictive functions ˆ f i ( X \ i , ν ) inherently non-deterministic.

This imparted stochasticity is crucial for REX as it: (i) yields more stable and graduated Shapley scores, particularly with low-noise or deterministic synthetic data, by preventing numerical artifacts that can arise from perfectly deterministic learned functions; and (ii) encourages DFNs to capture richer conditional distribution information beyond just the mean, thus leading to a more nuanced SHAP-based feature importance assessment for robust parent selection (Algorithm 2). The trained models are therefore defined as ˆ f i ( X \ i , ν ) = ˆ X i ≈ X i for DFNs, and ˆ f i ( X \ i ) = ˆ X i ≈ X i for GBTs (without the additional noise input).

The model hyperparameters are determined in the training phase for each dataset, using a tree-structured Parzen estimator (TPE) [27] with an 80/20 split for train and validation. Hyperparameter optimization is a critical step, as one of the assumptions stated in Section 2.2.5 is that regressor ˆ f accurately captures the true conditional probabilities.

## 3.2 Bootstrapping and initial causal graph construction

In this phase, we implement the bootstrapping approach to construct an initial undirected causal graph ( ̂ G undir). This process involves three main steps: computing SHAP values to assess feature importance, selecting potential parent features based on their impact, and updating an adjacency matrix that represents candidate causal connections. These steps are iteratively repeated in the bootstrapping loop (Algorithm 1) to build a stable causal structure.

## Algorithm 1

```
Require: Let ˆ f be a fine-tuned regressor model (DFN or GBT) 1: Initialize empty adjacency matrix A ∈ R p × p 2: for t = 1 to T do ▷ Number of bootstrap iterations 3: Sample a subset X ( t ) from X (observational data) 4: for each feature X i do 5: Fit SHAP explainer with X ( t ) ∖ i and model ˆ f to obtain Φ ( i ) = { ϕ 1 , . . . , ϕ i -1 , ϕ i +1 , . . . , ϕ p } 6: Run Algorithm 2 with Φ ( i ) to obtain Pa ( X i ) 7: for each X j ∈ Pa ( X i ) do 8: Increment the corresponding entry: a i,j ← a i,j +1 9: end for 10: end for 11: end for 12: Normalize: a i,j ← a i,j T for all ( i, j ) 13: Filter edges: a i,j ← 0 if a i,j < τ 14: return Adjacency matrix A representing a stable undirected graph ̂ G undir
```

## Algorithm 2

```
Bootstrapped DAG Construction Parent Selection from SHAP Values
```

```
Require: Input array Φ = { ϕ 1 , . . . , ϕ p -1 } (SHAP values) 1: Λ ← Pairwise Euclidean distances in Φ 2: Λ ′ ← Sort elements of Λ in descending order 3: Initialize ζ ← max(Λ) + constant 4: repeat 5: Run DBSCAN on Φ with parameter ζ 6: n ← Number of clusters 7: if n = 1 then 8: ζ ← ζ -max(Λ ′ ) 9: Remove max(Λ ′ ) from Λ ′ 10: end if 11: until n > 1 or Λ ′ = ∅ 12: if no clusters formed then return None 13: elsereturn Features in cluster with highest mean SHAP value 14: end if
```

## 3.2.1 Computing SHAP values for feature impact

The first step in Algorithm 1, after sampling a subset of the data, is to compute SHAP values for each feature to identify its impact on the target variable in each regressor model. For a given model (DFN or GBT), the SHAP value ϕ j is calculated for each feature X j when predicting the target variable X i . This value indicates the marginal contribution of X j to the prediction of X i , allowing us to interpret ϕ j &gt; 0 as evidence that X j is likely to influence X i (i.e.,

X i ̸ ⊥ ⊥ X j ). This computation provides a foundation for identifying variables that may play a causal role in influencing each target variable, as described in Section 2.2.5.

## 3.2.2 Parents selection

In the second step of Algorithm 1, we apply the DBSCAN clustering algorithm to group features based on their SHAP values and select those that have a significant impact on predicting the target variable. Feature clusters are adjusted dynamically by decreasing the neighborhood radius parameter ζ until more than one group are detected, helping to separate influential features from less impactful ones. This clustering process effectively selects potential parent features, as it isolates variables that likely have causal influence on the target. Algorithm 2 outlines this clustering procedure in detail, and Appendix B describes the process with a visual example and additional details.

DBSCAN is selected for clustering SHAP values because, unlike methods such as K-Means, it does not require specifying the number of clusters upfront-a crucial advantage given the data-dependent and unknown grouping of feature importances. Moreover, DBSCAN intrinsically distinguishes influential parent candidates (high SHAP values) from negligible-impact features by labeling low-density points as noise, a capability partition-based methods lack. Algorithm 2 further improves DBSCAN by adaptively setting the neighborhood radius parameter ζ according to the empirical distribution of SHAP values, avoiding arbitrary thresholds and providing a principled distinction between salient and negligible feature contributions.

Simpler heuristic methods, such as using average SHAP values with percentile thresholds, were explored but produced worse results, as they struggled to effectively distinguish between features with varying levels of impact.

## 3.2.3 Adjacency matrix update and graph construction

With the influential features identified, we proceed to construct an initial causal graph by updating an adjacency matrix, which records candidate edges between features. In each bootstrapping iteration, a subset of the data X ( t ) is sampled, and SHAP-based parent selection is performed. For each pair of selected features, the corresponding entry in the adjacency matrix A is incremented, capturing the frequency of selection across iterations. After bootstrapping, we apply a threshold τ to filter out edges with low frequencies, retaining only stable and meaningful connections. This thresholding process yields a plausible undirected causal graph, representing the initial structure without directed edges ( ̂ G undir). Thus, the stability of an edge in this initial graph is directly determined by the parameter τ ; an edge is included if its frequency of selection during the bootstrapping procedure (Algorithm 1) meets or exceeds this threshold, signifying its robustness against variations in data subsamples.

Step 3 of Algorithm 1 involves creating a bootstrap sample X ( t ) by selecting a subset of the original data samples (i.e., rows) from the full dataset X . To ensure robust coverage of the dataset across the T bootstrap iterations, the proportion c of original data samples to be included in each such bootstrap sample X ( r ) is set using the formula c ≥ 1 -q 1 /T . In this formula, q is the maximum acceptable probability that any given original data sample (row) is not selected across any of the T bootstrap iterations (typically q = 0 . 01 ), and T is the total number of iterations for the bootstrapping mechanism (typically T = 50 , as specified for Algorithm 1).

The undirected graph ̂ G undir produced at this stage is therefore stable under bootstrap perturbations; only these vetted edges are passed to the subsequent orientation module of Section 3.3.

## 3.3 Directing edges

Once a stable undirected graph ̂ G undir has been obtainedi.e. , after the bootstrapping procedure of Section 3.2.3 has normalized the adjacency matrix and applied the frequency threshold τ in Algorithm 1 (lines 12-13) to retain only edges with selection probability f ij ≥ τ -the next step is to orient the remaining candidate edges. The direction of each such pre-identified edge ( X i , X j ) is established using principles from Additive Noise Models (ANMs), as described in [3]. This ANM-based orientation procedure involves fitting regression models for both potential directions (e.g., X j as a function of X i , and X i as a function of X j ).

To determine the causal direction, we then use the Hilbert-Schmidt Independence Criterion (HSIC) [28] to test for independence between the residuals of each regression and the respective predictor variable. For a potential edge X i → X j , if the HSIC test accepts the hypothesis that the residuals from regressing X j on X i are independent of X i , we infer the causal direction X i → X j . Conversely, if the HSIC test indicates dependence between the residuals and X i (i.e., the null hypothesis of independence is rejected, typically if the test p -value is below a significance threshold), this direction X i → X j is considered not supported by the ANM assumption. The process is repeated for the reverse

direction. Thus, the HSIC test serves to ascertain whether the statistical dependence flagged by the Shapley values aligns with an identifiable ANM structure in one direction over the other.

The resulting graph is a directed acyclic graph (DAG) that captures the inferred causal relationships between the features.

## 3.4 Final DAG

The methodology presented here culminates in the generation of the final DAG, denoted as G REX . This final DAG is produced by taking the union ( ∪ ) of the two DAGs G DFN and G GBT, which are generated by the described procedure when using DFN and GBT as the regressors, respectively (see Appendix D for a performance comparison among G REX , G DFN and G GBT ).

The union ( ∪ ) is chosen to leverage the complementary strengths of these diverse model types in identifying potential causal links, aiming for a more comprehensive discovery of true edges. This strategy is empirically supported by a higher overall F1-score compared to using intersection or individual regressors, as detailed in Appendix D. The implementation provides a configurable parameter that allows users to choose between computing the union or the intersection, depending on their specific requirements.

The final DAG G REX must be acyclic to represent valid causal relationships. Hence, in those cases where the new proposed DAG contains cycles or bidirectional edges between two nodes, as a result of the union of two DAGs, we employ a strategy based on the use of the SHAP discrepancy δ ( i ) j to make it acyclic.

The SHAP discrepancy is introduced as a measure to assess how well the Shapley values of a feature explain the variability in the target variable. It quantifies the extent to which the contributions of a feature X j (captured by its Shapley values ϕ j ) align with the actual values of X i (see Appendix C for more details), computed as follows:

<!-- formula-not-decoded -->

where x k,i is the k -th observation of X i , ϕ k,j is the corresponding Shapley value of X j for predicting X i , X i is the mean of X i , and m is the total number of samples.

A lower SHAP discrepancy indicates that the Shapley values ϕ j closely approximate X i , suggesting a stronger potential causal influence from X j to X i . In the context of removing cycles in the DAG, the SHAP discrepancy helps identifying which edges are less supported by the data. Briefly, when a cycle is detected, edges with higher discrepancies are considered for reorientation or removal to break the cycle, ensuring the resulting graph accurately represents the underlying causal relationships.

The proposed method iteratively examines each detected cycle and seeks to resolve it by removing edges with the highest SHAP discrepancy. In particular, for each cycle, the proposed strategy computes the SHAP discrepancies for each edge in the cycle. By comparing these discrepancies, it identifies edges that could reduce the overall discrepancy by reversing their direction, indicating potential misorientations. Reorienting such edges can break the cycle while preserving the causal relationships suggested by the data. If reorientation is insufficient or not possible, the proposed method removes the edge with the largest SHAP discrepancy-implying it is the weakest link in the cycle-to eliminate the cyclicity.

This process ensures that the final graph is acyclic and retains the most significant causal structures inferred from the data. The algorithm effectively balances the need to break cycles with the goal of maintaining the integrity of the underlying causal relationships.

## 3.5 Visual example

The intermediate DAGs obtained by REX are summarized, in a simplified way, in Fig. 2. Each of the regressors tuned and trained to predict each of the variables in the dataset (Section 3.1) are used to compute SHAP values (Section 3.2.1), and from them, select the candidate parents (causes) of each variable (Section 3.2.2). Running SHAP from the trained models to select parents is done following a bootstrap approach, by sampling different portions of the data at each iteration to fit the selected Shapley explainer. After that, a filtered adjacency matrix (Section 3.2.3) is used to outline an unoriented graph, for each of the two regressors (Fig. 2a,b). As a next step, the unoriented graphs become directed graphs (Fig. 2c,d), by using ANM (HSIC) (Section 3.3). These may contain edges that do not exist in the true DAG (e.g., B ← C in (c)), or wrongly directed edges (e.g., E ← C in (c) and D → C in (d)). The union of these two graphs results in the graph depicted in Fig. 2e, where all edges found by any one of the regressors are

included. However, this union operation may result in some double-directed edges and cycles (e.g., C ↔ D , C ↔ E and B → D → C → B in (e)) that need to be resolved (Section 3.4) to produce the final DAG (Fig. 2f).

Figure 2: A visual summary of the intermediate graphs generated by REX to obtain the final DAG G REX . (a)-(b) represent the unoriented graphs for each of the two regressors after running the steps described in Sections 3.1, 3.2.1, 3.2.2 and 3.2.3. (c)-(d) are the result of establishing the direction of each edge presented in (a)-(b), respectively, following the approach described in Section 3.3. (e) is the initial DAG resulting from the union of DAGs (c)-(d). (f) is final DAG G REX resulting from removing cycles and/or bidirectional edged from (e), following the steps described in Section 3.4. The true DAG is also depicted for comparison. Green arrows represent correctly predicted edges, orange ones edges with the incorrect direction, and red ones edges that are missing in the true DAG.

<!-- image -->

## 4 Results on synthetic data

We first consider synthetic data for the evaluation, as it allows to operate under controlled assumptions, where the causal structure is fully specified, and the data generation process (see Appendix G) is explicitly governed by known structural equations. Consequently, in this synthetic setup, there are no unobserved confounders (i.e., variables that are common causes of two or more measured variables which could induce spurious associations), thereby fulfilling the causal sufficiency assumption (which posits that all such common causes are indeed part of the observed dataset). The structural equations are also stable under interventions, while all other structural equations remain unchanged (i.e., actively setting a variable's value to observe its effects on downstream variables, as in P ( Y | do( X = x ) ) [1]). Furthermore, the data respects the faithfulness assumption (formally defined in Section 2 and Definition 2), meaning all observed conditional independencies accurately reflect the d-separations in the true causal graph, with no accidental independencies due to parameter choices.

We conducted experiments using five families of synthetic datasets, each corresponding to a specific type of causal relationship: linear, polynomial, sigmoid additive, Gaussian additive, and Gaussian mixed models. These datasets vary in complexity and non-linearity, and were generated following the methodology described in [26] and [29].

For each synthetic generation mechanism family, we generated datasets with p = 10 , 15 , 20 and 25 variables, each containing m = 500 samples (see Table 1 for details on the distribution of number of edges in the ground truth DAGs). For a robust and reliable evaluation, 10 different datasets were generated for each combination of mechanism family and variable count, resulting in 200 datasets (5 families × 4 variable counts × 10 datasets). All synthetic datasets used to benchmark REX in this study have been made available in the GitHub repository, along the parametrization used for REX and compared methods.

These datasets, together with their corresponding unique ground truth DAGs, are used to benchmark REX against the state-of-the-art causal discovery methods (hereafter referred to as compared methods ) PC [2], NOTEARS [17], GES [13], LiNGAM [14], FCI [12], and CAM [15]. NOTEARS was trained and thresholded with the optimum value found for all datasets used in the study. The rest of the methods have been used with default parametrization.

It is important to note that while these default configurations do not typically incorporate an explicit internal bootstrapping loop for edge stability in the manner of REX, our method's bootstrapping procedure (Algorithm 1) is an integral component designed to ensure the robustness of parent sets derived from SHAP values of its machine learning regressors (DFNs and GBTs), whose explanations can exhibit variability with data resampling.

F1

0.9

0.8

0.7

U.

0.5

0.4

0.3

O0

10

20

300

250-

200

100

Table 1: Average and standard deviation number of edges in the generated DAGs for p = 10 , 15 , 20 , 25 .

25

3500

3000 -

2500 -

= 2000-

A 1000-

1000 -

| Nr of features              | 10         | 15          | 20          | 25           |
|-----------------------------|------------|-------------|-------------|--------------|
| Nr of edges in ground truth | 15 ± 5 . 5 | 23 ± 7 . 62 | 33 ± 6 . 47 | 42 ± 10 . 31 |

All computations were performed on a system equipped with an Apple M2-Pro processor and 32 GB of RAM, without the use of parallelization or GPU acceleration.

## 4.1 Evaluation metrics

To compare a predicted Directed Acyclic Graphs (DAG) against a ground truth DAG, we first compute the True Positives (TPs), False Positives (FPs), and False Negatives (FNs). We define a TP as an edge with the correct direction in both the predicted and ground truth DAGs; a FP as an inferred edge that is not present, or has the opposite direction, in the ground truth DAG; and a FN as an edge present in the ground truth but missing or misoriented in the predicted DAG. Note that a true edge inferred but with the opposite direction is counted both as a FP and a FN. With these metrics, we then compute standard graph-based metrics: precision, recall, and F1-score. Precision is the ratio of correctly predicted edges (TPs) to all predicted edges (TPs + FPs), recall is the proportion of true edges recovered in the predicted DAG (i.e., TPs/(TPs+FNs)), and the F1-score is the harmonic mean of precision and recall. We also consider the difference in the number of edges between the predicted DAG and the true DAG, to evaluate whether a method tends to over- or under-estimate the number of edges.

Additionally, we calculate the Structural Hamming Distance (SHD) [16], which measures the difference between two DAGs by counting missing or extraneous edges, and Structural Intervention Distance (SID) [30], which evaluates how closely two DAGs agree on causal inference statements derived from interventions. Unlike SHD, SID focuses purely on the causal implications of the graph structure, making it particularly suitable for assessing causal discovery algorithms.

A high-performing DAG will demonstrate high precision, recall, and F1, alongside low SHD and SID values.

## 4.2 Experimental results

Wefirst assess REX as a function of the number of variables p , across the five considered families of synthetic datasets, to evaluate its performance with increasingly complex datasets.

Looking at the F1 score (Fig. 3a), we observe that REX achieves high predictive performance, especially for lower and moderate values of p , with median F1 scores consistently above 0.7. This reflects REX's capacity to balance precision and recall effectively in identifying causal relationships. Although some variability is present, the high F1 scores across different values of p underscore REX's robustness. However, there is a slight decrease in the median F1 score for the highest value of p , suggesting that scalability might pose a limitation to its predictive power in very high-dimensional settings.

Figure 3: Effect of increasing the number of variables in the input dataset with REX in the following metrics: (a) F1 score; (b) SHD to true DAG; (c) SID to true DAG; and (d) computation time. Results are computed for the five considered families of synthetic data.

<!-- image -->

In terms of SHD (Fig. 3b), REX exhibits relatively low SHD values, with a concentrated distribution, particularly for smaller values of p . This suggests that REX maintains structural accuracy effectively, even as the dimensionality

80

60

40

grows. However, there is a noticeable upward trend in SHD for larger values of p , indicating that while REX performs well in smaller graphs, its structural precision may gradually decrease as the dimensionality increases. As for SID (Fig. 3c), we also see remarkable low values for lower p and a similar trend to that shown for SHD, as the number of variables increases.

Next, we evaluate the computation time (Fig. 3d), showing that REX 's time requirements increase as p grows, which is an expected outcome given the combinatorial nature of SHAP values computation. While the computational cost becomes more significant at higher values of p , the performance remains reasonable for moderate-dimensional datasets. Users should be aware that runtime will increase substantially with higher-dimensional data, indicating a potential challenge in scalability.

Overall, REX demonstrates strong performance in terms of structural accuracy and F1 score, especially for small to medium-sized datasets, maintaining computational demands and structural precision acceptable for moderate values of p .

## 4.2.1 Results by synthetic dataset

When evaluating these metrics per synthetic dataset family (see Appendix F, Fig. 14), we do not observe significant differences. The results indicate that REX performs robustly across a range of complexities in data structure, showing consistent trends across various settings.

REXachieves high F1 scores (Fig. 14a) across all dataset families, with particularly strong performance in the Gaussian (additive), Gaussian (mixed), and Sigmoid (additive) families, where F1 scores remain above 0.6 even as p increases. In contrast, the Linear and Polynomial datasets show more variability, particularly at higher p values, indicating greater difficulty for REX in these simpler structures. Precision and recall (Fig. 14b,c) follow similar trends, with consistently high values in the Gaussian and Sigmoid families, reflecting REX 's ability to capture relevant causal relationships effectively. Linear and Polynomial datasets, however, show greater variability, suggesting that REX may struggle with these structured relationships at higher dimensionalities.

The SHD and SID metrics (Fig. 14d,e) reinforce these findings, with low values in Gaussian and Sigmoid families, indicating REX 's accuracy in reconstructing causal structures and capturing correct causal directions. Higher values in Linear and Polynomial datasets, particularly as p increases, suggest occasional challenges in directionality and structure for these settings.

Training time (Fig. 14f) increases with p across all families, as expected. REX requires less time on Linear and Polynomial datasets compared to Gaussian and Sigmoid, likely due to the simpler relationships in these structures.

## 4.2.2 Comparison with other methods

Next, we compare REX against the compared methods (see also Appendix E for sample DAGs generated by each method). We observe that REX consistently shows strong performance, particularly in F1 score (Fig. 4a), where it achieves higher values relative to most other methods. Notably, REX exhibits a lower variance than the other methods, with values tightly concentrated around the median. This stability across metrics suggests that REX is less sensitive to variations in the data, resulting in more consistent performance. As suggested by the obtained F1 scores, REX stands out in aligning well with the true number of edges in the causal graph (Fig. 4b), indicating its effectiveness at avoiding both excessive edge predictions (false positives) and missing edges (false negatives). Compared to other methods, it exhibits a narrower interquartile range, implying lower variability in edge prediction across different datasets. This consistency suggests a robustness in REX 's performance, with fewer instances of significant deviations from the true edge count. The wider spread in edge difference for some of the compared methods observed in Figures 4 and 5 reflect their operation under default parameterizations (as stated in Section 4). These default settings might not be universally optimized for sparsity across all diverse synthetic data generation processes we tested, potentially leading them to predict denser graphs in certain scenarios compared to REX's intrinsic mechanisms for controlling graph density.

In terms of precision (Fig. 4c), REX performs comparably well with methods like CAM and FCI, demonstrating that it avoids excessive false positives while capturing meaningful causal links. This level of precision is advantageous for applications requiring high-confidence causal relationships. However, REX 's performance in recall (Fig. 4d) is moderate, aligning closely with other methods like PC and GES, indicating that while REX is effective in identifying relevant causal relationships, it may miss some causal connections compared to approaches with higher recall like CAM.

Overall, REX demonstrates strong structural accuracy and a balanced detection capability, particularly excelling in structural precision and F1 score, which are indicative of its robustness in maintaining the validity of discovered

DAG

1.0 H

60 -

0.8 -

50 -

40 -

SHD to true

Precision

0.6

30 -

0.4

20 -

0.2 -

10 -

0.0 C

1.0

0.8

0.6

0.4

0.2 -

0.0L

PC NOTEARS GES

PC

8

<!-- image -->

NOTEARS GES

LiNGAM

PC NOTEARS GES LINGAM FCI

(a)

LiNGAM

Figure 4: Comparative performance of REX against the benchmark methods PC, NOTEARS, GES, LiNGAM, FCI, and CAM across four key metrics: (a) F1 score; (b) edge difference, (c) precision; and (d) recall. Metrics are computed across all five considered synthetic families, with p = 10 features.

causal structures. However, some trade-offs in recall suggest areas where further improvements could be explored to enhance REX's sensitivity in capturing all causal connections, especially in more complex causal graphs.

Figure 5: Benchmark between REX and compared methods for (a) SHD to true DAG and (b) SID to true DAG metrics.

<!-- image -->

In addition, REX achieves competitive SHD values (Fig. 5a), with a median lower than most compared methods, indicating good alignment with the true DAG structure. The variability in SHD for REX is moderate, showing some datasets where performance is comparable to CAM and FCI, while outperforming methods like LiNGAM and GES. These results suggest that REX effectively balances its structural recovery, minimizing the number of missing and extra edges across diverse scenarios.

In terms of SID, REX shows favorable performance (Fig. 5b), with lower values than most methods and a narrower interquartile range compared to GES and NOTEARS. This indicates that REX better captures interventionally relevant causal directions, providing robust predictions under interventions. While CAM achieves comparable SID performance, REX demonstrates a more consistent behavior across datasets, highlighting its adaptability to various data structures. These results confirm REX's effectiveness in constructing causal graphs that align closely with both the observational and interventional ground truth.

## 5 Evaluation on biological and financial complex systems

## 5.1 Single-cell protein network data

To evaluate REX's performance on a real-world biological system and facilitate comparison within the broader causal discovery literature, we selected the Sachs et al. (2005) single-cell protein-signaling network dataset [31]. This dataset true

Diff. in edges to true DAG

20

80 -

10

40

- 10

-20

20-

T

Raf

Mek

Erk

Akt

PKC

PKA

P38

Jnk

(Pleg

PIP3

(PIP2)

Table 2: Summary metrics for all considered methods on the Sachs dataset.

| Method                            | Precision                                                                                                              | Recall                                                                                                                 | F1                                                                                                                     | SHD                                             | SID                                              |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------------------------------------------|
| PC FCI GES LiNGAM NOTEARS CAM REX | 0 . 462 ± 0 . 04 0 . 667 ± 0 . 05 0 . 200 ± 0 . 02 0 . 115 ± 0 . 01 0 . 212 ± 0 . 03 0 . 250 ± 0 . 05 0 . 952 ± 0 . 02 | 0 . 316 ± 0 . 05 0 . 421 ± 0 . 06 0 . 421 ± 0 . 03 0 . 158 ± 0 . 02 0 . 368 ± 0 . 04 0 . 600 ± 0 . 06 0 . 471 ± 0 . 01 | 0 . 376 ± 0 . 04 0 . 516 ± 0 . 05 0 . 273 ± 0 . 03 0 . 134 ± 0 . 02 0 . 270 ± 0 . 03 0 . 353 ± 0 . 05 0 . 629 ± 0 . 02 | 15 ± 2 13 ± 3 41 ± 1 37 ± 1 36 ± 2 39 ± 3 9 ± 1 | 53 ± 3 36 ± 4 50 ± 2 53 ± 1 50 ± 2 47 ± 4 39 ± 2 |

is a prominent benchmark due to several factors: it represents a complex, genuine biological system; it possesses a widely accepted, albeit simplified, ground truth causal network derived from experimental interventions, which is crucial for quantitative validation; and its inherent characteristics, such as potential non-linear interactions, make it a relevant testbed for methods like REX designed to handle such complexities. For our analysis, we utilize the cleaned version of the Sachs dataset, as provided in [32], and used in [33].

Following the structure described in Sachs et al. (2005) and Ramsey and Andrews (2018), we divide the 11 variables into two tiers: Plc, Pkc, Pka, PIP2, and PIP3 are categorized into tier 1, while the remaining variables (Raf, Mek, Erk, Akt, P38, and Jnk) belong to tier 2. Grouping variables in tiers is done to represent a causal dependency (temporal) order, such that variables at a given tier (e.g., 1 ) are not expected to receive directed edges from those in subsequent tiers (i.e., 2 , 3 , . . . ), but only from variables in the same or previous tiers. Hence, although feedback loops are suggested to exist within the network, they are thought to primarily occur among variables in tier 1, with no directed edges expected from tier 2 to tier 1. This prior information is added to REX and the compared methods prior to running them.

Fig. 6 shows the causal graph obtained by REX on the Sachs dataset. With a precision of 0 . 952 , a recall of 0 . 471 and a F1-score of 0 . 629 , REX is able to correctly recover important causal relationships from the data. The number of wrong edges or wrong directions are minimized, though the entire set of edges is not recovered. We note that to recover this DAG, REX was applied with exactly the same parametrization as in the synthetic datasets.

Figure 6: Plausible causal graph obtained by REX on the Sachs dataset. Green arrows correspond to correctly predicted edges, while gray arrows are true edges missed by REX.

<!-- image -->

We compare REX's performance with that of the compared methods on the Sachs dataset (Table 2). The results are based on five-runs with different seeds, all methods using parameter settings consistent with those applied to the synthetic datasets.

The results in Table 2 demonstrate that REX significantly outperforms other methods across most evaluation metrics on this dataset, achieving the highest score in precision and F1 score, indicating both high specificity and a strong balance between precision and recall. While its recall of 0 . 471 is moderate, it is still competitive and notably higher than that of most methods, except for CAM, which achieved a higher recall of 0 . 6 . In terms of structural accuracy, REX also reports the lowest SHD (median of 9 ), implying fewer structural errors compared to others, with FCI and CAM following at 13 and 39 , respectively. In addition, REX has a relatively good value for SID (median of 39 ), indicating also a noteworthy intervention-based structure interpretation capability. Overall, REX demonstrates strong precision and structural fidelity, supporting its robustness and reliability for causal discovery in this context (see Figure 7).

Raf

Akt

(PKC

PKC

LINGAM

FCI

Raf

Raf

Mek

P38

iJnk

Figure 7: DAGs obtained by the rest of comparable methods on the Sachs datasets.

<!-- image -->

While it is possible that further tuning could improve the performance of other methods, REX achieves top scores in precision and F1 without such adjustments, highlighting its robustness and reliability in causal discovery under default settings.

## 5.2 Financial decision-making data

The REX method was further evaluated on a synthetic financial dataset [34] designed to emulate real-world financial decision-making scenarios. This dataset captures the structural complexities typically encountered in financial credit risk assessments, including heterogeneous treatment effects, non-linear interactions, and positivity assumption violations. The dataset consists of 12 variables, with an observational sample of 12,000 entries. The variables include financial indicators such as credit history, debt levels, number of loans, and external debt exposure. The treatment variable represents the percentage of debt write-off assigned to customers, while the outcome variable reflects repayment probability. The known true causal DAG (see Fig. 8c) underlying the data generation process serves as ground truth for evaluating the effectiveness of different causal discovery methods (see [34] for details on the data generation process).

The causal discovery methods were applied ten times to different data subset samples (made of 1,000 samples), and performance was assessed based on precision, recall, F1-score, SHD and SID. REX was applied following the same pipeline as in previous experiments. As with the single-cell protein network data, a hyperparameter search (300 iterations) was conducted to optimize the regressors used for SHAP value computation. Each dataset was split into training and validation subsets, with the causal discovery process applied to the entire dataset after model training. Bootstrapping (10 iterations) was employed to enhance stability in the inferred causal structure, for the REX case. Two sample DAGs (best and worst) extracted from the bootstrapping process are show in Figure 8, in addition to the ground truth.

The summary metrics for all considered methods are provided in Table 3. The results indicate that REX consistently outperforms other methods in terms of structural accuracy. With an F1-score of 0 . 850 ± 0 . 05 , REX surpasses the second-best performing method, GES, which achieves an F1-score of 0 . 610 ± 0 . 06 . This improvement can be attributed to REX 's ability to effectively leverage machine learning explainability techniques for causal inference.

In terms of precision, REX achieves a score of 0 . 915 ± 0 . 12 , which is only slightly lower than the highest-performing method in this regard, FCI, which reaches 0 . 967 ± 0 . 07 . However, REX excels in recall with a significantly higher score of 0 . 799 ± 0 . 03 , substantially outperforming FCI and PC, both of which yield recall values of 0 . 375 ± 0 . 12 and 0 . 375 ± 0 . 10 , respectively. This suggests that REX is more effective at identifying true causal relationships without omitting key connections.

PKA

Pleg

X7

X8

x9

X4

X10

X6

X2

Table 3: Summary metrics for all considered methods on the Financial dataset.

X1

| Method             | Precision        | Recall                                                                                                         | F1                                                                                                                  | SHD                                                                                                      | SID                                                             |
|--------------------|------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| PC FCI GES CAM REX | 0 . 950 ± 0 . 11 | 0 . 375 ± 0 . 10 0 . 375 ± 0 . 12 0 . 575 ± 0 . 07 0 . 137 ± 0 . 08 0 . 562 ± 0 . 06 0 . 212 ± 0 . 12 ± 0 . 03 | 0 . 530 ± 0 . 11 0 . 533 ± 0 . 12 0 . 610 ± 0 . 06 0 . 170 ± 0 . 09 0 . 553 ± 0 . 03 0 . 171 ± 0 . 11 0 . 850 ± 0 . | 16 . 0 ± 0 . 0 10 . 2 ± 2 . 0 11 . 8 ± 2 . 5 21 . 0 ± 2 . 5 14 . 6 ± 2 . 2 35 . 4 ± 9 . 1 4 . 6 ± 1 . 82 | 13 . 0 ± 1 . 58 13 . 0 ± 1 . 87 34 . 6 ± 9 . 61 19 . 0 ± 0 . 00 |
|                    | 0 . 967 ± 0 . 07 |                                                                                                                |                                                                                                                     |                                                                                                          |                                                                 |
|                    | 0 . 659 ± 0 . 09 |                                                                                                                |                                                                                                                     |                                                                                                          |                                                                 |
| LiNGAM             | 0 . 227 ± 0 . 11 |                                                                                                                |                                                                                                                     |                                                                                                          |                                                                 |
| NOTEARS            | 0 . 558 ± 0 . 10 |                                                                                                                |                                                                                                                     |                                                                                                          | 9 . 2 ± 1 . 30                                                  |
|                    | 0 . 145 ± 0 . 11 |                                                                                                                |                                                                                                                     |                                                                                                          | 83 . 4 ± 15 . 04                                                |
|                    | 0 . 915 ± 0 . 12 | 0 . 799                                                                                                        | 05                                                                                                                  |                                                                                                          | 6 . 4 ± 1 . 52                                                  |

Furthermore, REX achieves the lowest Structural Hamming Distance (SHD) and Structural Intervention Distance (SID) among all tested methods, with values of 4 . 6 ± 1 . 82 and 6 . 4 ± 1 . 52 , respectively. These results highlight the method's ability to recover the true causal structure with minimal errors and to maintain robustness in interventionbased causal queries.

Compared to constraint-based methods such as PC and FCI, REX shows superior performance in terms of both recall and SHD. The performance of NOTEARS and LiNGAM remains significantly lower, with F1-scores of 0 . 553 ± 0 . 03 and 0 . 170 ± 0 . 09 , respectively. The CAM method performs the worst, with an F1-score of 0 . 171 ± 0 . 11 , an SHD of 35 . 4 ± 9 . 1 , and the highest SID value of 83 . 4 ± 15 . 04 , indicating that it struggles to correctly infer causal relationships in this domain.

Figure 8: Two samples of the DAGs inferred by REX on the financial dataset: (a) the DAG with the best inferred structure, and (b) the DAG with the worst inferred structure, as well as (c) the true DAG.

<!-- image -->

## 6 Practical considerations and future work

During the development of REX, we addressed several considerations and decisions, while also identifying potential areas for future improvement. Next, we discuss in detail topics such as opting for SHAP values over other explainability methods, the challenges of multicollinearity, and the potential for experimenting with various machine learning regressors. These factors play a crucial role in shaping the method's performance and open up opportunities for further research to refine and enhance the proposed approach.

## 6.1 Selection of Shapley values for feature importance assessment

Afundamental design decision in REX involves selecting a reliable measure of feature contributions to predict a target variable, which directly guides parent selection (Algorithm 2). Among various methods, Shapley values [5] were chosen due to their theoretical rigor and practical advantages in identifying stable and meaningful candidate parents.

Alternative methods have specific limitations. Model-based metrics, such as Gini importance, may bias results toward high-cardinality features or produce misleading attributions when features are correlated [35]. Permutation impor-

XS

X4

tance, although intuitive, struggles similarly in correlated scenarios, either underestimating collective feature importance or producing unrealistic permutations [36, 37]. Local explanation methods like LIME [38] require extensive aggregation for global insights and are sensitive to the choice of local neighborhoods and surrogate models [39].

Conversely, Shapley values as implemented through SHAP [4] provide several distinct benefits for REX: (i) they uniquely satisfy key axiomatic properties (efficiency, symmetry, and additivity) that enable principled attribution [5, 6]; (ii) they inherently accommodate feature interactions and redundancies by averaging contributions across all possible feature coalitions, ensuring stable attribution even in scenarios of high multicollinearity, as demonstrated empirically (Appendix A); (iii) they exhibit a theoretical and empirically validated link to conditional dependence (Section 2.2, Appendix A), aligning closely with the requirements of causal discovery under faithfulness; and (iv) they benefit from robust, model-specific implementations (e.g., TreeExplainer for GBTs, GradientExplainer for DFNs), improving computational efficiency and reliability [4, 7].

Due to their theoretical soundness, robustness to feature correlations, and clear interpretability regarding conditional dependence, Shapley values are particularly well-suited to identifying potential causal parents in REX.

## 6.2 Multicollinearity considerations

A significant challenge when using Shapley values for model interpretation, particularly in the context of establishing causal relationships among features in a dataset, comes from situations where some of these features are highly correlated [40]. An scenario with multicollinearity makes difficult to isolate the individual effect of each feature on the model's predictions. In such cases, Shapley values may be argued to distribute the importance across correlated features, potentially obscuring the true contribution of each individual feature to the model's predictions. This distribution of importance can be understood as source of ambiguity, as it could become challenging to discern whether the impact on the prediction is due to the intrinsic value of a feature or its correlation with other features.

The implementation of Shapley values in SHAP by [4] includes model-agnostic explainers like GradientExplainer or TreeExplainer as a potential solution [7] to mitigate the effects of multicollinearity.

Specifically, these 'interventional' variants of SHAP explainers, as discussed by Chen et al. [7] in contrast to 'observational' approaches that use conditional expectations, aim to estimate a feature's effect by breaking its correlations with other features not currently in the evaluated coalition. When computing the expected model output for a subset of features S , features X k / ∈ S are effectively marginalized out by sampling from their empirical distribution (often a background dataset) independently of the features in S , rather than conditioning on them.

For instance, TreeExplainer , when applied to tree-based models like GBTs, can implement this through pathdependent feature perturbation algorithms that effectively average over the marginal distribution of 'absent' features encountered along decision paths. GradientExplainer , used for DFNs, approximates SHAP values and typically relies on a background dataset to represent features not in the current coalition; if these are sampled independently, its behavior aligns with an interventional perspective.

Unlike other explainers that might treat features collectively or consider their joint distributions, interventional explainers operate under an interventional framework where the impact of each feature on the model's output is computed independently, assuming other features remain constant. This approach aligns with the concept of partial derivatives, focusing on the sensitivity of the output to changes in each feature, treated in isolation. By doing so, interventional explainers effectively disentangle the intertwined influences of correlated features, making it easier to interpret the model in terms of causal relationships. This is particularly useful in deep learning models where the intricate interactions of features can be complex and opaque.

In the context of REX, the adoption of model-agnostic explainers is intricately linked to the type of regressor being employed. When utilizing DFN, GradientExplainer is chosen, tailored to capture the complex, layered interactions and non-linear mappings characteristic of these models. Conversely, when employing Gradient Boosting Trees, TreeExplainer is utilized as it is more suited to dissecting the sequential, decision-tree-based learning process intrinsic to these algorithms. This adaptability in the choice of explainers is crucial, as it ensures that the explanations generated are not only robust but also appropriately aligned with the underlying mechanics of the chosen regression model.

Indeed, our synthetic experiments (detailed in Appendix A) provide empirical evidence for this robustness. In scenarios with highly collinear predictor variables, the Shapley values computed remained stable across repeated repeated trials ( σ/µ &lt; 5% for ϕ across 50 replicates) and provided an intuitive partitioning of importance between the correlated features, a task that can be challenging for other interpretation methods or even for establishing stable DAG structures using iterative CI-tests.

## 6.3 Stability of the discovered DAG

The stability of the causal graph discovered by REX is primarily addressed through the bootstrapping mechanism (Algorithm 1) and the subsequent frequency-based thresholding using the τ parameter (Section 3.2.3). An edge is incorporated into the initial graph for each regressor type ( G DFN or G GBT ) only if its selection frequency across T bootstrap iterations surpasses τ . This procedure ensures that the foundational structure is based on relationships that consistently emerge despite resampling of the data, thereby filtering out less robust, potentially spurious connections.

While REX does not compute a separate, global quantitative metric for DAG stability over multiple identical full runs, the design inherently promotes the stability of the identified edges. The consistency and relatively low variance observed in performance metrics across diverse synthetic datasets (e.g., F1 scores in Figure 14) further provide indirect evidence of the method's ability to produce reliable DAGs. The overall stability of the final G REX also depends on the robustness of the hyperparameter optimization, the ANM-based edge orientation (Section 3.3), and the cycle resolution mechanism (Section 3.4).

## 6.4 Future work

As discussed in Section 3.1, a key direction for future research is the investigation of alternative regressors beyond the deep feed-forward neural networks (DFN) and gradient boosting trees (GBT) used in this study. Both models support SHAP, enabling interpretability of their predictions. However, other machine learning models that also facilitate SHAP explainability could be integrated into the REX framework. For instance, Lasso-based methods, such as Group Lasso [41], commonly used in causal discovery approaches like CAM [15], offer inherent sparsity. This property may enhance interpretability by reducing irrelevant connections, potentially yielding more interpretable causal graphs. SHAP compatibility with these methods positions them as promising candidates for future integration into REX.

Further exploration could also include models like Elastic Net regression [42], which combines the sparsity of Lasso with the regularization of Ridge regression, possibly improving feature selection while retaining SHAP explainability. Tree-based ensemble methods such as CatBoost and LightGBM, which have native SHAP support, also present an attractive alternative to GBTs, potentially offering computational advantages on large datasets. Expanding REX to incorporate these alternatives could enhance its adaptability and performance across various data structures and application contexts.

Another research area is its application to datasets with distributional shifts, where the data-generating process may vary across different populations or over time. Recent work [43] has addressed this challenge by identifying features that remain causally invariant across changing distributions. Integrating such principles into the REX could enhance its applicability to real-world datasets where the assumption of consistent distributions does not hold, improving its robustness and generalization capability.

Exploring the behavior of the different SHAP values of Equation (4) in each of the S subsets may provide more information on how the variables are grouped together within a larger causal structure. Relating this future analysis to techniques that give us information about how they are grouped may help us better discern relationships between nodes so that we can distinguish between parents and ancestors.

Additionally, efforts could be made to refine the model's scalability and enhance its ability to handle larger datasets and higher-dimensional graphs.

While our SHAP discrepancy measure (Section 3.4) offers a novel, explainability-guided heuristic for resolving cycles that may arise from the union of DAGs, a comparative evaluation against other cycle-breaking strategies could be pursued. Future investigations might explore graph-theoretic approaches, such as heuristics for the Feedback Arc Set problem (e.g., [44]), or methods adapted from score-based Bayesian network learning to further refine this crucial step.

## 7 Conclusion

In this paper, we introduced REX, a novel causal discovery method that leverages machine learning and Shapleybased explainability to address key challenges in uncovering causal structures, particularly in complex and nonlinear datasets. The approach combines interventional explainers to compute Shapley values with a bootstrapping mechanism, followed by an Additive Noise Model (ANM) for edge orientation. A novel SHAP discrepancy measure further refines the causal graph by quantifying the agreement between Shapley values and their associated features, filtering out spurious relationships.

Our experiments on five families of synthetic datasets, the Sachs single-cell protein-signaling dataset [31], and the financial decision-making data [34] illustrate the effectiveness and robustness of REX. Not only does it recover causal relationships with high precision and low orientation error, but it also compares favorably with existing state-of-the-art methods. By providing interpretable insights into the underlying causal dynamics, REX enables contrasting inferred causal links with domain expertise.

Wehighlight three major strengths of REX. First, its results are comparable to leading causal discovery methods across varying nonlinearities. Second, the use of Shapley values brings a valuable layer of interpretability, fostering trust and supporting evidence-based decisions. Third, the integration of data-driven modeling, explainability, and a bootstrapping mechanism provides robustness against spurious edges and orientation errors. However, the computational cost of calculating Shapley values in high-dimensional settings remains a key limitation, prompting future work on scalable, approximate solutions and more efficient architectures. Ongoing advances in GPU computing or distributed ML can significantly help to mitigate the scalability limitation in future implementations.

Looking ahead, we see potential for extending REX in multiple directions (see Section 6.4). Investigating its performance under distributional shifts, employing ensemble SHAP strategies to mitigate collinearity issues, and adopting computational frameworks that handle large or streaming datasets will further broaden its applicability. Moreover, integrating domain-specific priors and exploring advanced regressors such as Lasso-based or alternative tree-based ensemble implementations could further enhance performance and interpretability.

In fields like healthcare, finance, and economics, where reliable causal insights are crucial, REX 's interpretability is especially beneficial. We encourage the community to adopt and adapt REX for a wide range of applications, and have made all code and datasets publicly available at https://github.com/renero/causalgraph . Through such collaborations, we anticipate continued progress in interpretable causal discovery, ultimately advancing our ability to unravel complex causal relationships across diverse domains.

## Acknowledgements

This work has been partially supported by a Ramon y Cajal contract (RYC2019-028578-I), a Gipuzkoa Fellows grant (2022-FELL-000003-01), and a grant from the Spanish MCIN (PID2021-126718OA-I00).

## References

- [1] J. Pearl, Causality: Models, Reasoning, and Inference, 2nd Edition, Cambridge Univ. Press, 2009.
- [2] P. Spirtes, C. Glymour, R. Scheines, Causation, Prediction, and Search, 2nd Edition, MIT Press, 2000.
- [3] J. Peters, D. Janzing, B. Sch¨ olkopf, Elements of Causal Inference: Foundations and Learning Algorithms, Adaptive Computation and Machine Learning, MIT Press, Cambridge, MA, 2017.
- [4] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Advances in Neural Information Processing Systems, 2017, pp. 4765-4774.
- [5] L. S. Shapley, A value for n-person games, in: H. W. Kuhn, A. W. Tucker (Eds.), Contributions to the Theory of Games, Vol. 2, Princeton University Press, 1953, pp. 307-317.
- [6] E. Winter, The shapley value, in: R. J. Aumann, S. Hart (Eds.), Handbook of Game Theory with Economic Applications, Vol. 3, Elsevier, 2002, Ch. 54, pp. 2025-2054.
- [7] H. Chen, J. D. Janizek, S. Lundberg, S.-I. Lee, True to the Model or True to the Data?, arXiv:2006.16234 [cs, stat] version: 1 (Jun. 2020). doi:10.48550/arXiv.2006.16234.
- [8] J. Pearl, D. Mackenzie, The Book of Why: The New Science of Cause and Effect, Basic Books, 2018.
- [9] J. Chen, L. Song, M. J. Wainwright, Explaining decisions with causal shapley values, Journal of Machine Learning Research 23 (77) (2022) 1-45.
- [10] A. Datta, S. Sen, Y. Zick, Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems, in: 2016 IEEE Symposium on Security and Privacy (SP), IEEE, 2016, pp. 598-617.
- [11] M. Sundararajan, A. Najmi, The many shapley values for model explanation, in: International Conference on Machine Learning, PMLR, 2019, pp. 9269-9278.
- [12] P. Spirtes, C. Meek, T. Richardson, Causal discovery in the presence of latent variables and selection bias, in: C. Glymour, G. F. Cooper (Eds.), Computation, Causation, and Discovery, MIT Press, Cambridge, MA, 1999, pp. 211-252.

- [13] D. M. Chickering, Optimal structure identification with greedy search, Journal of machine learning research 3 (Nov) (2002) 507-554.
- [14] S. Shimizu, P. O. Hoyer, A. Hyv¨ arinen, A. Kerminen, A linear non-gaussian acyclic model for causal discovery, Journal of Machine Learning Research 7 (Oct) (2006) 2003-2030.
- [15] P. B¨ uhlmann, J. Peters, J. Ernest, CAM: Causal additive models, high-dimensional order search and penalized regression, The Annals of Statistics 42 (6) (2014) 2526-2556.
- [16] I. Tsamardinos, L. E. Brown, C. F. Aliferis, The max-min hill-climbing bayesian network structure learning algorithm, Machine Learning 65 (1) (2006) 31-78.
- [17] X. Zheng, B. Aragam, P. K. Ravikumar, E. P. Xing, Dags with no tears: Continuous optimization for structure learning, in: Advances in neural information processing systems, Vol. 31, 2018.
- [18] D. Kalainathan, O. Goudet, SAM: Structural agnostic model, causal discovery and penalized adversarial learning, in: Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR, 2019, pp. 10-19.
- [19] Y. Yu, J. Chen, T. Gao, M. Yu, DAG-GNN: DAG structure learning with graph neural networks, in: International Conference on Machine Learning, PMLR, 2019, pp. 7154-7163.
- [20] T. Heskes, E. Sijben, I. G. Bucur, T. Claassen, Causal shapley values: Exploiting causal knowledge to explain individual predictions of complex models, Advances in neural information processing systems 33 (2020) 47784789.
- [21] C. Min, G. Wen, L. Gou, X. Li, Z. Yang, Interpretability and causal discovery of the machine learning models to predict the production of CBM wells after hydraulic fracturing, Energy 283 (2023) 129211.
- [22] G. Xu, T. Duong, Q. Li, S. Liu, X. Wang, Causality learning: A new perspective for interpretable machine learning, IEEE Intelligent Informatics Bulletin (1) (2020) 24-33.
- [23] S. Ma, R. Tourani, Predictive and causal implications of using shapley value for model interpretation, in: Proceedings of the 2020 KDD workshop on causal discovery, PMLR, 2020, pp. 23-38.
- [24] K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks are universal approximators, Neural networks 2 (5) (1989) 359-366.
- [25] J. H. Friedman, Greedy function approximation: A gradient boosting machine, Annals of statistics 29 (5) (2001) 1189-1232.
- [26] D. Kalainathan, Generative Neural Networks to infer Causal Mechanisms : algorithms and applications, Theses, Universit´ e Paris Saclay (COmUE) (Dec. 2019).
- [27] J. Bergstra, R. Bardenet, Y. Bengio, B. K´ egl, Algorithms for hyper-parameter optimization, in: Advances in neural information processing systems, Vol. 24, 2011, pp. 2546-2554.
- [28] A. Gretton, K. Fukumizu, C. Teo, L. Song, B. Sch¨ olkopf, A. Smola, A kernel statistical test of independence, Advances in neural information processing systems 20 (2007) 585-592.
- [29] J. Peters, J. M. Mooij, D. Janzing, B. Sch¨ olkopf, Causal discovery with continuous additive noise models, J. Mach. Learn. Res. 15 (1) (2014) 2009-2053.
- [30] J. Peters, P. B¨ uhlmann, Structural Intervention Distance for Evaluating Causal Graphs, Neural Computation 27 (3) (2015) 771-799.
- [31] K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger, G. P. Nolan, Causal protein-signaling networks derived from multiparameter single-cell data, Science 308 (5721) (2005) 523-529.
- [32] J. D. Ramsey, B. Andrews, FASK with interventional knowledge recovers edges from the sachs model, CoRR abs/1805.03108 (2018). arXiv:1805.03108.
- [33] T.-H. Chang, Z. Guo, D. Malinsky, Post-selection inference for causal effects after causal discovery (2024). arXiv:2405.06763.
- [34] J. Moral Hern´ andez, C. Higuera-Caba˜ nes, A. Ibra´ ın, An end-to-end pipeline for causal ML with continuous treatments: An application to financial decision making, in: Proc. 3rd Workshop on Causal Inference and Machine Learning in Practice (KDD 2025), Vol. 21, Association for Computing Machinery, 2025, p. 6.
- [35] C. Strobl, A.-L. Boulesteix, A. Zeileis, T. Hothorn, Bias in random forest variable importance measures: Illustrations, sources and a solution, BMC Bioinformatics 8 (1) (2007) 25. doi:10.1186/1471-2105-8-25.
- [36] G. Hooker, L. Mentch, Please stop permuting features: An explanation and alternatives, in: Proceedings of the ICML Workshop on Human Interpretability in Machine Learning (WHI), 2019, pp. -.

- [37] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, 2nd Edition, Lulu.com, 2022.
- [38] M. T. Ribeiro, S. Singh, C. Guestrin, 'Why Should I Trust You?': Explaining the Predictions of Any Classifier, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2016, pp. 1135-1144.
- [39] D. Alvarez-Melis, T. S. Jaakkola, On the robustness of interpretability methods, in: Proceedings of the ICML Workshop on Human Interpretability in Machine Learning (WHI), 2018, pp. -.
- [40] D. Janzing, L. Minorics, P. Bloebaum, Feature relevance quantification in explainable AI: A causal problem, in: Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, 2020, pp. 2907-2916, iSSN: 2640-3498.
- [41] M. Yuan, Y. Lin, Model selection and estimation in regression with grouped variables, Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68 (1) (2006) 49-67.
- [42] H. Zou, T. Hastie, Regularization and variable selection via the elastic net, Journal of the Royal Statistical Society: Series B (Statistical Methodology) 67 (2) (2005) 301-320.
- [43] Y. Wang, K. Yu, G. Xiang, F. Cao, J. Liang, Discovering causally invariant features for out-of-distribution generalization, Pattern Recognition 150 (2024) 110338.
- [44] P. Eades, X. Lin, W. F. Smyth, A fast and effective heuristic for the feedback arc set problem, Information Processing Letters 47 (6) (1993) 319-323.

## Appendices

## A Experimental validation of Shapley values versus statistical dependence

This appendix details the design, implementation, and results of a series of synthetic data experiments specifically conducted to elucidate the relationship between Shapley feature importance scores (as utilized in REX) and traditional measures of statistical dependence (marginal and conditional). These experiments complement the broader performance benchmarks of REX presented in Section 4 by providing a focused analysis on the behavior of Shapley values in canonical causal structures known to pose challenges for causal discovery algorithms that rely on CI tests. The goal is to empirically validate the theoretical foundations discussed in Section 2.2 and address common questions regarding the interpretation of Shapley values in causal contexts, particularly concerning confounders, mediators, colliders, and multicollinearity.

## A.1 Experimental design and methodology

The experiment was designed to systematically compare Shapley values with statistical dependence measures across various controlled causal scenarios. The key steps are outlined below:

Four fundamental causal structures involving three variables ( X , Y , Z , where Y is the target variable for prediction) or two highly correlated parent variables ( X 1 , X 2 ) were generated (see Table 4).

Table 4: Causal structures considered in the analysis

| Structure                 | Description                                                                                                                                                                                          |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Confounder                | Z → X , Z → Y . This structure induces a spurious correlation between X and Y due to the common cause Z .                                                                                            |
| Chain (Mediator) Collider | X → Z → Y . Here, the influence of X on Y is mediated by Z . X → Z ← Y . In this structure, X and Y are marginally independent but become conditionally dependent given their common effect Z .      |
| Collinear Parents         | X 1 → Y , X 2 → Y , with X 1 ≈ X 2 . This scenario tests behavior under high multicollinearity between parent nodes. For this structure, Y is the target, and X 1 ,X 2 are the features of interest. |

For the Confounder, Chain, and Collider structures, the target variable for prediction and Shapley value computation is always Y . The features of interest for which Shapley values and dependence measures are reported are X and Z .

Data was generated from linear Gaussian structural equations with coefficients set to 1.0, except in the collinear case, where a small perturbation induced multicollinearity. Each dataset comprised n = 5000 samples with noise terms drawn from a N (0 , 0 . 10 2 ) distribution. The process was repeated R = 50 times per structure to ensure robust estimation of means and standard deviations for all metrics.

A gradient-boosted tree model (XGBoost) was fitted to predict the target variable Y from the relevant features ( X,Z or X 1 , X 2 , depending on the causal structure). The model hyperparameters were fixed at 200 trees, maximum depth of 4, and learning rate of 0.05 across all experiments to isolate the effects of data structure, despite the linearity of the data generation process.

Shapley values were computed using TreeExplainer from the SHAP library [4] to quantify feature importance in predicting Y . For each of the R = 50 replicates, mean absolute Shapley values ( ϕ ) per feature were calculated on a held-out test set of 1,000 points.

To compare Shapley values with traditional statistical measures, marginal and conditional dependence metrics were computed between each relevant feature and the target Y . Marginal dependence was quantified using the Pearson correlation coefficient ρ ( Feature , Y ) . Conditional dependence was assessed via partial Pearson correlation coefficients, e.g., ρ ( X,Y | Z ) for structures involving additional non-target features. Statistical significance of conditional dependence was evaluated through p-values calculated using Fisher's Z-transform.

## A.2 Aggregated results

The mean and standard deviation (s.d.) of the computed metrics across the R = 50 replications are presented in Table 5. Columns are: ρ (mean marginal Pearson correlation with Y ), σ ρ (s.d. of marginal Pearson correlation), ρ p (mean partial correlation with Y , conditioned as described above), σ ρ p (s.d. of partial correlation), p (mean p -value for the

partial correlation), σ p (s.d. of p -value), ϕ (mean absolute Shapley value for predicting Y ), and σ ϕ (s.d. of mean absolute Shapley value).

Table 5: Aggregated results (mean ± s.d. over 50 replications) comparing Shapley values with marginal and conditional dependence measures for different causal structures. The target variable is Y . For Confounder, Chain, and Collider, features are X and Z . For Collinear, features are X 1 and X 2 .

| Structure   | Feature   |     ρ |   σ ρ |    ρ p |   σ ρ p |   p -val |   σ p -val |   | ϕ | |   σ | ϕ | |
|-------------|-----------|-------|-------|--------|---------|----------|------------|---------|-----------|
| Confounder  | X         | 0.99  | 0     |  0     |   0.013 |    0.514 |      0.277 |   0.356 |     0.009 |
| Confounder  | Z         | 0.995 | 0     |  0.706 |   0.007 |    0     |      0     |   0.447 |     0.011 |
| Chain       | X         | 0.99  | 0     | -0.001 |   0.011 |    0.625 |      0.293 |   0.426 |     0.009 |
| Chain       | Z         | 0.995 | 0     |  0.707 |   0.006 |    0     |      0     |   0.372 |     0.008 |
| Collider    | X         | 0     | 0.014 | -0.99  |   0     |    0     |      0     |   0.556 |     0.018 |
| Collider    | Z         | 0.705 | 0.008 |  0.995 |   0     |    0     |      0     |   0.873 |     0.023 |
| Collinear   | X 1       | 0.995 | 0     |  0.097 |   0.015 |    0     |      0     |   0.435 |     0.012 |
| Collinear   | X 2       | 0.995 | 0     |  0.102 |   0.015 |    0     |      0     |   0.361 |     0.01  |

## A.3 Interpretation and discussion of results

The results from Table 5 provide quantitative evidence supporting the theoretical connection between Shapley values and conditional dependence, as articulated in Section 2.2 (particularly Equation (11)).

## A.3.1 Divergence from single-Set CI tests

The experiments show that Shapley values are not mere proxies for marginal correlation ( ρ ) nor for conditional independence tests based on a single, specific conditioning set (like the partial correlation ρ p and its p -value).

- Confounder ( Z → X, Z → Y ): For feature X , the marginal correlation ρ ( X,Y ) is very high (mean 0.990) due to the confounder Z . However, when conditioned on Z , the partial correlation ρ p ( X,Y | Z ) is effectively zero (mean 0.000), with a high mean p -value (0.514), indicating X ⊥ ⊥ Y | Z . Despite this conditional independence given Z , the Shapley value for X (mean | ϕ | X = 0 . 356 ) is substantial. This occurs because, according to Equation (10), Shapley values consider all coalitions S . In coalitions where Z / ∈ S , X still carries predictive information about Y (as X is a proxy for Z 's influence). The non-zero Shapley value reflects this average utility across all contexts. Feature Z , the true direct cause of Y (in this simplified model), has a high partial correlation (mean ρ p ( Z, Y | X ) = 0 . 706 , p -value ≈ 0 ) and the highest Shapley value (mean | ϕ | Z = 0 . 447 ).
- Chain ( X → Z → Y ): Similarly for feature X , ρ ( X,Y ) is high (mean 0.990). When conditioned on the mediator Z , ρ p ( X,Y | Z ) is effectively zero (mean -0.001), with a high mean p -value (0.625), indicating X ⊥ ⊥ Y | Z . Yet, | ϕ | X is substantial (mean 0.426). This is because in coalitions S where Z / ∈ S , the path X → Z → Y is 'open' in terms of information flow from X to Y for the predictive model, making X valuable. The direct parent Z has a high partial correlation ρ p ( Z, Y | X ) (mean 0.707, p -value ≈ 0 ) and a significant Shapley value (mean | ϕ | Z = 0 . 372 ).

These cases demonstrate that if a standard CI test (like Fisher-Z on partial correlation) indicates conditional independence given a specific variable (e.g., Z ), the Shapley value for X can still be high. This is because the Shapley value integrates over all possible conditioning sets (coalitions), capturing the feature's predictive contribution in contexts where the specific variable Z is not part of the conditioning set. This aligns with the theoretical premise that Shapley values offer a more global assessment of feature importance.

## A.3.2 Collider structure and conditional dependence

- Collider ( X → Z ← Y ): Feature X is marginally independent of Y (mean ρ ( X,Y ) = 0 . 000 ). However, when conditioned on the collider Z , X and Y become strongly dependent (mean ρ p ( X,Y | Z ) = -0 . 990 , p -value ≈ 0 ). The Shapley value for X (mean | ϕ | X = 0 . 556 ) is high, reflecting this strong conditional dependence that emerges when Z is in the coalition. Feature Z itself, being a child of both X and Y (and thus highly informative about Y when X is known, and vice-versa for the model), has the highest Shapley value (mean | ϕ | Z = 0 . 873 ). This scenario illustrates that Shapley values correctly identify the importance of variables whose relevance is only revealed upon conditioning.

## A.3.3 Robustness and stability in high correlation scenarios

The Collinear Parents case addresses concerns about behavior with highly correlated variables.

- Collinear Parents ( X 1 ≈ X 2 → Y ): Both X 1 and X 2 are highly correlated with Y marginally (mean ρ ≈ 0 . 995 ). The partial correlations ρ p ( X 1 , Y | X 2 ) and ρ p ( X 2 , Y | X 1 ) are low (means 0.097 and 0.102 respectively), indicating that once one collinear parent is known, the other offers little unique additional linear information. Despite these low partial correlations, their p -values are effectively zero due to the large sample size, indicating statistical significance. Crucially, the Shapley values for X 1 (mean | ϕ | X 1 = 0 . 435 ) and X 2 (mean | ϕ | X 2 = 0 . 361 ) are substantial and intuitively distribute the importance between the two highly correlated features. The standard deviations for these Shapley values are very small ( 0 . 012 and 0 . 010 respectively), highlighting the stability of Shapley attributions in this scenario. This contrasts with the potential instability or difficulty in interpreting individual coefficients in a linear regression model facing severe multicollinearity. This empirical finding supports the discussion in Section 6.2 regarding the robustness of interventional SHAP explainers.

## A.3.4 Confirmation of theoretical connection (Equation (10))

The results presented along this appendix confirm the theoretical connection established in 2.2, which can equivalently represented as:

<!-- formula-not-decoded -->

- When a feature X j is strongly and consistently conditionally dependent on Y across many coalitions (high P weighted ( X j ̸ ⊥ ⊥ Y | S ) and/or high average marginal contribution ∆ j ), its Shapley value is high (e.g., Z in Confounder/Chain, X and Z in Collider, X 1 , X 2 in Collinear).
- When a feature X j 's conditional dependence on Y is only evident in specific contexts (e.g., X in Confounder/Chain is dependent on Y mainly when Z is not in the coalition), its Shapley value is moderated but still non-zero, reflecting this averaged importance.
- The low standard deviations of the mean Shapley values across 50 runs (typically &lt; 5% of the mean, often much lower) underscore the stability of this importance measure under the experimental conditions.

## A.4 Conclusion of experimental validation

This synthetic study empirically illustrates that Shapley values utilized in REX capture aspects of feature importance beyond those measured by standard statistical dependence tests. Specifically, Shapley values demonstrate increased robustness under multicollinearity, reveal important distinctions from conventional measures, and substantiate their theoretical relationship to conditional dependence in aggregate. Collectively, these results support the integration of Shapley values into causal discovery frameworks such as REX.

## B Parents selection example

Fig. 9 illustrates how the parents selection described in Algorithm 2 mechanism works. On panel (a) there is a significant shift between the features that are more influential ( V 1 and V 2 ) and the rest. On panel ( b ), such shift is not so clear, though the first group of three ( V 0 , V 2 and V 4 ) present a more pronounced influence than the rest. On top of each plot, the variables selected by Algorithm 2 are written to the right of the target variable, as parents ( ← ) of the target.

While Algorithm 2 is designed for fully automated parent selection, we acknowledge the theoretical possibility of edge cases where its standard DBSCAN-based clustering (detailed in § 3.2.2) might discard features that a user, upon closer inspection of SHAP value distributions (as exemplified in Figure 9), could still deem influential, particularly if SHAP values are very closely clustered or if the automated separation appears too conservative for a specific application. For instance, in a scenario like that depicted in Figure 9b, if the automated process were to hypothetically select only variable V 0 instead of all three visually distinct top variables ( V 0 , V 2 , V 4 ) due to such an edge case in the clustering, a user might seek further options.

For such specific situations, where an application user might desire finer-grained control or wish to incorporate domain expertise, the concept of 'manual supervision' is noted here as a practical recourse. This could involve, for example, a domain expert directly reviewing the SHAP value distributions and, based on their judgment, potentially adjusting the

V1

V2

V6

V5

V7

VO

V4

V9

V8

V6

V8

0.02

V3 + V1,V2

V3

0.04

V4

0.06

V5

V9

(a)

0.08

VO

V2

V4

V1

V9

V5

V4/фv3

V4/фv6

Figure 9: Example of SHAP values (x-axis) for: (a) endogenous variable V 3 ; and (b) V 8 .

<!-- image -->

VA

V5

automatically selected parent set. As an alternative built-in option for users seeking a more inclusive selection without direct manual intervention, an additional parametrization of Algorithm 2 allows it to be run in a greedy mode. This mode can be configured to select all features whose SHAP values exceed a user-defined percentile of the distribution, rather than relying solely on the primary DBSCAN clustering outcome. Furthermore, as an optional heuristic, this greedy tuning can be automatically activated if the standard feature selection process results in a causal graph (see § 3.2.3) with a number of edges significantly below what might be expected for a densely connected system (e.g., compared to p ( p -1) 2 ).

It is crucial to emphasize, however, that the performance and results reported for R e X throughout this paper (Section 4) are based on its fully automated execution, without any such manual supervision or the use of the 'greedy mode' or other special parametrizations of Algorithm 2. The discussion of these alternative operational modes is included in this appendix purely as a practical consideration for users who might apply R e X to new, potentially unique, or particularly challenging datasets where such adjustments could be exceptionally beneficial for fine-tuning the discovery process.

## C SHAP discrepancy example

Figure 10: (a) Ground truth DAG for a system of 10 variables ( V 0 to V 9 ). Highlighted in green are nodes V 3 , V 4 , and V 6 , which are either direct or indirect causes or effects of one another. (b)-(e) SHAP discrepancy analysis for variable pairs in the graph. As observed, discrepancy values between pairs of V 3 , V 4 , and V 6 are low ((b), (c) and (d)), in contrast to the one obtained when analyzing the pair V 4 , V 5 (e).

<!-- image -->

V8 + VO, V2, V4

X;vs.(X:lФj).

55 :0.08

X;vs.[Xiløi), 5%:0.15

1.0,

0.8-

0.6

0.4-

0.2-

0.0

F1 by regressor and combination

DFN

GBT

Union Intersection

Figure 11: REX's F1 scores for all synthetic datasets with 10, 15, 20 and 25 variables, comparing the DAGs obtained by a single regressor (DFN or GBT) to those obtained by combining them through union ( ∪ ) and intersection ( ∩ ) operations.

<!-- image -->

Fig. 10 illustrates the role of SHAP values and the discrepancy between two potential causes or parent features X j , and one effect or dependent feature X i .

In (b) we can see discrepancy values for V 4 versus V 3 ( is V 3 a direct cause of V 4 ? ), showing a low discrepancy value δ ( t ) ij = 0 . 08 , consistent with the fact that V 4 is a direct child of V 3 . (c) shows that V 6 could also be considered a direct cause of V 4 . However, in (d) we can also check that V 4 could also be considered a direct cause of V 6 . In (e) we can check how V 5 present a very high discrepancy (1.99), indicating that V 5 should be discarded among the potential parents of V 4 .

The low SHAP discrepancy values for causally related variables support the ability of SHAP discrepancy to distinguish causal dependencies, whereas the high discrepancy for non-causal relationships ( V 4 → V 5 ) demonstrates its effectiveness in filtering out irrelevant connections.

## D Performance comparison between single and combinations of regressors

REX employs two complementary regressors-Deep Feed-forward Networks (DFN) and Gradient Boosting Trees (GBT)-to generate individual causal graph hypotheses from the data (see Section 3.1). The final DAG, G REX, is then constructed by taking the union ( ∪ ) of the DAGs derived from each regressor, G DFN and G GBT .

The rationale for choosing the union operation is to maximize the discovery of true causal relationships by leveraging the distinct modeling capabilities of DFNs (effective at capturing complex, non-linear functions) and GBTs (robust for tabular data and possessing different inductive biases). Since either type of regressor might uniquely identify certain true edges that the other misses due to its specific learning mechanism, the union aims to create a more comprehensive initial graph, prioritizing the recall of potentially true edges at this stage of the pipeline. It is important to note that each individual graph, G DFN and G GBT , is already the result of a robust bootstrapping process (Algorithm 1) designed to stabilize edge selection and reduce spurious connections before the union.

We also evaluated the alternative of using the intersection ( ∩ ) of G DFN and G GBT . The intersection is a more conservative approach, as it retains only those edges agreed upon by both regressors. This can lead to higher precision but may also discard true causal edges identified by only one model type, potentially resulting in a less comprehensive causal graph.

Experiments on the synthetic datasets proposed in this study (see Section 4) indicate that the union approach results in better overall performance. As shown in Fig. 11, the union of the regressors' outputs yields higher F1 scores (0.712) across datasets with 10, 15, 20, and 25 variables compared to both individual regressors (DFN F1: 0 . 577 ; GBT F1: 0 . 59 ) and the intersection strategy (F1: 0 . 667 ). Specifically, the union's F1 score is 23 . 4% higher than the average F1 score of the individual DFN and GBT models, and 4 . 9% higher than that of the intersection approach. This suggests that, for the datasets and configurations tested, the union provides a superior balance of precision and recall, capturing more true causal structures overall.

While the union carries a theoretical risk of including more spurious edges than the intersection if one model is prone to specific errors not fully mitigated by its individual bootstrapping phase, the subsequent cycle removal and edge orientation steps (Section 3.3 and Section 3.4), particularly the SHAP discrepancy measure for resolving conflicts,

GES

GES

REX

REX

®

FCI

(5)

Figure 12: Sample DAGs generated from a Gaussian additive process.

<!-- image -->

further refine the graph. The empirically superior F1-score ultimately guided our decision to adopt the union approach as the default strategy in REX.

## E Sample DAGs

In this section, a number of sample DAGs are presented to illustrate the differences between the output DAG of the proposed method REX and the ones from the comparison methods.

Fig. 12 and 13 show sample DAGs generated from a Gaussian additive and Sigmoid additive processes, respectively, with p = 10 variables. Solid green edges represent correctly predicted causal relationships that match the ground truth, light gray dotted lines indicate edges that were present in the ground truth but not predicted by the method, dashed orange lines represent edges where the causal pair was correctly identified but with the wrong direction, and red dash-dot lines correspond to incorrectly predicted edges.

Figure 13: Sample DAGs generated from a Sigmoid additive process.

<!-- image -->

In Fig. 12, REX outperforms the other methods by achieving a higher proportion of correctly oriented causal edges (solid green), particularly in areas with more complex relationships, such as nodes V 3 , V 4 , and V 6 . Moreover, REX

demonstrates a lower number of incorrect predictions (red dash-dot lines) and fewer reversed causal directions (dashed orange) compared to methods like LiNGAM and GES, which show more frequent orientation errors. This highlights REX's ability to both recover true causal relationships and accurately predict their direction, even in challenging scenarios.

In Fig. 13, REX method continues to demonstrate strong performance by recovering a larger number of true causal relationships (solid green lines) compared to other methods. Notably, REX accurately captures relationships involving nodes V 3 , V 4 , and V 6 , while methods like PC, LiNGAM, and GES tend to miss several true edges or predict a higher number of reversed causal directions (dashed orange lines). Furthermore, REX shows fewer incorrect predictions (red dash-dot lines) relative to CAM and FCI, both of which exhibit a greater number of spurious connections. This highlights REX's robustness in discovering complex causal structures and correctly orienting edges, particularly in this dataset where sigmoid additive relationships present a more challenging structure.

## F Additional results

Here we provide additional results of REX for the five considered families of synthetic datasets and the different values of input features p .

## G Synthetic data generation

Each synthetic dataset is built with p variables ( p = 10 , 15 , 20 , 25 ) and m = 500 samples.

1. The DAG structure is such that the number of parents for each variable is uniformly drawn in { 0 , . . . , 5 } ;
2. For the i -th DAG, the mean µ i and variance σ i of the noise variables are drawn as µ i ∼ U ( -2 , 2) and σ i ∼ U (0 , 0 . 4) , and the distribution of the noise variables is set to N ( µ i , σ i ) ;
3. For each graph, a 500 -sample dataset is i.i.d. generated following the topological order of the graph, with, for ℓ = 1 , . . . , 500 ,

<!-- formula-not-decoded -->

All variables are normalized to zero mean and unit variance.

Five categories of causal mechanisms have been considered, replicating those described in [26]:

- I. Linear: X i = ∑ j ∈ Pa( i ) a i,j X j + ε i , where a i,j ∼ N (0 , 1) .
- II. Polynomial: X i = ∑ j ∈ Pa( i ) c i,j X d j + ε i , where c i,j ∼ N (0 , 1) , and d is the degree of the polynomial.
- III. GP AM: X i = ∑ j ∈ Pa( i ) f i,j ( X j ) + ε i , where f i,j is a univariate Gaussian process with a Gaussian kernel of unit bandwidth.
- IV. GP Mix: X i = f i ( [ X Pa( i ) , ε i ] ) , where f i is a multivariate Gaussian process with a Gaussian kernel of unit bandwidth.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## H Computational complexity

The computational cost of REX, as depicted in Figure 1 (overall workflow) and detailed in Algorithms 1 and 2, is influenced by several stages. Understanding these helps contextualize the empirical runtime results presented in Figure 3d.

The primary computational stages of REX are:

- Model Training (Section 3.1): This involves training p predictive models for each of the two regressor types (DFN and GBT), where p is the number of variables. The cost is O ( p · ( C HPO + C train )) , where C HPO is the cost of hyperparameter optimization and C train is the cost of training a single regressor (DFN or GBT) on m samples and p -1 features. This depends on the chosen model's complexity (e.g., network architecture, number of trees) and training epochs.

Fl Score

Recall

1.0

0.8

0.6 -

0.4

0.2

0.0

1.0

0.8 -

0.6 -

0.4

0.2

0.0

300

250 h

200 -

€ 150 -

100 -

50

p = 10

= 15

p = 20

p = 25

Linear

-

Linear

Linear

1.0

0.8

0.6 -

Figure 14: F1 (a), Precision (b), Recall (c), SHD (d) and SID (e) for REX as a function of the number of features and synthetic data generation mechanism (Appendix G). Panel (f) represents the training time for the different families of synthetic data and number of variables used (x-axis).

<!-- image -->

- Bootstrapped SHAP Value Computation (Algorithm 1): This stage iterates T times. In each iteration, for every one of the p target variables, SHAP values are computed for the p -1 input features using a subset of m ′ samples. The cost of computing SHAP values, C SHAP, varies: TreeExplainer for GBTs is generally efficient, scaling with m ′ , p , number of trees, and tree depth (e.g., approximately O ( m ′ · N trees · D · p ) for m ′ samples explained over p features). GradientExplainer for DFNs can be more intensive, often scaling with m ′ , p , the number of samples for expectation, and the cost of model evaluation. This makes the overall SHAP computation roughly O ( T · p · C SHAP ) .

- Parent Selection (Algorithm 2): For each target variable within each bootstrap iteration, this algorithm processes p -1 SHAP values. It involves pairwise distance calculations and an iterative DBSCAN clustering approach. Given its current structure involving iterative adjustments of ζ based on distances, its complexity is estimated to be polynomial in p , potentially around O ( p 3 ) .
- Edge Orientation (Section 3.3): This is applied to E undir candidate edges (where E undir ≤ p ( p -1) / 2 ). It involves fitting two regression models (GAMs are employed for efficiency with m samples) and performing two HSIC tests per edge. HSIC tests typically scale as O ( m 2 ) (or O ( m log m ) with approximations). The overall complexity for this stage is roughly O ( E undir · ( C GAM + C HSIC )) .
- Final DAG Construction (Section 3.4): This includes graph union operations ( O ( p 2 ) ) and cycle detection/resolution. The cost of resolving cycles using SHAP discrepancy depends on the number and complexity of cycles and whether per-sample SHAP values ( ϕ k,j in Equation 12) are recomputed or readily available.

The dominant computational burden in REX typically arises from the bootstrapped SHAP value computation, especially as p and m increase, leading to a high-degree polynomial scaling in p . This theoretical expectation aligns with the empirical runtime trends observed in Figure 3d. Compared to other methods:

- Constraint-based methods (PC [2], FCI [12]): These can be exponential in p in the worst-case due to the need to test conditional independencies over many subsets, though they are often faster for sparse graphs. REX avoids this exhaustive CI testing by leveraging feature importance from global predictive models.
- Score-based methods (GES [13]): These perform a heuristic search in the space of DAGs, and their complexity depends on the size of this space and the cost of evaluating the score for each candidate structure.
- SCM-based methods: LiNGAM[14] is relatively efficient (e.g., O ( p 3 + p 2 m ) ) under its specific assumptions (linear, non-Gaussian). CAM [15] can be computationally intensive due to its search for parent sets and fitting of non-linear additive models.
- Continuous optimization methods (NOTEARS [17]): These typically have polynomial complexity (e.g., O ( mp 2 + p 3 ) per iteration or overall for some variants).

While REX incorporates steps that are computationally demanding, particularly SHAP value estimation, its design aims to harness the expressive power of machine learning models and the insights from XAI to uncover complex causal relationships. This represents a trade-off where enhanced accuracy or the ability to handle non-linearities (as demonstrated in Section 4) may come at a higher computational cost compared to methods with more restrictive assumptions or less exhaustive feature analysis.