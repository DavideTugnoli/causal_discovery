Oi = [Xi1, Xi2, •

Synthetic observations

Autoregressive model

*ij ~ P(X|xi1, Xi2, ...,Xij-1)

Causal LM Head

GPT-2

Xil, Xi2, ...,Xij-1

Parent table input

REaLTabFormer

•, Xin]

si = loi, 0},..., 071

Synthetic related data

, 27j-1, Ск) ;

## REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers

Ck

Encoder

Child table input

## Abstract

Tabular data is a common form of organizing data. Multiple models are available to generate synthetic tabular datasets where observations are independent, but few have the ability to produce relational datasets. Modeling relational data is challenging as it requires modeling both a 'parent' table and its relationships across tables. We introduce REaLTabFormer (Realistic Relational and Tabular Transformer), a tabular and relational synthetic data generation model. It first creates a parent table using an autoregressive GPT-2 model, then generates the relational dataset conditioned on the parent table using a sequence-to-sequence (Seq2Seq) model. We implement target masking to prevent data copying and propose the Q δ statistic and statistical bootstrapping to detect overfitting. Experiments using real-world datasets show that REaLTabFormer captures the relational structure better than a baseline model. REaLTabFormer also achieves state-of-the-art results on prediction tasks, 'out-of-the-box', for large nonrelational datasets without needing fine-tuning.

## 1. Introduction

Tabular data is one of the most common forms of data. Many datasets from surveys, censuses, and administrative sources are provided in this form. These datasets may contain sensitive information that cannot be shared openly (Abdelhameed et al., 2018). Even when statistical disclosure methods are applied, they may remain vulnerable to malicious attacks (Cheng et al., 2017). As a result, their dissemination is restricted and the data have limited utility (O'Keefe &amp; Rubin, 2015). Differential privacy methods (Ji et al., 2014), homomorphic encryption approaches (Aslett et al., 2015; Wood et al., 2020), or federated machine learning (Yang et al., 2019; Lin et al., 2021) may be implemented, allowing

1 Development Economics Data Group, The World Bank, USA. Correspondence to:

Aivin V. Solatorio &lt; asolatorio@worldbank.org &gt;

GitHub: @avsolatorio.

Seq2Seq model

## Aivin V. Solatorio 1 Olivier Dupriez 1

<!-- image -->

Figure 1. Illustration of the REaLTabFormer model. The left block shows the non-relational tabular data model using GPT-2 with a causal LM head. In contrast, the right block shows how a relational dataset's child table is modeled using a sequence-to-sequence (Seq2Seq) model. The Seq2Seq model uses the observations in the parent table to condition the generation of the observations in the child table. The trained GPT-2 model on the parent table, with weights frozen, is also used as the encoder in the Seq2Seq model.

insights from sensitive data to be accessible to researchers. Synthetic tabular data with similar statistical properties as the real data offer an alternative, offering more value especially in granular and segmentation analyses. To comply with data privacy requirements, the generative models that produce these synthetic data must provide guarantees that 'data copying' does not happen (Meehan et al., 2020; Carlini et al., 2023).

Formally, tabular data is a collection of observations (rows) o i that may or may not be independent. A single observation in a tabular data T with n columns is defined by o i = [ x i 1 , x i 2 , ..., x ij , ..., x in ] , and j indicating the j th column. We refer to tabular data having observations independent of each other as non-relational tabular data . Tabular data having observations related to each other are referred to as relational tabular data . Relational datasets have at least one pair of tabular data files with a one-to-many mapping of observations between the parent table and the child table,

respectively, linked by a unique identifier. In the context of a relational dataset, a parent table is a non-relational tabular data, whereas the child table is a relational tabular data. Relational tabular databases model the logical partitioning of data and prevent unnecessary duplication of observations from the parent to child tables (Jatana et al., 2012). Despite its ubiquity, limited work has been done in generating synthetic relational datasets. This may be due to the challenging nature of modeling the complex relationships within and across tables.

The field of synthetic data generation has seen significant development in recent years (Gupta et al., 2016; Abufadda &amp;Mansour, 2021; Hernandez et al., 2022; Figueira &amp; Vaz, 2022). Generative models have become mainstream with the advent of synthetic image generation models such as DALLE (Ramesh et al., 2021), and most recently, ChatGPT. While generative models for images and text are common, models for producing synthetic tabular data are comparatively limited despite their multiple possible applications. Synthetic tabular data can contribute to addressing data privacy issues and data sparseness (Appenzeller et al., 2022). They can help to make sensitive data accessible to researchers (Goncalves et al., 2020), and to fill gaps in data availability for counterfactual research and agent-based simulations (Fagiolo et al., 2019), and for synthetic control methods (Abadie et al., 2015). Further value can be derived from tabular data by building predictive models using machine learning (Shwartz-Ziv &amp; Armon, 2022). These predictive models can infer variables of interest in the data that may otherwise be expensive to collect or correspond to some success metrics that can guide business decisions. Synthetic data produced by deep learning models have been shown to perform well in predictive modeling tasks. This extends the utility of real-world data that may otherwise be unused due to privacy concerns.

This paper introduces the REalTabFormer , a transformerbased framework for generating non-relational tabular data and relational datasets. It makes the following contributions:

Unified framework The REalTabFormer uses an autoregressive (GPT-2) transformer to model non-relational tabular data for modeling and generating parent tables. It then models and generates observations in the child table using the sequence-to-sequence (Seq2Seq) (Yun et al., 2019) framework. The encoder network uses the pre-trained weights of the network for the parent table, contextualizing the input for generating arbitrary-length data corresponding to observations in a child table, via the decoder network.

Strategies for privacy-preserving training Synthetic data generation models must not only be able to generate realistic data but also implement safeguards to prevent the model from 'memorizing' and copying observations in the training data during sampling. We use the distance to clos- est record (DCR), a data-copying measure, and statistical bootstrapping to detect overfitting during training robustly. We introduce target masking for regularization to reduce the likelihood of training data being replicated by the model.

Open-sourced models We publish the REaLTabFormer models as an open-sourced Python package. Install the package using: pip install realtabformer . 1

Comprehensive evaluation Weevaluate the performance of our models on a variety of real-world datasets. We use open-sourced state-of-the-art models as baselines to assess the performance of REaLTabFormer in generating non-relational and relational tabular datasets.

Our experiments demonstrate the effectiveness of the REaLTabFormer model for non-relational tabular data, beating current state-of-the-art in machine learning tasks for large datasets. We further demonstrate that the synthesized observations for the child table generated by the REaLTabFormer capture relational statistics more accurately than the baseline models.

## 2. Related Work

Recent advances in deep learning, such as generative adversarial networks (Park et al., 2018; Xu et al., 2019; Zhao et al., 2021), autoencoders (Li et al., 2019; Xu et al., 2019; Darabi &amp;Elor, 2021), language models (Borisov et al., 2022), and diffusion models (Kotelnikov et al., 2022) have been applied to synthetic non-relational tabular data generation. These papers demonstrate deep learning models' capacity to produce more realistic data than traditional approaches such as Bayesian networks (Xu et al., 2019).

On the other hand, generative models for relational datasets are limited (Patki et al., 2016; Gueye et al., 2022). Existing models are based on Hierarchical Modeling Algorithms (Patki et al., 2016) where traditional statistical models, Gaussian Copulas, are used to learn the joint distributions for each and across tables. While these models can synthesize data, the quality of the generated data does not accurately capture the nuanced conditions within and across tables (Fig. 2 and Fig. 3).

Padhi et al. (2021) presented TabGPT for generating synthetic transactional data. They showed that autoregressive transformers, particularly GPT, can synthesize arbitrarylength data. One limitation of TabGPT is that one has to train independent models to produce transactions for each user. This becomes impractical for real-world applications. Our work generalizes the use of GPT by proposing a sequence-to-sequence framework for generating arbitrarylength synthetic data conditioned on an input.

1 https://github.com/avsolatorio/ REaLTabFormer

## 3. REaLTabFormer

REaLTabFormer is a transformer-based framework for generating non-relational tabular data using an autoregressive model and relational tabular data using a sequenceto-sequence (Seq2Seq) architecture. The framework also consists of strategies for encoding tabular data (Section 3.2), a statistical method to detect overfitting (Section 3.3.2), and a constrained sampling strategy during generation.

Details of the framework are described in this section. First, we present our proposed models to synthesize realistic relational datasets. Next, we outline and describe the data processing applied to the tabular data as input for training the models. We then discuss solutions to improve our model's training and sampling process.

## 3.1. The REaLTabFormer Models

Parent table model To generate synthetic observations for a non-relational tabular data T , we model the conditional distribution of columnar values in each row of the data. Consider a single observation o i = [ x i 1 , x i 2 , ..., x ij , ..., x in ] in T as defined earlier. We treat o i as a sequence with potential dependencies across values x ij , similar to a sentence in a text. This re-framing provides us with a framework to learn the conditional distribution x ij ∼ P ( X | x i 1 , x i 2 , ..., x ij -1 ) and sequentially generate the next values in the sequence, eventually generating the full observation (Jelinek, 1985; Bengio et al., 2000). We use an autoregressive model to learn this distribution, Fig. 1. In the context of relational datasets, we use this approach to generate synthetic observations for the parent table T 0 . We extend this formulation to generate the child table T ′ -a relational tabular dataassociated with the parent table T 0 .

Child table model The extension is established by introducing a context learned by an encoder network from observations in T 0 . Instead of generating o i in T ′ independently, we concatenate the child table observations related to the same observation in T 0 . This forms an arbitrary-length sequence s i = [ o 1 i , o 2 i , ..., o n i ] , where n is the number of related observations in T ′ .

We propose to model the generation of s i given an observation in T 0 as x n ij ∼ P ( X | o 1 i , ..., x n i 1 , x n i 2 , ..., x n ij -1 , C k ) , where C k is a context captured from a related observation in the parent tabular data T 0 . We also use the same network trained on the parent table, with weights frozen , as the Seq2Seq model's encoder. This choice is expected to speed up the training process since only the cross-attention layer and the decoder network are needed to be trained for the child table model. The encoder network is assumed to have learned the properties of the parent table and will transfer this information to the decoder without further fine-tuning its weights, Fig 1.

Figure 2. Graph of the daily mean of the Sales variable computed from the original Rossmann dataset (blue), synthetic data produced by REaLTabFormer (orange), and data generated by SDV (green). The REaLTabFormer closely captures the seasonality in the data compared with the HMA model from the SDV.

<!-- image -->

GPT-2: an autoregressive transformer Previous works have shown that transformer-based autoregressive models can capture the conditional distribution of sequential data very well (Radford et al., 2019; Padhi et al., 2021). REaLTabFormer uses the GPT-2 architecture-a transformerdecoder architecture designed for autoregressive tasks-as its base model. We adopt the same architecture for all GPT-2 instances in the framework for simplicity. The GPT-2 architecture used in the REaLTabFormer has 768-dimensional embeddings, 6 decoder layers, and 12 attention heads-a set of parameters similar to DistilGPT2. We use the implementation from the HuggingFace transformers library (Wolf et al., 2020).

## 3.2. Tabular Data Encoding

The GReaT model that uses pretrained large language models (LLMs) proposed by Borisov et al. (2022) offers insight into the minimal data processing requirements for language models in generating tabular data. There is, however, the potential for optimization in using autoregressive language models for this task, as the fine-tuning process of a large pretrained model incurs computational costs. Particularly, LLMs are trained on a large vocabulary where most of the tokens are not needed for generating the tabular data at hand. These unnecessary tokens increase the model's computational requirements and prolong training and sampling times. To improve the efficiency of our model, we adopt a fixed-set vocabulary as initially proposed by Padhi et al. (2021). Generating a fixed vocabulary for each column in the tabular data offers various advantages in training performance and sampling. One of the main advantages is being

Figure 3. Joint distributions of the age group variable in the parent table and the device type in the child table of the Airbnb test dataset (left), the SDV (middle), and the REalTabFormer (right). The plots show that the REaLTabFormer can synthesize values across the domain of the variables, while SDV learned only two device types out of thirteen. The REaLTabFormer also generalized and imputed age values for users with 'iPodtouch' device (red box). This device type group has missing values for age in the original data.

<!-- image -->

able to filter irrelevant tokens when generating values for a specific column. This directly contributes to efficiency in sampling by reducing the chances of generating invalid samples. Our model performs minimal transformation of the raw data. First, we identify the various data types for each column in the data. We then perform a series of data processing specific to the column and data type. Notably, we adopt a fully text-based strategy in handling numerical values. These transformations produce a transformed tabular data used to train the model. Borisov et al. (2022) showed that variable order has an insignificant impact on language models, so we did not apply variable permutation. We discuss the processing for each data type in Appendix A.

Training data for the parent table The GPT-2 model we use requires a set of token ids as input. To generate these sequences of token ids, we first create a vocabulary. This vocabulary maps the unique tokens in each column to a unique token id. Then, for each row in the modified data, we apply the mapping in the vocabulary to the tokens. This produces a list of token ids for each row of the data. The model is then trained on an autoregressive task wherein the target data corresponds to the right-shifted tokens of the input data.

Training data for the child table We concatenate the transformed observations corresponding to related rows in the child table. A special token is added before and after the set of tokens representing an individual observation. In this form, the data we use to train the Seq2Seq model contains input-output pairs. An input value contains a fixedlength array of token ids representing the observation in the parent table. The input is similar to the input used in the parent table model. An arbitrary-length array with the concatenated token ids for each related observation in the child table represents the output value. The number of related observations that can be modeled is limited by computational resources.

## 3.3. REaLTabFormer Training and Sampling

Deep learning models for generative tasks face challenges of overfitting the data resulting in issues such as data-copying (Meehan et al., 2020; Carlini et al., 2023). Furthermore, observations generated by generative models for tabular data could face issues of validity and inconsistency. These issues in the generated samples impact the efficiency of the generative process. Our proposed framework addresses the aforementioned issues by, (i) introducing a robust statistical method to monitor overfitting, and (ii) target masking to further reduce the risk of data copying. To improve the rate of producing valid observations by the model, we also implement a constrained generation strategy during the sampling stage.

## 3.3.1. TARGET MASKING

Data copying is a critical issue for deep learning-based generative tabular models as it can expose and compromise sensitive information in the training data. To mitigate datacopying, we introduce target masking . Target masking is a form of regularization aimed at minimizing the likelihood of records in the training data being 'memorized' and copied by the generative model. Unlike the token masking introduced in BERT (Devlin et al., 2018), where input tokens are masked and the model is expected to predict the correct token, target masking implements random replacement of the target or label tokens with a special mask token. This artificially introduces missing values in the data.

We intend for the model to learn the masks instead of the actual values. During the sampling stage, we then restrict the mask token generation, forcing the model to fill the value with a valid token probabilistically. Notably, even when the model learns to copy the input-output pair, the learned output corresponds to the masked version of the input. Therefore, when we process the output, the probabilistic nature of replacing the mask token reduces the likelihood of generat-

ing the training data. The mask rate parameter controls the proportion of the tokens that are masked. We use a mask rate of 10% in our experiments.

## 3.3.2. OVERFITTING ASSESSMENT

Applying deep learning models to small datasets may easily result in overfitting. This may cause privacy-related issues when the model generates observations that are copied from the training data. Knowing when the model overfits the data is also crucial when the purpose is to generate diverse out-ofsample data. An overfitted model tends to generate samples generally closer to the training data, thereby limiting the generalization capacity of the model. While the former issue can be resolved by post-generation filtering, the latter must be detected during the model training.

Taking hold-out data to detect overfitting is a common strategy in machine learning. Unfortunately, this strategy could result in the premature termination of model training. It may also penalize a model based only on a small subset of the data (Blum et al., 1999). The training procedures of existing state-of-the-art models do not explicitly check for overfitting (Xu et al., 2019; Borisov et al., 2022; Kotelnikov et al., 2022). We propose and describe below an empirical statistical method to inform the generative model when overfitting happens. The method allows for the full data to be used in the training without the need for a hold-out set. The design of the method is expected to also help prevent data copying and the production of data that is riskily close to the training data.

Distance to closest record We use the distance to the closest record (DCR) (Park et al., 2018) to measure the similarity of synthetic samples to the original data. The DCR is evaluated by taking a specified distance metric L between the training data T r and the generated data G . We then find the smallest distance for each record. Consider the distance matrix between T r and G ,

<!-- formula-not-decoded -->

The minimum value in each row i of D is the minimum distance of the i th record in the training data with respect to all records in the generated data. We denote this set of minimum values as glyph[vector] d i . The minimum value in each column j of D is the minimum distance of the j th record in the generated data with respect to all records in the training data. We denote this set of minimum values as glyph[vector] d j . We then take glyph[vector] d g = [ glyph[vector] d i , glyph[vector] d j ] as the distribution of distances to closest records between the training data and the generated data. We define the quantity

<!-- formula-not-decoded -->

as the distance to closest record distribution for some T r and some arbitrary sample. We also derive the DCR between the train dataset and some hold-out data T h . Let us denote this distribution of distances as glyph[vector] d h . We use the distributions glyph[vector] d g and glyph[vector] d h in our proposed non-parametric method.

Quantile difference ( Q δ ) statistic Two samples from a similar distribution should, on average, have approximately the same values at each quantile of the distribution. To detect whether two samples come from different distributions, we define a set of quantiles over which we compare the two samples. For each quantile in the set, we find the value at the given quantile in one sample and measure the proportion of the values in the other sample that are below it. If the distributions are similar, the proportion should be close to the given quantile, for all quantiles being tested.

Formally, let S h and S g having m and n observations, respectively, be two samples being compared. Let Q be a set of N quantiles, and q ∈ Q is a specific quantile in the set. Consider v q as the value in S h at quantile q . Then, we compute the value

<!-- formula-not-decoded -->

where p q is the proportion of values in S g that are less than or equal to v q . We define a statistic

<!-- formula-not-decoded -->

This formulation has similarities with the Cramer-von Mises ω 2 criterion, but the Q δ statistic has one key difference: the asymmetry of the statistic. This stems from the fact that the choice of which sample is considered as S h -the distribution from which v q is identified-matters. Since we are averaging over the quantiles, this statistic may not yield conclusive guidance for distributions with cumulative distribution functions (CDFs) intersecting at some quantile. Nonetheless, this statistic works best in detecting the dissimilarity of the two samples at the left tail of the distribution which matters most for our purpose. This is because the distributions we are comparing are the DCRs. We want to detect when the distance between the sample and the training data is significantly closer to zero than expected.

We use the Q δ statistic as the basis for detecting overfitting. The threshold against which this statistic will be compared during training is produced through an empirical bootstrapping over random samples from the training data. The details of the bootstrapping method are explained next.

Q δ statistic threshold via bootstrapping We use three hyperparameters in estimating the threshold that will signal when overfitting occurs during training. First, a sample proportion ρ corresponds to a fraction of the training data. This

Table 1. Machine learning efficacy (MLE) and discriminator measure (DM) evaluated on the synthetic data generated by the models (columns) trained on six real-world datasets (rows): Abalone (AB), Adult income (AD), Buddy (BU), California housing (CA), Diabetes (DI), and Facebook Comments (FB). The MLE is measured by the R 2 for regression, while macro average F 1 is used for classification tasks; higher scores are better. A discriminator measure closer to 50% is better. Best scores are highlighted for the MLE measure, considering standard deviation. No reported results for GReaT on the FB dataset due to impractical training time.

|            |                    | Original           | TVAE                            | CTABGAN+                        | Tab-DDPM                        | GReaT                           | REaLTabFormer                   |
|------------|--------------------|--------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| AB ( R 2 ) | MLE ( ↑ ) DM ( ↓ ) | 0.5562 ± 0 . 004 - | 0.3943 ± 0 . 012 82.96 ± 2 . 42 | 0.4697 ± 0 . 014 75.64 ± 1 . 20 | 0.5248 ± 0 . 011 59.88 ± 2 . 22 | 0.3530 ± 0 . 031 70.46 ± 0 . 92 | 0.5035 ± 0 . 011 63.08 ± 1 . 18 |
| AD ( F 1 ) | MLE DM             | 0.8155 ± 0 . 001 - | 0.7695 ± 0 . 004 95.48 ± 1 . 34 | 0.7778 ± 0 . 003 61.17 ± 0 . 40 | 0.7922 ± 0 . 002 53.73 ± 0 . 22 | 0.7997 ± 0 . 002 68.04 ± 0 . 26 | 0.8113 ± 0 . 002 55.78 ± 0 . 20 |
| BU ( F 1 ) | MLE DM             | 0.9303 ± 0 . 002 - | 0.9233 ± 0 . 002 66.56 ± 0 . 56 | 0.9267 ± 0 . 002 58.33 ± 0 . 49 | 0.9057 ± 0 . 003 54.43 ± 0 . 47 | 0.9279 ± 0 . 003 62.18 ± 0 . 45 | 0.9278 ± 0 . 003 55.86 ± 0 . 47 |
| CA ( R 2 ) | MLE DM             | 0.8568 ± 0 . 001 - | 0.7373 ± 0 . 004 62.06 ± 0 . 60 | 0.5231 ± 0 . 006 90.14 ± 1 . 03 | 0.8252 ± 0 . 002 54.30 ± 0 . 89 | 0.7189 ± 0 . 004 66.78 ± 0 . 47 | 0.8076 ± 0 . 003 57.29 ± 0 . 56 |
| DI ( F 1 ) | MLE DM             | 0.7759 ± 0 . 014 - | 0.7395 ± 0 . 035 90.16 ± 1 . 31 | 0.7339 ± 0 . 024 70.94 ± 1 . 99 | 0.7448 ± 0 . 031 69.00 ± 1 . 56 | 0.7419 ± 0 . 03 74.88 ± 1 . 79  | 0.7315 ± 0 . 027 75.56 ± 2 . 84 |
| FB ( R 2 ) | MLE DM             | 0.8371 ± 0 . 001 - | 0.6374 ± 0 . 007 97.72 ± 0 . 80 | 0.4722 ± 0 . 053 93.60 ± 0 . 28 | 0.6850 ± 0 . 006 66.07 ± 0 . 23 | - -                             | 0.7702 ± 0 . 004 65.46 ± 0 . 83 |

fraction will be randomly sampled during the bootstrapping and evaluation phases of the generative model training. Second, the α value for choosing the critical threshold for the bootstrap statistic. Third, we specify a bootstrap round B corresponding to the number of times we compute the Q δ statistic between three random samples-two, each of size ρ , and the rest having size 1 -2 ρ of the training data.

Formally, for a given training data T r with N observations, we define a bootstrap method to generate a confidence interval for the Q δ statistic specific to the tabular data at hand. For each bootstrap round b ∈ B , we take three random samples S tr , S h , and S g , without replacement. S h and S g are each of size ρN , while S tr contains (1 -2 ρ ) N samples. We compute the DCR distributions glyph[vector] d g and glyph[vector] d h for the two samples S h and S g , respectively, relative to sample S tr . We then compute the Q δ statistic between glyph[vector] d g and glyph[vector] d h , where we take glyph[vector] d h as the distribution from which we compute the value v q in Equation 3. We store the statistic computed across the bootstrap rounds. We use the specified α to get the cutoff value that will be used as the statistic threshold. We use this threshold Q ′ δ during training to compare the Q δ statistic derived from the generated samples by the model. We set ρ = 0 . 165 , α = 0 . 95 , and B = 500 in our experiments.

Early stopping with Q ′ δ Our training procedure is paused at each epoch that is a multiple of E . We generate data from the model during these epochs. The generated data has size S g . We then take two mutually exclusive random samples from the training data, without replacement, to represent S tr and S h . We compute the Q e δ for this epoch based on the samples generated and drawn. Then, we compare this statistic to the previously computed threshold Q ′ δ . We continue training the model if Q e δ &lt; Q ′ δ . We save a checkpoint of this model. We terminate the model training when Q e δ &gt; Q ′ δ for X consecutive epochs. We then load the checkpoint for the most recent model that satisfied the condition Q e δ &lt; Q ′ δ . In our experiments, we set E = 5 as the period of our overfitting evaluation and X = 2 as our grace period before training termination.

## 3.3.3. SAMPLING

The models we use build each observation sequentially, one token at a time. We leverage the structure of our data processing to optimize the generation of samples from the trained models. Using a vocabulary specific to a column in the input data allows us to implement a constrained generation of tokens for each column.

We track the token ids that form the domain of each column during the generation of the vocabulary using a hash map. Based on this, the tokens that are invalid for the columns will not be considered for generation in the timestep representing the column. This strategy allows for efficient sampling wherein the likelihood of generating an invalid sample is close to zero. In our experiments, glyph[lessmuch] 1% invalid samples are generated during the sampling phase.

## 4. Experiments and Results

This section outlines the evaluation process we conducted to quantify the performance of the proposed REaLTabFormer framework compared with baseline models. We first demonstrate that the performance of the model we use to generate the parent tables, and non-relational tabular data in general, compares with or exceeds the performance of state-of-

Table 2. Logistic detection (LD) measure using random forest model for the generated parent, child, and merged tables by the Hierarchical Modeling Algorithm (HMA) from SDV and the REaLTabFormer (RTF) models. Our model consistently beats the SDV model on this metric.

| DATASET   | TABLE               | SDV                                        | RTF                                          |
|-----------|---------------------|--------------------------------------------|----------------------------------------------|
| ROSSMANN  | PARENT CHILD MERGED | 31.77 ± 3 . 41 6.53 ± 0 . 39 2.80 ± 0 . 25 | 81.04 ± 4 . 54 52.08 ± 0 . 89 28.33 ± 2 . 31 |
| AIRBNB    | PARENT CHILD MERGED | 7.37 ± 0 . 72 0.00 ± 0 . 00 0.00 ± 0 . 00  | 89.65 ± 1 . 92 30.48 ± 0 . 79 21.43 ± 1 . 10 |

the-art models in real-world tabular data generation tasks measured by the machine learning efficacy metric. We also use the discriminator measure to quantify how realistic the samples generated by each model are. We proceed to model real-world relational datasets and show, quantitatively using logistic detection, that the synthetic data produced by the REaLTabFormer are more realistic and accurate.

## 4.1. Data

We use a collection of real-world datasets, listed in Table 3, commonly used in previous works for non-relational tabular data generation (Xu et al., 2019; Zhao et al., 2021; Gorishniy et al., 2021; Borisov et al., 2022; Kotelnikov et al., 2022). These datasets differ with respect to the number of observations, ranging from 768 up to 197,080 observations. There is also variation in the number of variables they contain, ranging from 8 to 50 numerical variables and 0 up to 8 categorical variables. The datasets cover regression, binary, and multi-class classification prediction tasks.

We use two real-world datasets to compare the performance of the REaLTabFormer on modeling relational tabular data compared with the baseline. These datasets are the Rossmann dataset and the Airbnb dataset used in prior work on synthetic relation data generation (Patki et al., 2016).

## 4.2. Baseline models

Non-relational tabular data We use models that apply different deep learning architectures for generating nonrelational tabular data as baselines to compare the REaLTabFormer model with. The TVAE is based on variational autoencoder (Xu et al., 2019), the CTABGAN+ on GAN architecture (Zhao et al., 2022), the Tab-DDPM on diffusion (Kotelnikov et al., 2022), and GReaT uses pretrained LLM (Borisov et al., 2022).

Relational datasets Models for generating relational datasets are limited. Gueye et al. (2022) published work on using GAN for relational datasets but no open-sourced im-

Figure 4. Summary of the average 'Sales' variable in the child table of the Rossmann dataset grouped by 'StoreType' variable in the parent table. The values shown are from the original data (blue), synthetic data produced by REaLTabFormer (orange), and data generated by SDV (green). This graph shows that REaLTabFormer captures the inter-table variations and relationships well.

<!-- image -->

plementation is available. We choose to limit our baselines to open-sourced models; hence, we only use the Hierarchical Modeling Algorithm (HMA) available in the Synthetic Data Vault (SDV) as our baseline (Patki et al., 2016).

## 4.3. Generative models training

The GReaT model was trained for 100 epochs for each data. The parameters for the TVAE, CTABGAN+, and Tab-DDPM models had been tuned for the predictive task itself using the real validation data from Kotelnikov et al. (2022). For the relational datasets, we trained the HMA model as prescribed in the SDV documentation. In contrast, the REaLTabFormer model was not tuned against any of the machine learning tasks. The model solely relied on the overfitting metric discussed in Section 3.3.2. We used the same parameters for the different datasets to test how the REaLTabFormer performs 'out-of-the-box'.

## 4.4. Measures and Results

We select the following measures to quantify the quality and utility of the generated samples by the generative models.

Machine Learning (ML) efficacy The machine learning (ML) efficacy (Xu et al., 2019; Kotelnikov et al., 2022; Borisov et al., 2022) measures the potential utility of the synthetic data to supplant the real data for machine learning tasks, in particular, training a prediction model. The ML efficacy reported by Borisov et al. (2022) in their work used ML models that were not fine-tuned. Kotelnikov et al.

(2022) showed that the ML efficacy computed from models that are not fine-tuned may show spurious results. They instead optimized the ML models-CatBoost (Prokhorenkova et al., 2018)-they used in reporting the ML efficacy. This approach is closer to what researchers are expected to do in the real world, therefore, we adopt these tuned models in our experiments. We generate a validation set from the generative models to signal the early-stopping condition during the ML model training. This is in contrast with the method used by Kotelnikov et al. (2022) where they still relied on the real validation data for the early-stopping of the ML model.

We report the macro average F1 score (Opitz &amp; Burst, 2019) for classification tasks and the R 2 metric for regression tasks. Our results presented in Table 1 (MLE) show that REaLTabFormer, despite not being fine-tuned, produces ML efficacy scores that are the best or second-best compared with the baselines. This demonstrates that REaLTabFormer can be used, 'out-of-the-box', to generate synthetic data with state-of-the-art performance in machine learning tasks.

The FB comments dataset, where REaLTabFormer obtained the best performance, is the largest dataset tested and has the largest number of columns. Training the GReaT model on this dataset yielded impractical runtime so no result is reported. This supports our view that using LLM trained on a large vocabulary, containing a majority of irrelevant tokens, limits the efficiency of the model.

Discriminator measure We adopt the discriminator measure (Borisov et al., 2022) to quantify whether the data generated by a model is easily distinguishable from real data. A dataset is made by combining an equal number of real and synthetic data. Real observations in this dataset are labeled as '1' and synthetic observations are labeled as '0'. Similar to Borisov et al. (2022), we train a random forest model to predict the labels given an observation. A held-out dataset containing a combination of synthetic samples and real test data is then used to report the final measure.

An accuracy that is closer to 50% implies better synthetic data quality since the discriminative model is not able to distinguish the real from the synthetic observations. We report our results in Table 1 (DM). The DM measure shows that the Tab-DDPM has the most indistinguishable synthetic data. Nonetheless, REaLTabFormer, without the need for tuning, has DM measures that are close to the Tab-DDPM. This suggests that the synthetic data produced by a diffusionbased model and REaLTabFormer are realistic compared with the other baselines.

Logistic Detection For relational datasets, we use logistic detection (LD) (Fisher et al., 2019; Gueye et al., 2022) to quantify the quality of the parent, child, and merged tables generated by REaLTabFormer compared with the HMA

model. We evaluate ROC-AUC scores averaged over (N=3) cross-validation folds,

<!-- formula-not-decoded -->

The value reported is LD = 100 × (1 -µ RA ) , where scores range from 0 to 100, and scores closer to 100 imply better synthetic data quality. We use random forest in measuring the logistic detection instead of the standard logistic regression model. The random forest model captures non-linearity in the data well than logistic regression (Couronn´ e et al., 2018), reducing the likelihood of spurious results. We report the results in Table 2. Additional LD results using logistic regression are shown in Table 4.

The REaLTabFormer model produces significantly higherquality synthetic data than the HMA model across the datasets tested. The high values of LD for the child and the merged tables highlight the ability of REaLTabFormer to accurately synthesize relational datasets in comparison with the leading baseline. The LD metric shows that data generated by SDV for the Airbnb child table is entirely distinguishable from the real data. The quantitative results are supported by relational statistics computed from synthetic datasets produced by REaLTabFormer and the HMA model: Figures 2 to 4.

## 5. Conclusion

We presented REaLTabFormer, a framework capable of generating high-quality non-relational tabular data and relational datasets. This work extends the application of sequence-to-sequence models to modeling and generating relational datasets. We introduced target masking as a component in the model to mitigate data-copying and safeguarding from potentially sensitive data leaking from the training data. We proposed a statistical method and the Q δ statistic for detecting overfitting in model training. This statistical method may be adapted to other generative model training. We showed that our proposed model generates realistic synthetic tabular data that can be a proxy for real-world data in machine learning tasks. REaLTabFormer's ability to model relational datasets accurately compared with existing opensourced alternative contributes to solving existing gaps in generative models for realistic relational datasets. Finally, this work can be extended and applied to data imputation, cross-survey imputation, and upsampling for machine learning with imbalanced data. A BERT-like encoder can be used instead of GPT-2 with the REaLTabFormer for modeling relational datasets. We also see opportunities to improve privacy protection strategies and the development of more components like target masking embedded into synthetic data generation models to prevent sensitive data exposure.

## 6. REaLTabFormer Python Package

We publish the REaLTabFormer as a package on PyPi. We show below how the model can be easily trained on any tabular dataset, loaded as a Pandas DataFrame.

## 6.1. Non-relational tabular model

Use the following snippet to fit the REaLTabFormer on a non-relational tabular dataset. One can control the various hyper-parameters of the model and the fitting method, e.g., the number of bootstrap rounds num bootstrap , the fraction of training data frac used to generate the Q δ statistic, etc. Keyword arguments for the HuggingFace transformers Trainer class can also be passed as **kwargs when initializing the model.

```
1 # pip install realtabformer 2 import pandas as pd 3 from realtabformer import REaLTabFormer 4 5 # NOTE: Remove any unique identifiers in the 6 # data that you don't want to be modeled. 7 df = pd.read_csv("foo.csv") 8 9 # Non-relational or parent table. 10 rtf_model = REaLTabFormer( 11 model_type="tabular", 12 gradient_accumulation_steps=4) 13 14 # Fit the model on the dataset. 15 # Additional parameters can be 16 # passed to the ` .fit ` method. 17 rtf_model.fit(df) 18 19 # Save the model to the current directory. 20 # A new directory ` rtf_model/ ` will be created. 21 # In it, a directory with the model's 22 # experiment id ` idXXXX ` will also be created 23 # where the artefacts of the model will be stored. 24 rtf_model.save("rtf_model/") 25 26 # Generate synthetic data with the same 27 # number of observations as the real dataset. 28 samples = rtf_model.sample(n_samples=len(df)) 29 30 # Load the saved model. The directory to the 31 # experiment must be provided. 32 rtf_model2 = REaLTabFormer.load_from_dir( 33 path="rtf_model/idXXXX")
```

## 6.2. Non-relational tabular model

REaLTabFormer for relational databases requires a twophase training. First, the model for the parent table is trained as a non-relational tabular data, then saved. Second, we pass the path of the saved parent model when creating the REaLTabFormer instance for the child model to be used as its encoder, then train. Generate synthetic samples from the parent table and use as input to the trained child model to generate the synthetic relational observations.

```
1 # pip install realtabformer 2 import os 3 import pandas as pd 4 from realtabformer import REaLTabFormer 5 6 pdir = Path("rtf_parent/") 7 parent_df = pd.read_csv("foo.csv") 8 child_df = pd.read_csv("bar.csv") 9 join_on = "unique_id" 10 11 # Make sure that the key columns in both the 12 # parent and the child table have the same name. 13 assert ((join_on in parent_df.columns) and 14 (join_on in child_df.columns)) 15 16 # Non-relational or parent table. Don't include the 17 # unique_id field. 18 parent_model = REaLTabFormer(model_type="tabular") 19 parent_model.fit(parent_df.drop(join_on, axis=1)) 20 parent_model.save(pdir) 21 22 # # Get the most recently saved parent model, 23 # # or a specify some other saved model. 24 # parent_model_path = pdir / "idXXX" 25 parent_model_path = sorted([ 26 p for p in pdir.glob("id*") if p.is_dir()], 27 key=os.path.getmtime)[-1] 28 29 child_model = REaLTabFormer( 30 model_type="relational", 31 parent_realtabformer_path=parent_model_path, 32 output_max_length= None , train_size=0.8) 33 34 child_model.fit( 35 df=child_df, in_df=parent_df, join_on=join_on) 36 37 # Generate parent samples. 38 parent_samples = parent_model.sample(len(parend_df)) 39 40 # Create the unique ids based on the index. 41 parent_samples.index.name = join_on 42 parent_samples = parent_samples.reset_index() 43 44 # Generate the relational observations. 45 child_samples = child_model.sample( 46 input_unique_ids=parent_samples[join_on], 47 input_df=parent_samples.drop(join_on, axis=1), 48 gen_batch=64)
```

Acknowledgments This project was supported by the 'Enhancing Responsible Microdata Access to Improve Policy and Response in Forced Displacement Situations' project funded by the World Bank-UNHCR Joint Data Center on Forced Displacement (JDC) - KP-P174174-GINPTF0B5124. We also thank Patrick Brock for providing insightful comments. The findings, interpretations, and conclusions expressed in this paper are entirely those of the authors. They do not necessarily represent the views of the International Bank for Reconstruction and Development/World Bank and its affiliated organizations, or those of the Executive Directors of the World Bank or the governments they represent.

## References

- Abadie, A., Diamond, A., and Hainmueller, J. Comparative politics and the synthetic control method. American Journal of Political Science , 59(2):495-510, 2015.
- Abdelhameed, S. A., Moussa, S. M., and Khalifa, M. E. Privacy-preserving tabular data publishing: a comprehensive evaluation from web to cloud. Computers &amp; Security , 72:74-95, 2018.
- Abufadda, M. and Mansour, K. A survey of synthetic data generation for machine learning. In 2021 22nd International Arab Conference on Information Technology (ACIT) , pp. 1-7. IEEE, 2021.
- Appenzeller, A., Leitner, M., Philipp, P., Krempel, E., and Beyerer, J. Privacy and utility of private synthetic data for medical data analyses. Applied Sciences , 12(23):12320, 2022.
- Aslett, L. J., Esperanc ¸a, P. M., and Holmes, C. C. A review of homomorphic encryption and software tools for encrypted statistical machine learning. arXiv preprint arXiv:1508.06574 , 2015.
- Bengio, Y., Ducharme, R., and Vincent, P. A neural probabilistic language model. Advances in neural information processing systems , 13, 2000.
- Blum, A., Kalai, A., and Langford, J. Beating the hold-out: Bounds for k-fold and progressive cross-validation. In Proceedings of the twelfth annual conference on Computational learning theory , pp. 203-208, 1999.
- Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., and Kasneci, G. Language Models are Realistic Tabular Data Generators, October 2022. arXiv:2210.06280 [cs].
- Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tram` er, F., Balle, B., Ippolito, D., and Wallace, E. Extracting training data from diffusion models. arXiv preprint arXiv:2301.13188 , 2023.
- Cheng, L., Liu, F., and Yao, D. Enterprise data breach: causes, challenges, prevention, and future directions. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery , 7(5):e1211, 2017.
- Couronn´ e, R., Probst, P., and Boulesteix, A.-L. Random forest versus logistic regression: a large-scale benchmark experiment. BMC bioinformatics , 19:1-14, 2018.
- Darabi, S. and Elor, Y. Synthesising multi-modal minority samples for tabular data. arXiv preprint arXiv:2105.08204 , 2021.
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
- Fagiolo, G., Guerini, M., Lamperti, F., Moneta, A., and Roventini, A. Validation of agent-based models in economics and finance. In Computer simulation validation , pp. 763-787. Springer, 2019.
- Figueira, A. and Vaz, B. Survey on synthetic data generation, evaluation methods and gans. Mathematics , 10(15):2733, 2022.
- Fisher, C. K., Smith, A. M., and Walsh, J. R. Machine learning for comprehensive forecasting of alzheimer's disease progression. Scientific reports , 9(1):1-14, 2019.
- Goncalves, A., Ray, P., Soper, B., Stevens, J., Coyle, L., and Sales, A. P. Generation and evaluation of synthetic patient data. BMC medical research methodology , 20(1): 1-40, 2020.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting deep learning models for tabular data. Advances in Neural Information Processing Systems , 34: 18932-18943, 2021.
- Gueye, M., Attabi, Y., and Dumas, M. Row conditionaltgan for generating synthetic relational databases. arXiv preprint arXiv:2211.07588 , 2022.
- Gupta, A., Vedaldi, A., and Zisserman, A. Synthetic data for text localisation in natural images. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 2315-2324, 2016.
- Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., and Rankin, D. Synthetic data generation for tabular health records: A systematic review. Neurocomputing , 2022.
- Jatana, N., Puri, S., Ahuja, M., Kathuria, I., and Gosain, D. A survey and comparison of relational and non-relational database. International Journal of Engineering Research &amp;Technology , 1(6):1-5, 2012.
- Jelinek, F. Markov source modeling of text generation. In The impact of processing techniques on communications , pp. 569-591. Springer, 1985.
- Ji, Z., Lipton, Z. C., and Elkan, C. Differential privacy and machine learning: a survey and review. arXiv preprint arXiv:1412.7584 , 2014.
- Kotelnikov, A., Baranchuk, D., Rubachev, I., and Babenko, A. Tabddpm: Modelling tabular data with diffusion models. arXiv preprint arXiv:2209.15421 , 2022.

- Li, S.-C., Tai, B.-C., and Huang, Y. Evaluating variational autoencoder as a private data release mechanism for tabular data. In 2019 IEEE 24th Pacific Rim International Symposium on Dependable Computing (PRDC) , pp. 1981988. IEEE, 2019.
- Lin, J., Ma, J., and Zhu, J. Privacy-preserving household characteristic identification with federated learning method. IEEE Transactions on Smart Grid , 13(2):10881099, 2021.
- Meehan, C., Chaudhuri, K., and Dasgupta, S. A nonparametric test to detect data-copying in generative models. In International Conference on Artificial Intelligence and Statistics , 2020.
- O'Keefe, C. M. and Rubin, D. B. Individual privacy versus public good: protecting confidentiality in health research. Statistics in medicine , 34(23):3081-3103, 2015.
- Opitz, J. and Burst, S. Macro f1 and macro f1. arXiv preprint arXiv:1911.03347 , 2019.
- Padhi, I., Schiff, Y., Melnyk, I., Rigotti, M., Mroueh, Y., Dognin, P., Ross, J., Nair, R., and Altman, E. Tabular Transformers for Modeling Multivariate Time Series. Institute of Electrical and Electronics Engineers Inc., June 2021. doi: 10.1109/ICASSP39728.2021.9414142. ISSN: 15206149.
- Park, N., Mohammadi, M., Gorde, K., Jajodia, S., Park, H., and Kim, Y. Data synthesis based on generative adversarial networks. arXiv preprint arXiv:1806.03384 , 2018.
- Patki, N., Wedge, R., and Veeramachaneni, K. The Synthetic Data Vault. In 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA) , pp. 399410, 2016. doi: 10.1109/DSAA.2016.49.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. Catboost: unbiased boosting with categorical features. Advances in neural information processing systems , 31, 2018.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language Models are Unsupervised Multitask Learners. 2019.
- Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot textto-image generation. In International Conference on Machine Learning , pp. 8821-8831. PMLR, 2021.
- Shwartz-Ziv, R. and Armon, A. Tabular data: Deep learning is not all you need. Information Fusion , 81:84-90, 2022.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., and Rush, A. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations , pp. 38-45, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6.
- Wood, A., Najarian, K., and Kahrobaei, D. Homomorphic encryption for machine learning in medicine and bioinformatics. ACM Computing Surveys (CSUR) , 53(4):1-35, 2020.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. Modeling tabular data using conditional GAN. In Proceedings of the 33rd International Conference on Neural Information Processing Systems , number 659, pp. 7335-7345. Curran Associates Inc., Red Hook, NY, USA, December 2019.
- Yang, Q., Liu, Y., Chen, T., and Tong, Y. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology (TIST) , 10(2):1-19, 2019.
- Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., and Kumar, S. Are transformers universal approximators of sequence-to-sequence functions? arXiv preprint arXiv:1912.10077 , 2019.
- Zhao, Z., Kunar, A., Birke, R., and Chen, L. Y. Ctab-gan: Effective table data synthesizing. In Asian Conference on Machine Learning , pp. 97-112. PMLR, 2021.
- Zhao, Z., Kunar, A., Birke, R., and Chen, L. Y. Ctabgan+: Enhancing tabular data synthesis. arXiv preprint arXiv:2204.00401 , 2022.

## A. Raw data processing

Numerical data Various methods have been proposed for representing numerical data as input to generative models in the context of tabular data. The CTGAN and TVAE models suggest the use of gaussian mixture models to encode numerical values (Xu et al., 2019). On the other hand, the TabFormer model introduced quantization as a way to encode numeric data (Padhi et al., 2021). However, these approaches are lossy. As argued by Borisov et al. (2022), these lossy transformations may not be optimal.

In our model, we adopt a fully text-based strategy in handling numerical values. We apply a sequence of transformations that converts a column of numeric value into, possibly, multi-columnar data. We use the following transformation of numerical columns. We also show the outcome of each transformation step on the sample numerical series below.

For illustration, this example numerical-valued series

<!-- formula-not-decoded -->

is converted into

<!-- formula-not-decoded -->

- We set a rounding resolution to normalize the size of the numerical values. For example, round to at most 2 decimal places.

<!-- formula-not-decoded -->

- We then cast the values to string.

<!-- formula-not-decoded -->

- We identify the magnitude of the most significant digit of the largest value in the column by looking for the location of the decimal point of the largest value. The magnitude of the most significant digit for the largest value in the example is 4.

<!-- formula-not-decoded -->

- We use the magnitude to left-align all the other values in the data by padding them with leading zeros.

<!-- formula-not-decoded -->

- We then take the length of the longest string after this transformation and left-justify the data by padding zeros to the right of the values that are shorter than the longest string.
- Then, the negative sign for negative values is transposed to the leftmost part of the string.

```
-['1032 . 33' , '0010 . 29' , '00 -3 . 00']
```

<!-- formula-not-decoded -->

- Note that for integral values, we only perform the left alignment by padding the values with leading zeros.

After this series of transformations, we tokenize the values into fixed-length partitions. For the same example values, say we choose the partition size to be 2, we get the following tokenized table.

<!-- formula-not-decoded -->

This transformation is done to mitigate the explosion of the vocabulary if the numeric values are all distinct. We found in our experiments that using single-character partitioning works best. We suppose that this effect is attributable to the inherent regularization of generating an entire sequence of numbers one digit at a time.

Datetime data For date or time data types, we first perform a transformation of the raw data into Unix timestamp representation. This representation is then treated as regular numeric data; hence, we apply the data processing discussed for numeric data types.

Categorical data Unique values in categorical columns are treated as unique tokens in the vocabulary. No additional processing is done.

Missing values No transformation is done for missing values present in the data. We let the model learn the distribution of the missing values. This strategy gives us the flexibility to let the model impute or generate missing values during the sampling process.

Input data aggregation As illustrated above, the transformation of numerical data types expands the dataset by partitioning the string version of the values. As such, we combine the processed columns into modified tabular data. We use this modified tabular data as input for our models. Each unique value in the new columns in this data will be mapped to a unique token in the vocabulary that is independent of values in the other columns. This means that in the illustrated numerical transformation shown above, the '1' in the first column will have a different token id than the '1' present in the third column.

Table 3. Summary of the datasets used in the experiments for non-relational tabular data.

| Abbr   | Name                     |   # Train |   # Validation |   # Test |   # Num |   # Cat | Task type   |
|--------|--------------------------|-----------|----------------|----------|---------|---------|-------------|
| AB     | Abalone                  |      2672 |            669 |      836 |       7 |       1 | Regression  |
| AD     | Adult ROC                |     26048 |           6513 |    16281 |       6 |       8 | Binclass    |
| BU     | Buddy                    |     12053 |           3014 |     3767 |       4 |       5 | Multiclass  |
| CA     | California Housing       |     13209 |           3303 |     4128 |       8 |       0 | Regression  |
| DI     | Diabetes                 |       491 |            123 |      154 |       8 |       0 | Binclass    |
| FB     | Facebook Comments Volume |    157638 |          19722 |    19720 |      50 |       1 | Regression  |

## B. Datasets

## B.1. Non-relational tabular data

We used six real-world datasets to assess the performance of our proposed model for generating realistic and useful synthetic tabular data. The datasets are diverse with respect to the types of variables-mix of numerical and categorical data types-as well as the number of variables in each dataset-ranging from 8 to 51 columns. The collection includes, Abalone (OpenML) 2 , Adult (income estimation) 3 , Buddy (Kaggle) 4 , California Housing (real estate data) 5 , Diabetes (OpenML) 6 , and Facebook Comments 7 . Original source, copyright, and license information are available in the links in the footnote.

We used the data splits by Kotelnikov et al. (2022) published in Tab-DDPM GitHub. Based on their pickled numpy data dumps, we recreated the splits to create data frames that we can use for our experiments with REaLTabFormer and GReaT. The latter model expects contextual input from the column names.

We also used the open-sourced optimized model parameters published in the above GitHub repo after reviewing the code, and the correctness of the code relevant to producing the assets of interest has been confirmed. We trained the TVAE, CTABGAN+, and Tab-DDPM models from scratch using the parameters on each dataset.

## B.2. Relational tabular data

To test the REaLTabFormer in modeling relational datasets, we used two real-world data: the Rossmann store sales 8 dataset and the Airbnb new user bookings 9 dataset.

We created train and test splits. For the Rossmann dataset, we used 80% of the stores data and their associated sales records for our training data. We used the remaining stores as the test data. We also limit the data used in the experiments from 2015-06 onwards spanning 2 months of sales data per store. In the Airbnb dataset, we considered a random sample of 10,000 users for the experiment. We take 8,000 as part of our training data, and we assessed the metrics and plots using the 2,000 users in the test data. We also limit the users considered to those having at most 50 sessions in the data.

## C. Reproducibility

We used be great==0.0.3 for the GReaT model. We used the Tab-DDPM GitHub repo version with this permanent link https://github.com/rotot0/tab-ddpm/tree/41f2415a378f1e8e8f4f5c3b8736521c0d47cf22. We used sdv==0.17.2 and sdmetrics==0.8.1 ; however, we fixed a bug in the HyperTransformer implementation. We used transformers==4.25.1 and torch==1.13.1 . We will open-source the REaLTabFormer package and experiments repository. We used Python version 3.9 .

We ran our experiments on a standalone workstation with the following specs: 2x AMD EPYC 7H12 64-Core Processor, 2x RTX 3090 GPU, and 1TB RAM running Ubuntu 20.04 LTS.

2 Abalone (OpenML)

3 Adult (income estimation)

4 Buddy (Kaggle)

5 California Housing (real estate data)

6 Diabetes (OpenML)

7 Facebook Comments

8 Rossmann store sales

9 Airbnb new user bookings

Table 4. Logistic detection measure for the generated parent, child, and merged tables by the Hierarchical Modeling Algorithm (HMA) from SDV and the REaLTabFormer (RTF) models. This uses the logistic regression model as the detector.

| DATASET   | TABLE               | SDV                                          | RTF                                          |
|-----------|---------------------|----------------------------------------------|----------------------------------------------|
| ROSSMANN  | PARENT CHILD MERGED | 78.67 ± 6 . 79 16.62 ± 0 . 86 12.00 ± 0 . 73 | 92.75 ± 4 . 28 59.00 ± 2 . 92 50.69 ± 2 . 41 |
| AIRBNB    | PARENT CHILD MERGED | 98.66 ± 1 . 34 0.00 ± 0 . 00 96.71 ± 2 . 79  | 99.68 ± 0 . 38 26.33 ± 0 . 78 98.93 ± 0 . 82 |

## D. Other measures and results

We also computed the logistic detection measure with the standard approach of using a logistic regression model. We find that the logistic regression model appears to not provide reliable results Table 4. In particular, the scores returned by the model are too high which is suspicious given that qualitative observation of the synthetic data hints at inaccuracies by both models in producing perfect alignment with the original data. These spurious results may be due to the model's limited capacity of learning the structure of the data. While techniques can be applied to help the model detect non-linearities better, we opted to report the results using the random forest as the base detector since it naturally is able to learn non-linearities and appears to give reasonable results.

## D.1. Joint plots

The joint plot provides a qualitative assessment of the quality of the synthetic data generated by each model. We show in the sequence of figures below the joint plots of two numerical variables in the datasets used.

Figure 5. Joint plot of two numerical variables in the Abalone data compared across the samples generated by the different models.

<!-- image -->

Figure 6. Joint plot of two numerical variables in the Adult data compared across the samples generated by the different models.

<!-- image -->

Figure 7. Joint plot of two numerical variables in the Buddy data compared across the samples generated by the different models.

<!-- image -->

Figure 8. Joint plot of two numerical variables in the California housing data compared across the samples generated by the different models.

<!-- image -->

Figure 9. Joint plot of two numerical variables in the Diabetes data compared across the samples generated by the different models.

<!-- image -->

## D.2. Distance to closest record (DCR) distribution

We present earlier the distance to closest record (DCR) distribution, Equation 2, as part of our proposed strategy to detect overfitting in training the REaLTabFormer model. Here, we use the DCR distribution to visually assess whether the generative models create exact copies of observations from the training data. We also show the DCR distribution of the real test data as a reference.

<!-- image -->

Figure 10. Distance to closest record (DCR) distributions of the different models for the Abalone data.

<!-- image -->

Figure 11. Distance to closest record (DCR) distributions of the different models for the Adult data.

<!-- image -->

Figure 12. Distance to closest record (DCR) distributions of the different models for the Buddy data.

Figure 13. Distance to closest record (DCR) distributions of the different models for the California housing data.

<!-- image -->

Figure 14. Distance to closest record (DCR) distributions of the different models for the Diabetes data.

<!-- image -->