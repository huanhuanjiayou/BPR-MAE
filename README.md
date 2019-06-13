# BPR-MAE
**Neural Fashion Experts: I Know How to Make the Complementary Clothing Matching**

![image](https://github.com/huanhuanjiayou/BPR-MAE/tree/master/img/framework.png)
Fig. 1. Illustration of the proposed BPR-MAE scheme. We employ an improved multiple autoencoder network to learn the latent compatibility space, where we jointly model the coherent relationship between the visual and textual modalities and the implicit preferences among items via the Bayesian Personalized Ranking. “>” indicates the category hierarchy.

Abstract：
  In modern society, clothing has gradually become the beauty enhancing product rather than a basic necessity. In fact, the key to a proper outfit usually lies in the harmonious clothing matching. However, not everyone has a good taste in clothing matching, which thus propells several efforts in the fashion domain. Nevertheless, the existing clothing matching techniques mainly rely on the visual features but overlook the textual metadata, which may be insufficient to comprehensively understand fashion items. Fortunately, nowadays fashion experts are enabled to share their fashion tips by showcasing their outfit compositions on fashion-oriented online communities. Each outfit usually consists of several complementary fashion items (e.g., a top, a bottom and a pair of shoes), which usually involves a visual image along with certain textual metadata (e.g., the title and categories). The rich fashion data provide us an opportunity for the clothing matching, especially the complementary fashion item matching. In this work, we propose a multiple autoencoder neural network based on the Bayesian Personalized Ranking, dubbed BPR-MAE. This framework is able to not only comprehensively model the compatibility between fashion items (e.g., tops and bottoms, bottoms and shoes) but also fulfill the complementary fashion item matching among multiple fashion items by seamlessly exploring the multi-modalities (i.e., the visual and textual modalities) of fashion items. Experimental results on the real-world dataset demonstrate the effectiveness of our proposed model, based on which we provide certain deep insights that can benefit the future research.

Contributions：
*• We present a multiple autoencoder neural network, which is able to not only model the compatibility between fashion items, but also accomplish the complementary clothing matching by seamlessly exploring the multi-modalities (i.e., the visual and textual modalities) of fashion items.

*• We propose a content-based neural scheme (BPR-MAE) based on the BPR framework, which is able to learn the latent compatibility space and bridge the semantic gap between fashion items from heterogeneous spaces.

*• We construct the fashion dataset FashionVC+, which consists of both images and textual metadata of fashion items (i.e., tops, bottoms and shoes) on Polyvore. We have released our code and parameters to allow other researchers to repeat our experiments and verify their approaches.

Datasets：
*  In this work, we construct the fashion dataset FashionVC+ to evaluate our proposed model (BPR-MAE). FashionVC+ consists of 20,726 outfits with 14,871 tops, 13,663 bottoms and 14093 pairs of shoes, collected from the online fashion community Polyvore, where a great amount of outfit compositions shared by fashion experts are publicly available. Each fashion item (i.e., top, bottom and shoes) in FashionVC+ is associated with a visual image, categories and title description. 

Code：
*• BPR_MAE.py: To implement the complementary clothing matching. 

Environment requirements：
*  The code is written in Python (2.7) and Theano (0.9).
