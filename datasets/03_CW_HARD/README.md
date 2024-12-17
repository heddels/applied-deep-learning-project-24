### 03 CW_HARD

- binary classification
- Sentences were automatically extracted from wikipedia leveraging NPOV information but subsequently small subsample was
  annotated by human crowdworking (resulting in this dataset). There is ~1800 biased sentences and then 3 different
  neutral datasets: cw_hard, type_balanced, featured. We chose featured as a neutral dataset because in the paper they
  had the most success with it and also it is drawn from generally less biased articles on wikipedia.
- preprocessing steps: just cleaning and putting files together
- Domain of the labels:
    - `text`: The plain text containing a sentence from wikipedia.
    - `label`: binary bias label. `0=neutral, 1=biased`.
- final size: `6843`
- Citation Identifier: `hube_neural_2019`
- Title: `Neural Based Statement Classification for Biased Language`
-

original [repository](https://github.com/ChristophHubeL3S/Neural_Based_Statement_Classification_for_Biased_Language_WSDM2019)
