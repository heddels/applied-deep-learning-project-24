### 103 MPQA

- binary classification
- Dataset has annotations for sentiment as well as subjectivity (eg expressive subjectivity, implicit, etc). With many
  fields regarding annotators certainity etc. Original corpus was very complex, I refer to the original disertation
  thesis where corpus was introduced.
- Preprocessing steps done:
    - loading and matching docs with its annotations
    - extracting annotations spans that included 'attitude-type' field
    - matching these annotation spans with sentence spans such:
        - sentence is extracted as annotated if there is an annotation span INSIDE the sentence span
        - sentence is labeled according to the label of this annotation. If multiple annotations spans inside the
          sentence, majority vote label is selected. If there is conflict (50-50 labels) the sentence is discarded
    - finally the text of the sentences is cleaned via some regexes.
- Original dataformat: two directories: docs and annotations. docs contained plain text of article. Annotations
  consisted of two files: sentences, where spans of the whole sentences were, and annotations where spans of annotations
  and fields of annotations itself were (eg: span:12,56 fields:polarity=neutral,attitude=sentiment-pos,...)
- Domain of the labels:
    - `text`: The plain text containing a sentence.
    - `label`: The binary label. `0=positive, 1=negative`.
- final size: `3582`
- Citation Identifier: `wilson_fine-grained_2008`
- Title:
  `Fine-grained Subjectivity and Sentiment Analysis: Recognizing the intensity, polarity, and attitudes of private states`