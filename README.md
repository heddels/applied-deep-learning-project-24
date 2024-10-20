# Applied Deep Learning: Project Proposal

## Project Overview
In this project, I aim to tackle the challenge of media bias detection, particularly in an age where information 
overload and cognitive biases make it increasingly difficult for individuals to critically analyze the media they consume. 
Drawing on established research in automated bias detection, I plan to build a tool that helps users identify 
potential biases in news articles, primarily as a reminder to remain critical without replacing personal judgment.


### Idea and Approach
The project will be a mix of Bring Your Own Method and Bring Your Own Data. Existing models, such as MAGPIE 
(Horych et al., 2024) and earlier works like form Spinde et al. (2022), have made significant advancements in 
automated media bias detection, particularly for English-language data. My approach involves adapting these models 
for German-language news by filtering the Large Bias Mixture (LBM) dataset and simplifying the architecture to 
make it computationally feasible.

Key modifications include:

 - Replacing MAGPIE’s pre-trained model with DistilBERT, which is more efficient and suitable for the available 
computational resources.
- Simplifying and redesigning MAGPIE’s architecture, leveraging the foundational work of Spinde et al. (2022), 
whose model uses a simpler framework.

To present my work, I aim to build a simple interface where users can input text and receive bias detection results. 
Menzner & Leidner (2024) developed a similar interface using GPT-3.5. However, I aim to implement the architecture 
described above for more targeted and resource-efficient bias detection. 


### The Dataset
As mentioned I want to use the LBM data set filtered for German articles.However, I have to find those datasets 
also available in German language and possibly add other existing datasets o the collection. Therefore, 
I take the listed datasets from the MAGPIE paper as a starting point and search for German versions of them.


### Work Breakdown Structure
Table 1: Work Breakdown Structure

| Task | Time Estimate (hrs) | Due Date |
| --- |---------------------|----------|
| Dataset Collection | 8-18                | 27.10.24 |
| Designing and Building an Appropriate Network | 15-20               | 17.11.24 |
| Training and Fine-tuning that Network | 20-25               | 17.12.24 |
| Building an Application to Present the Results | 15-20               | 05.01.25 |
| Writing the Final Report | 10                  | 19.01.25 |
| Preparing the Presentation of Your Work | 5-10                | 28.01.25 |


## References
1. Horych, T., Wessel, M., Wahle, J. P., Ruas, T., Waßmuth, J., Greiner-Petter, A., Aizawa, A., Gipp, B., & Spinde, T. (2024). 
**Magpie: Multi-task media-bias analysis generalization for pre-trained identification of expressions.**
[Link to the paper](https://arxiv.org/abs/2403.07910),
[Link to the Github implementation](https://github.com/Media-Bias-Group/magpie-multi-task)

2. Menzner, T., & Leidner, J. L. (2024). BiasScanner: Automatic detection and classification of news bias to strengthen democracy. arXiv. 
[Link to the paper](https://arxiv.org/abs/2407.10829),
[Link to the implementation web page](https://biasscanner.org/#links)

3. Rodrigo-Ginés, F.-J., Carrillo-de-Albornoz, J., & Plaza, L. (2024). **A systematic review on media bias detection: What is media bias, how it is expressed,
and how to detect it.** Expert Systems with Applications, 237, 121641.
[Link to the paper](https://doi.org/10.1016/j.eswa.2023.121641)

4. Spinde, T., Hinterreiter, S., Haak, F., Ruas, T., Giese, H., Meuschke, N., & Gipp, B. (2024). **The media bias taxonomy:
A systematic literature review on the forms and automated detection of media bias.** 
[Link to the paper](https://arxiv.org/abs/2312.16148)

5. Spinde, T., Krieger, J.-D., Ruas, T., Mitrovi´c, J., G¨otz-Hahn, F., Aizawa, A., & Gipp, B. (2022). **Exploiting transformer-based multitask learning
for the detection of media bias in news articles.** In Information for a better world: Shaping the global future (pp. 225–235). Springer International Publishing. 
[Link to the paper](https://arxiv.org/abs/2211.03491), 
[Link to the Github implementation](https://github.com/Media-Bias-Group/Exploiting-Transformer-based-Multitask-Learning-for-the-Detection-of-Media-Bias-in-News-Articles)




   
   

