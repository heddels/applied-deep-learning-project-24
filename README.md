# Applied Deep Learning: Project Proposal

## Project Overview
In this project, I aim to tackle the challenge of media bias detection, particularly in an age where information 
overload and cognitive biases make it increasingly difficult for individuals to critically analyze the media they consume. 
Drawing on established research in automated bias detection, I plan to build a model that helps identify 
potential biases in news articles, primarily as a reminder to remain critical without replacing personal judgment.


## Idea and Approach
The project will be a mix of *Bring Your Own Method* and *Bring Your Own Data*. Existing models, such as MAGPIE 
(Horych et al., 2024) and earlier works like form Spinde et al. (2022), have already made significant advancements in 
automated media bias detection with deep learning models, particularly for English-language data. 
My approach involves adapting these models by simplifying the architecture to make it computationally more feasible 
while trying to preserve accuracy.

Key modifications include:

- Replacing MAGPIE’s pre-trained model encoder RoBERTa with DistilBERT, which is more efficient and suitable for the available computational resources.
- Simplifying and redesigning MAGPIE’s architecture, leveraging the foundational work of Spinde et al. (2022), whose model uses a simpler framework.
- Reducing the number of datasets and tasks in the multitask learning setup to decrease the computational load.

To present my work, I aim to build a simple interface where users can input text and receive bias detection results. 
Menzner & Leidner (2024) developed a similar interface using GPT-3.5. However, I will implement the architecture 
described above for a more targeted and resource-efficient bias detection solution.


## The Dataset
As mentioned above, in order for the project to be feasible, I will need to simplify the data for the multitask learning setup.
Therefore, I will use the following  datasets from the MAGPIE paper, being the smallest datasets in each task family, 
except for the News bias task family:

instead of first (or in addition to, to also have regression task?):
CW_HARD (Hube and Fetahu, 2019) 6.843 Binary Classification

| Task Family| Dataset | # sentences | Task |
|--------------------|-----------------------------------------|--------|----------------------------|
| Subjective bias    | NewsWCL50 (Hamborg et al., 2019)        | 731    | Regression                 |
| News bias          | BABE (Spinde et al., 2021c)             | 3,672  | Token-Level Classification |
| Hate speech        | MeTooMA (Gautam et al., 2020)           | 7,388  | Binary Classification      |
| Gender bias        | GAP (Webster et al., 2018)              | 4,373  | Binary Classification      |
| Sentiment analysis | MDGender (Din et al., 2020)             | 2,332  | Binary Classification      |
| Fake news          | MPQA (Wilson, 2008)                     | 5,508  | Token-Level Classification |
| Emotionally        | GoodNewsEveryone (Bostan et al., 2020)  | 4,428  | Token-Level Classification |
| Group bias         | StereotypeDataset (Pujari et al., 2022) | 2,208  | Binary Classification      |
| Stance detection   | GWSD (Luo et al., 2020)                 | 2,010  | Multi-Class Classification |

The datasets are available in the MAGPIE repository, and I will use the same data preprocessing steps as in the original paper.

However, the final choice of datasets is subject to change based on the computational resources available and the 
model's performance during the hacking phase of the project.

## Work Breakdown Structure
The following table outlines the tasks and their respective time estimates and due dates for the project:

| Task | Time Estimate (hrs) | Due Date |
| --- |---------------------|----------|
| Dataset Collection | 2                   | 24.10.24 |
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

############################################################################################################


### From End to end Deep Learning Project Implementation

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional] # secret credentials that I do not want to share
3. Update params.yaml
4. Update the entity 
5. Update the configuration manager in src config 
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml





## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtube.com/playlist?list=PLkz_y24mlSJZrqiZ4_cLUiP0CBN5wFmTb&si=zEp_C8zLHt1DzWKK)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)


Run this to export as env variables:
for tokens: https://dagshub.com/user/settings/tokens

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/heddels/applied-deep-learning-project-24.mlflow

export MLFLOW_TRACKING_USERNAME=heddels 

export MLFLOW_TRACKING_PASSWORD=1fbf32fd355adaa752281459f4516b0d37b109eb
  
```

dlproject-mlflow-buc 



### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


   
   

