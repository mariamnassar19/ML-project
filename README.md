# ML-project
## Predicting french text difficulty level
## 1 Project Overview
### 1.1 Problem Description
When learning a new language like French, it is critical for learners to engage with texts and multimedia content that are appropriately challenging—neither too easy nor too difficult. People learning French often face difficulties in finding reading materials and video content that match their specific proficiency levels, which range from beginner (A1) to advanced (C2). Traditional methods of selecting texts and videos can be subjective and inconsistent, hindering steady progress in language acquisition. Moreover, learners may struggle to identify which materials suit their current comprehension abilities, leading to potential frustration and decreased motivation. To support learners effectively, there is a pressing need for a tool that can automatically evaluate and classify the difficulty of French texts and the spoken content in French-language videos on platforms like YouTube. This project aims to develop a predictive model that assesses the difficulty level of both written texts and video subtitles in French, thereby helping learners to select materials that align with their learning needs and enhance their overall educational experience. 
### 1.2 Sustainability
The predicting French text difficulty model also aligns with the current sustainable development goals. Here’s how the app’s sustainability features correspond to specific Sustainable Development Goals (SDGs):

**SDG 4: Quality Education**<img width="30" alt="截屏2024-05-21 16 17 37" src="https://github.com/mariamnassar19/ML-project/assets/150010028/f7520e76-3064-4621-ab3a-efd7aa70aeec">         

- **Educational Sustainability:** The app enhances lifelong and autonomous learning by providing tailored language materials according to learners' proficiency levels, improving educational efficiency and reducing resource use.

**SDG 9: Industry, Innovation, and Infrastructure**<img width="30" alt="截屏2024-05-21 16 17 56" src="https://github.com/mariamnassar19/ML-project/assets/150010028/81c01ba0-24dd-48d5-9554-4cdc999f40d0">

- **Technological Sustainability:** The app continually updates with the latest language processing technologies and machine learning advancements, ensuring accuracy, relevance, and the ability to include additional languages, thus extending its market longevity.

**SDG 11: Sustainable Cities and Communities**<img width="30" alt="截屏2024-05-21 16 18 11" src="https://github.com/mariamnassar19/ML-project/assets/150010028/f5cbabbd-1750-4c32-aff1-f837cdf38a78">

- **Environmental Sustainability:** By decreasing reliance on physical educational resources and supporting remote learning, the app helps reduce paper usage, waste, and carbon emissions associated with traditional educational methods.

**SDG 10: Reduced Inequalities**<img width="30" alt="截屏2024-05-21 16 18 03" src="https://github.com/mariamnassar19/ML-project/assets/150010028/1a02a98d-ef8d-4af8-9472-a0f93b3d2710">

- **Cultural and Social Sustainability:** The app facilitates access to language learning for diverse user groups, including economically disadvantaged learners, promoting multicultural engagement and educational equity.

**SDG 8: Decent Work and Economic Growth**<img width="30" alt="截屏2024-05-21 16 17 48" src="https://github.com/mariamnassar19/ML-project/assets/150010028/d2a2a15b-6ef8-4fdf-873e-93f29c99bf21">

- **Economic Sustainability:** The app creates a sustainable revenue stream through subscriptions and licenses, funding continuous improvement and attracting investment from the educational technology sector, contributing to economic growth and sustainable employment."

### 1.3 Overall Objective

To address the challenges outlined previously, this project is divided into two main components:

**The first part** concentrates on solving the text classification issue by identifying and optimizing the best model to predict the difficulty of French texts.

**The second part** is dedicated to enhancing the user interface to ensure that the application is user-friendly and effective for learners.

## 3: Find the best model to predict the difficulty of French texts
### 3.1 Objective
This part aims to identify the most effective approach for predicting the difficulty level of French texts. To achieve this, the following steps are implemented:

**Feature Extraction:** The model processes French texts by converting them into a structured numerical format that encapsulates essential linguistic features.

**Classifier Selection:** An advanced language model specifically designed for French is utilized to interpret and analyze the complexity of the texts.

**Difficulty Prediction:** The trained model evaluates and classifies the difficulty level of French texts, optimizing learning paths for users at various proficiency levels.

###### Data Preparation

For LogoRank, effective data preparation is crucial for training our models to accurately predict the difficulty level of French texts. Below is an overview of our data preparation process:
**Data Augmentation**

To enhance the robustness of our models and increase the diversity of our training examples, we implemented a data augmentation strategy. Our approach involved the following steps:

- 1. Sentence Swapping: Each sentence in the dataset was swapped within its document while preserving the original structure of the text. This method maintains the semantic integrity of the document and ensures that the augmented sentences retain the same difficulty level as the original.

- 2. Appending Augmented Data: Each swapped sentence was then appended to the training dataset as a new data point. This effectively increased the size of our training dataset, allowing our models to learn from a broader array of sentence structures and contexts without introducing label noise.

**Rationale**

The decision not to clean the data was based on an initial assessment which suggested that the dataset was already in good form without major inconsistencies or errors. This approach allowed us to focus our efforts on model training and refinement.

Through this data preparation process, we aimed to create a training environment that mimics real-world applications, where the variety of text structures and contexts is vast. This method prepares our models to perform well across different styles and complexities of French texts.

### 3.3 Not Good Model - Feature Extraction and Difficulty Prediction with Camembert
#### 3.3.1 Overview

This model utilizes the Camembert BERT model to extract features from French sentences and predict their difficulty levels using a classifier trained on different language complexity data. It is designed to assess the proficiency level required to comprehend each sentence, categorizing them from A1 (easiest) to C2 (most difficult) based on the Common European Framework of Reference for Languages (CEFR).

#### 3.3.2 Feature Extraction
The `bert_feature` function is responsible for converting French sentences into a numerical format that represents linguistic features. This is achieved by:
- Tokenizing the sentences using the `CamembertTokenizer` from the Hugging Face `transformers` library.
- Feeding these tokens into the `CamembertModel` to obtain embeddings.
- Calculating the average of these embeddings across all tokens produces a single feature vector per sentence. This averaging method incorporates the contributions of each word, ensuring a balanced representation that reflects the entirety of the sentence. `feature = np.mean(model(inputs_embeds=input_embeds)[0][:, :, :].cpu().numpy(), axis=1)` This approach is particularly valuable in tasks like difficulty prediction, where the context and complexity of each word collectively determine the overall difficulty level of the sentence.
- `X = bert_feature(X)`Then, display the outcome after the feature extraction.`pd.DataFrame(X)`
<img width="1252" alt="截屏2024-05-21 15 50 06" src="https://github.com/mariamnassar19/ML-project/assets/150010028/0e40ef1c-d4d5-4849-ad49-7ae0082d19a4">

#### 3.3.3 Select Classification 
After extracting features, we use these features to test various pre-trained classifiers, including SVC, Logistic Regression, Random Forest, Extra Trees, Decision Tree, LightGBM, CatBoost, and MLP. We assess their performance on a test set, which is derived from the original dataset.

We have split the dataset into a training set (x_train and y_train) and a test set (x_test and y_test) in a 9:1 ratio. The target variables have been numerically encoded using LabelEncoder to represent the six difficulty levels: A1, A2, B1, B2, C1, and C2 as numbers 0 through 5.

Additionally, we implemented an `evaluation` function to measure each model's effectiveness on the test set. This function calculates accuracy, precision, recall, and F1 scores to assess the models' overall accuracy, precision in identifying correct classifications, completeness of the positive predictions, and the balance between precision and recall. The comparison results are as follows:
<img width="930" alt="截屏2024-05-21 15 54 10" src="https://github.com/mariamnassar19/ML-project/assets/150010028/1f95a852-654b-4038-ae30-a214f4f65747">

#### 3.3.4 Discussion and Conclusion
Despite our MLP model achieving high accuracy on our test set (approximately 0.8), its performance dropped significantly to 0.553 when predicting unlabeled test data in a Kaggle competition. Notably, other models also showed similar accuracy around 0.5 on Kaggle.

We find that the model demonstrates strong performance on the test data but significantly underperforms on the diverse and unseen data from Kaggle, suggesting overfitting to the training data’s specific characteristics and noise. This means it's really good at handling the specific types of data it trained on but struggles with new, different data from Kaggle. The likely cause of this issue is that our training data might not represent the wider variety of data on Kaggle, particularly in terms of how the labels are distributed. Our training data could have more of certain types of labels, which biases the model towards these labels and doesn't prepare it well for the more balanced labels in the Kaggle dataset. 

Therefore, due to these issues, we ultimately decided to abandon the "Not Good Model - Feature Extraction and Difficulty Prediction with Camembert." This decision was driven by the model's poor adaptability to new and diverse datasets, as evidenced by its performance in the Kaggle competition.

### 3.4 Comparison of Model Approaches

Upon evaluating different models for predicting the difficulty of French texts, we observed significant differences in their performances. Initially, we considered using the Camembert-based method for its advanced linguistic understanding capabilities. However, our comparative analysis showed that while Camembert excelled in controlled test environments, it was prone to overfitting and had scalability issues when applied to broader, more diverse datasets such as those found in the Kaggle competition.

In contrast, the FlaubertForSequenceClassification model demonstrated a higher accuracy of 0.640 on the Kaggle platform. Unlike Camembert, Flaubert proved to be better suited for this specific task due to its robustness across different data distributions and its more stable performance in varied testing conditions. This model provided a more balanced approach, efficiently handling the nuances of text difficulty classification without the computational intensity and overfitting concerns associated with Camembert.

### 3.5  Results and Discussion

The decision to switch to the FlaubertForSequenceClassification model was further validated by its outstanding performance on the Kaggle dataset, where it achieved an accuracy of 0.640. This not only demonstrated the model's practical effectiveness in real-world applications but also led us to achieve a remarkable **second-place** finish in the Kaggle competition. 😄🎉
<img width="1045" alt="截屏2024-05-21 16 06 28" src="https://github.com/mariamnassar19/ML-project/assets/150010028/c02fd211-6e9a-4060-a5d9-e6c16a19c410">

The Flaubert model's success can be attributed to several factors:
- **Generalization**: Unlike Camembert, Flaubert was less sensitive to noise and specific training data patterns, which translated into better generalization on unseen data.
- **Computational Efficiency**: Flaubert required less computational resources, which allowed us to refine and iterate the model more rapidly, adjusting to the Kaggle competition's demands effectively.
- **Adaptability**: The model adapted well to the scoring metrics and conditions of the competition, indicating a strong alignment between the model’s output and the evaluation criteria used.

Our findings emphasize the importance of ongoing evaluation and adaptation in machine learning projects. Adjusting strategies based on model performance comparisons leads to better-informed decisions and more effective deployments.

In conclusion, choosing the FlaubertForSequenceClassification model over the Camembert approach marks a strategic pivot towards practical deployment scenarios. This decision highlights the need for flexibility in model selection within the dynamic field of AI and machine learning;)





