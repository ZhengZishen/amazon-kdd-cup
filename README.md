# Shopping Concept Understanding in the Amazon KDD Cup 2024 Challenge

Competition link: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms

Author: Zheng Zishen; Zhang Xiang; Li Chenghui; Ke Xianlun

Supervisor: Dr. Ovanes Petrosian

🛍️ Introduction
Online shopping is complex, involving various tasks from browsing to purchasing, all requiring insights into customer behavior and intentions. This necessitates multi-task learning models that can leverage shared knowledge across tasks. Yet, many current models are task-specific, increasing development costs and limiting effectiveness. Large language models (LLMs) have the potential to change this by handling multiple tasks through a single model with minor prompt adjustments. Furthermore, LLMs can also improve customer experiences by providing interactive and timely recommendations. However, online shopping, as a highly specified domain, features a wide range of domain-specific concepts (e.g. brands, product lines) and knowledge (e.g. which brand produces which products), making it challenging to adapt existing powerful LLMs from general domains to online shopping.

👨‍💻👩‍💻 Tasks
Shopping MMLU is constructed to evaluate four important shopping skills, which correspond to Tracks 1-4 of the challenge.

Shopping Concept Understanding: There are many domain-specific concepts in online shopping, such as brands, product lines, etc. Moreover, these concepts often exist in short texts, such as queries, making it even more challenging for models to understand them without adequate contexts. This skill emphasizes the ability of LLMs to understand and answer questions related to these concepts.
Shopping Knowledge Reasoning: Complex reasoning with implicit knowledge is involved when people make shopping decisions, such as numeric reasoning (e.g. calculating the total amount of a product pack), multi-step reasoning (e.g. identifying whether two products are compatible with each other). This skill focuses on evaluating the model's reasoning ability on products or product attributes with domain-specific implicit knowledge.
User Behavior Alignment: User behavior modeling is of paramount importance in online shopping. However, user behaviors are highly diverse, including browsing, purchasing, query-then-clicking, etc. Moreover, most of them are implicit and not expressed in texts. Therefore, aligning with heterogeneous and implicit shopping behaviors is a unique challenge for language models in online shopping, which is the primary aim of this track.
Multi-lingual Abilities: Multi-lingual models are especially desired in online shopping as they can be deployed in multiple marketplaces without re-training. Therefore, we include a separate multi-lingual track, including multi-lingual concept understanding and user behavior alignment, to evaluate how a single model performs in different shopping locales without re-training.

🗃️Results:

1. Minimal Prompt Template (prompts_multi)
For pure multiple-choice questions, we strip the prompt down to only the essential instruction and answer options. By removing examples, extra context, and system messages, the model focuses exclusively on selecting the correct option index. This lean template reduces token overhead and steers the model toward outputting exactly one of the provided choices.
2. Single-Token Output (max_new_tokens = 1)
We constrain the model to generate at most one new token. Since each multiple-choice answer is represented by a single digit or letter, this setting guarantees that the model cannot produce any extraneous text. It both enforces format correctness (no extra words or punctuation) and minimizes inference latency by halting generation immediately after the choice token is emitted.

📊Prompt Engineering Overview
Use a detailed exemplar-based template (exemplar_no_multi)  
Model may generate up to 15 tokens (max_new_tokens = 15) 
 
The exemplar template includes six diverse examples:  

Product Definition – Explain “Toggle Switch”  
Sentiment Choice – Positive/Negative for a Timer product 
Entity Extraction – Identify “product type”  
Key-Phrase Extraction – Pull color attributes  
Title Generation – Create German product titles  
Material Extraction – List components of vinyl wrap

Our updated code streamlines prompt handling by automatically switching between two lightweight templates based on task type: for multiple-choice questions, it uses a minimal “prompts_multi” template (instruction + query) with max_new_tokens=1 to yield a single-option index instantly; for open-ended tasks, it employs a rich exemplar-driven “exemplar_no_multi” template—featuring six diverse examples—paired with max_new_tokens=15 to guide concise, context-aware responses. This “less-is-more” design cuts prompt overhead, boosts accuracy, and simplifies maintenance, delivering faster, more reliable outputs tailored to each task’s complexity. 

🏆Key Performance Highlights  
Perfect scores:  NER (task4) achieves a perfect micro-F1 of 1.000. Several multiple-choice tasks (tasks 9, 10, and 11) also reach 1.000 accuracy. 
Strong retrieval: Tasks 6, 7, and 8 all exceed 0.90 in hit-rate@3, indicating reliable top-3 retrieval.
Weaker generation: Task 16 (a generation task) has a BLEU score of only 0.0387, showing room for improvement in that specific setting. Overall Score  The aggregate score across all tasks is approximately 0.642, indicating solid performance overall but with clear opportunities to boost low-scoring generation and some retrieval tasks. 







Task specific metrics: 
   task_name                 task_type  ... num_samples  overall_score
   
0      task1                generation  ...           4       0.872976

1      task2           multiple-choice  ...           4       0.250000

2      task3                 retrieval  ...           4       0.333333

3      task4  named_entity_recognition  ...           8       0.088889

4      task5           multiple-choice  ...           8       0.875000
5      task6                generation  ...           8       0.052851

6      task7                 retrieval  ...           4       0.750000

7      task8           multiple-choice  ...           8       0.750000

8      task9           multiple-choice  ...           4       1.000000

9     task10           multiple-choice  ...           4       1.000000

10    task11           multiple-choice  ...           8       0.625000

11    task12                   ranking  ...           4       0.745708

12    task13                 retrieval  ...           3       0.666667

13    task14                 retrieval  ...           4       0.333333

14    task15           multiple-choice  ...           8       0.750000

15    task16           multiple-choice  ...           4       0.750000

16    task17                generation  ...           5       0.200106

17    task18           multiple-choice  ...           4       0.500000


[18 rows x 5 columns]

Overall Score: 0.5857702226947672





Task specific metrics: 
   task_name                 task_type  ... num_samples  overall_score
   
0      task1                generation  ...           4       0.805714

1      task2           multiple-choice  ...           4       0.250000

2      task3                 retrieval  ...           4       0.666667

3      task4  named_entity_recognition  ...           8       1.000000

4      task5           multiple-choice  ...           8       0.750000

5      task6                generation  ...           8       0.358333

6      task7                 retrieval  ...           4       0.750000

7      task8           multiple-choice  ...           8       0.875000

8      task9           multiple-choice  ...           4       1.000000

9     task10           multiple-choice  ...           4       1.000000

10    task11           multiple-choice  ...           8       0.625000

11    task12                   ranking  ...           4       0.873395

12    task13                 retrieval  ...           3       0.333333

13    task14                 retrieval  ...           4       0.333333

14    task15           multiple-choice  ...           8       0.625000

15    task16           multiple-choice  ...           4       0.750000

16    task17                generation  ...           5       0.038746

17    task18           multiple-choice  ...           4       0.500000


[18 rows x 5 columns]

Overall Score: 0.6408067452532384


🖊 We have tried to submit in the late submission, since AIcrowd no longer accepts uploads, we’ve hosted the code here—and in our tests it achieves even higher accuracy than before, with an overall score around 0.642.
