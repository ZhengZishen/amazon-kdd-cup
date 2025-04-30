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
Our updated code streamlines prompt handling by automatically switching between two lightweight templates based on task type: for multiple-choice questions, it uses a minimal “prompts_multi” template (instruction + query) with max_new_tokens=1 to yield a single-option index instantly; for open-ended tasks, it employs a rich exemplar-driven “exemplar_no_multi” template—featuring six diverse examples—paired with max_new_tokens=15 to guide concise, context-aware responses. This “less-is-more” design cuts prompt overhead, boosts accuracy, and simplifies maintenance, delivering faster, more reliable outputs tailored to each task’s complexity. 

🖊 We have tried to submit in the late submission, since AIcrowd no longer accepts uploads, we’ve hosted the code here—and in our tests it achieves even higher accuracy than before.
