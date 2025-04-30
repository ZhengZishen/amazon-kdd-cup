# Shopping Concept Understanding in the Amazon KDD Cup 2024 Challenge

Author: Zheng Zishen; Zhang Xiang; Li Chenghui; Ke Xianlun

Supervisor: Dr. Ovanes Petrosian

Our updated code streamlines prompt handling by automatically switching between two lightweight templates based on task type: for multiple-choice questions, it uses a minimal “prompts_multi” template (instruction + query) with max_new_tokens=1 to yield a single-option index instantly; for open-ended tasks, it employs a rich exemplar-driven “exemplar_no_multi” template—featuring six diverse examples—paired with max_new_tokens=15 to guide concise, context-aware responses. This “less-is-more” design cuts prompt overhead, boosts accuracy, and simplifies maintenance, delivering faster, more reliable outputs tailored to each task’s complexity. Since AIcrowd no longer accepts uploads, we’ve hosted the code here—and in our tests it achieves even higher accuracy than before.
