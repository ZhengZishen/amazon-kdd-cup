from typing import List, Union
import random
import os

from .base_model import ShopBenchBaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class DummyModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        """
        Generates a prediction based on the input prompt and task type.

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Args:
            prompt (str): The input prompt for the model.
            is_multiple_choice (bool): Indicates whether the task is a multiple choice question.

        Returns:
            str: The prediction as a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        possible_responses = [1, 2, 3, 4]

        if is_multiple_choice:
            # Randomly select one of the possible responses for multiple choice tasks
            return str(random.choice(possible_responses))
        else:
            # For other tasks, shuffle the possible responses and return as a string
            random.shuffle(possible_responses)
            return str(possible_responses)
            # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
            # For generation tasks, this should ideally return an unconstrained string.




# class Llama3ZeroShot(ShopBenchBaseModel):
#     def __init__(self):
#         random.seed(AICROWD_RUN_SEED)
#         model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
#         self.tokenizer = AutoTokenizer.from_pretrained('./models/Meta-Llama-3-8B-Instruct/', trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained('./models/Meta-Llama-3-8B-Instruct/', device_map='auto', torch_dtype='auto', trust_remote_code=True, do_sample=True)
#         # self.model.cuda()
#         self.system_prompt = ("""You are a helpful online shopping assistant. Please answer the following question about
#                               online shopping and give the output or answer. Note that, you should, and only should,
#                               give direct answers to relevant questions and then conclude your answer immediately,
#                               without any pleasantry, explanation, reminder, etc.. \nThe given question:\n""")
#
#     def predict(self, prompt: str, is_multiple_choice: bool) -> str:
#         prompt = self.system_prompt + prompt
#         inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
#         # inputs.input_ids = inputs.input_ids.cuda()
#         generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=10, temperature=0.02)
#         result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         generation = result[len(prompt):].strip().split("\n")[0]
#
#         # print("\n\n===========\nPrompt: {}\n---------------------\nGeneration: {}\n===========\n\n".format(prompt, generation))
#         if is_multiple_choice:
#             generation = generation.strip()[0]
#             if not generation.isdigit():
#                 generation = "1"
#         return generation

from vllm import LLM, SamplingParams

class vllm_Llama3ZeroShot_v3(ShopBenchBaseModel):
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)
        self.exampler = """
        Question: Instructions: Tell me what this product category is about\nInput: Toggle Switch\nOutput:\n
        A toggle switch is an electric switch operated by means of a projecting lever that is moved up and down.\n

        Question (multiple choice question): Which of the following product categories may have the attribute power source?\n0. table\n1. writing tools\n2. car seat cover\n3. comb\nAnswer: \n
        3\n

        Question: A customer has bought a(n) timer product and wants to write a review to express a positive sentiment on the performance aspect. \nYou are given a numbered list of 15 potential review snippets. Please select 3 snippets from the list that the customer is most likely to write in his review. \nYou should output three numbers, separated with comma. Generate only indices. Do not include review snippets in your answer. Do not give explanations. \nReview Snippet List: \n1. this matches perfectly\n2. Lightweight\n3. paper quality is nice\n4. affordable\n5. looks great\n6. great bang for your buck\n7. wrapped to shreds\n8. work wonderfully\n9. coating seems very durable\n10. gorgeous\n11. well made\n12. good corner toaster with quality features\n13. timer works great\n14. it does the job\n15. soft to the ear sound\nOutput: \n
        14,13,8\n

        Question: You are a helpful online shop assistant and a linguist. A customer on an online shopping platform has made the following query. Please extract phrases from the query that correspond to the entity type 'product type'. \nPlease directly output the entity without repeating the entity type. If there are multiple such entities, separate them with comma. Do not give explanations. \nQuery: tablette asus\nOutput: \n
        tablette\n

        Question (multiple choice question): The product 'APRATIM Women's Cotton Bandhani Dupatta With Mirror Work Free Size Blue' appears on an e-commerce website.  What type of fabric is used in it?\n0. spandex, polyester\n1. cotton\n2. microfiber\n3. It cannot be inferred.\nAnswer: \n
        1\n

        Question: You are given a user review to a(n) shoes product and an aspect covered in the review. \nPlease extract the keyphrase from the review that mentions the given aspect. \nYou should only extract ONLY ONE keyphrase from the review. \nYou should not generate new keyphrases that do not exist in the review.\nDo not give explanations or irrelevant text. Generate short keyphrases, not whole sentences.  \nReview: The stitches started ripping the 2nd day my 2 and 3 year old wore them... other than that they were comfy and cute for the price.\nAspect: color\nOutput: \n\n
        cute for the price\n

        Question: \nYou are given a user review given to a(n) bra product. You are also given a numbered list of ten aspects. \nPlease choose three aspects from the list that are covered by the review. \nYou should ONLY output three numbers, separated by comma. Do not generate explanations or other texts. \nReview: \nVery comfortable and supportive, as a 38D it\u2019s hard to find a good bra. True to size\nAspect List: \n1.stability\n2.magnet strength\n3.straps\n4.lid\n5.hook\n6.comfort\n7.value\n8.support\n9.quality\n10.fit\nOutput: \n
        6,8,10\n
        """
        # Q2: \n
        # A2: \n
        # MC2: \n

        self.prompts = """
            You are a helpful online shopping assistant who can give reasonable answers or outputs after the online shopping questions. 
            This is your Q&A history, please continue to answer questions concisely until
            the answer or output of the last question is finished.\n
            {}
            """.format(self.exampler)
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # submission
        self.model = LLM(model="./models/Llama-3.2-1B-Instruct/", trust_remote_code=True, dtype='float16', tensor_parallel_size=2, enforce_eager=True)
        # baichuan
        # self.model = LLM(model="./models/Meta-Llama-3-8B-Instruct/", trust_remote_code=True, dtype='float16',
        #                  tensor_parallel_size=1, enforce_eager=True)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        mc = " (multiple choice question)" if is_multiple_choice else ""
        new_prompt = """
        Question{}: {}
        """.format(mc, prompt)
        prompts = self.prompts + new_prompt
        outputs = self.model.generate(prompts, self.sampling_params)
        generation = outputs[0].outputs[0].text.strip().split("\n")[0].strip()
        if "Question" in generation:
            generation = generation.split("Question")[0].strip()
        if len(generation) == 0:
            generation = prompt
        if is_multiple_choice:
            generation = generation[0]
            if not generation.isdigit():
                print("Generation not digital!!!")
                generation = "1"
        # baichuan
        # print("\n\n===========\nPrompt: {}\n---------------------\nGeneration: {}\n===========\n\n".format(prompt,
        #                                                                                                   generation))
        return generation