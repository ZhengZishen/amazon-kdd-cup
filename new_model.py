import os
import random
from typing import Any, Dict, List

import vllm

from .base_model import ShopBenchBaseModel

#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 2 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.95 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class Qwen_3B_ZeroShotModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()
        self.exampler_no_multi = """
                Question: Instructions: Tell me what this product category is about\nInput: Toggle Switch\nOutput:\n
                A toggle switch is an electric switch operated by means of a projecting lever that is moved up and down.\n

                Question: A customer has bought a(n) timer product and wants to write a review to express a positive sentiment on the performance aspect. \nYou are given a numbered list of 15 potential review snippets. Please select 3 snippets from the list that the customer is most likely to write in his review. \nYou should output three numbers, separated with comma. Generate only indices. Do not include review snippets in your answer. Do not give explanations. \nReview Snippet List: \n1. this matches perfectly\n2. Lightweight\n3. paper quality is nice\n4. affordable\n5. looks great\n6. great bang for your buck\n7. wrapped to shreds\n8. work wonderfully\n9. coating seems very durable\n10. gorgeous\n11. well made\n12. good corner toaster with quality features\n13. timer works great\n14. it does the job\n15. soft to the ear sound\nOutput: \n
                14,13,8\n

                Question: You are a helpful online shop assistant and a linguist. A customer on an online shopping platform has made the following query. Please extract phrases from the query that correspond to the entity type 'product type'. \nPlease directly output the entity without repeating the entity type. If there are multiple such entities, separate them with comma. Do not give explanations. \nQuery: tablette asus\nOutput: \n
                tablette\n


                Question: You are given a user review to a(n) shoes product and an aspect covered in the review. \nPlease extract the keyphrase from the review that mentions the given aspect. \nYou should only extract ONLY ONE keyphrase from the review. \nYou should not generate new keyphrases that do not exist in the review.\nDo not give explanations or irrelevant text. Generate short keyphrases, not whole sentences.  \nReview: The stitches started ripping the 2nd day my 2 and 3 year old wore them... other than that they were comfy and cute for the price.\nAspect: color\nOutput: \n\n
                cute for the price\n

                Question: A product entitled 'Steadtler Fimo Soft Starter Pack 12 x 57 g Multicolour Blocks' exists on an online shopping website. Generate an adequate title for the product when it appears on a(n) German online shopping website.\nOutput: \n
                Fimo Soft Starter Pack 12 x 56g Multicolour Blocks by Steadtler\n

                Question: You are a helpful online shop assistant and a linguist. A customer on an online shopping platform has made the following query. Please extract phrases from the query that correspond to the entity type 'composition'. \nPlease directly output the entity without repeating the entity type. If there are multiple such entities, separate them with comma. Do not give explanations. \nQuery: 3m 1080 vinyl wrap\nOutput: \n
                vinyl\n
                """

        #  self.exampler_multi = """
        #          Question:Which of the following product categories may have the attribute eu spare part availability duration?\n0. mouse pad\n1. leash\n2. surveillance camera\n3. garbage bin\n
        #          0\n
        
        #          Question:The product 'APRATIM Women's Cotton Bandhani Dupatta With Mirror Work Free Size Blue' appears on an e-commerce website.  What type of fabric is used in it?\n0. spandex, polyester\n1. cotton\n2. microfiber\n3. It cannot be inferred.\nAnswer:1\n
        
        #          Question:A customer has bought a(n) timer product and wants to write a review to express a positive sentiment on the performance aspect. \nYou are given a numbered list of 15 potential review snippets. Please select 3 snippets from the list that the customer is most likely to write in his review. \nYou should output three numbers, separated with comma. Generate only indices. Do not include review snippets in your answer. Do not give explanations. \nReview Snippet List: \n1. this matches perfectly\n2. Lightweight\n3. paper quality is nice\n4. affordable\n5. looks great\n6. great bang for your buck\n7. wrapped to shreds\n8. work wonderfully\n9. coating seems very durable\n10. gorgeous\n11. well made\n12. good corner toaster with quality features\n13. timer works great\n14. it does the job\n15. soft to the ear sound\n
        #          [14,13,8]\n
        
        #          Question:The product 'Simply Asia Garlic Basil Singapore Street Noodles, 9.24 oz (Pack of 6)' appears on e-commerce website. What is the total weight of the noodles?\n0. 8 ounce\n1. 55.44 ounce\n2. 14.19 ounce\n3. 60 ounce\n
        #          1\n
        
        #          Question:A user on an online shopping website has just purchased a product 'Steven Harris Mathematics Math Equations Necktie - Red - One Size Neck Tie'. The following numbered list contains 15 products. Please select 3 products from the list that the user may also purchase.\nProduct List: \n1. Under Armour Men`s ColdGear Lite Cushion Boot Socks, 1 Pair\n2. Little Angel Tasha-685E Patent Bow Mary Jane Pump (Toddler\/Little Girl\/Big Girl) - Fuchsia\n3. Men's Solar System Planets Necktie-Black-One Size Neck Tie by\n4. Crocs Women's Malindi Flat\n5. Wrangler Men's Big & Tall Rugged Wear Unlined Denim Jacket\n6. NIKE Sunray Protect 2 (TD) Womens Fashion-Sneakers 943829\n7. Calvin Klein Women's Seductive Comfort Customized Lift Bra with Lace\n8. Steven Harris Mens Smiley Face Necktie - Yellow - One Size Neck Tie\n9. ComputerGear Math Formula Tie Engineer Silk Equations Geek Nerd Teacher Gift\n10. Harley-Davidson Boys Baby Twin Pack Creeper My Daddy Rides a Harley Orange\n11. Liverpool Football Club Official Soccer Gift Mens Crest T-Shirt\n12. SITKA Traverse Beanie Waterfowl One Size Fits All (90002-WL-OSFA)\n13. The Magic Zoo Sterling Silver Snake Chain with Lobster Clasp\n14. Napier\"Classics\" Silver-Tone Round Button Earrings\n15. Tru-Spec Men's Base Layers Series Gen-iii ECWCS Level-2 Bottom\nYou should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations.\n
        #          [3,8,9]\n
        
        #          Question:A product with description 'Available in both 3 and 6 packs' exists on an online shopping website. Which of the following descriptions may describe the same product in a different language?\n0. \u3010Auto Schlaf\/Aufwach\u3011- Man kann die Abdeckung zum Aufwecken \u00f6ffnen und zum Ruhezustand schlie\u00dfen. An der Innenseite der vorderen Abdeckung befindet sich eine praktische Handschlaufe, um das Lesen beim Halten des Tablets zu erleichtern. Mit einem Gummiband, damit sich der Deckel nicht leicht \u00f6ffnet.\n1. COMPATIBILITY \u2013 The TORRO Magnetic Leather Cardholder is compatible with any MagSafe device (iPhone 14 \/ 13 \/ 12 Series). The built-in magnets ensure it connects to your device with precision for a seamless and secure attach\/detach.\n2. Adoucit l'eau du robinet\n3. Des yeux plus charmants: Les cils magnetique naturel vous aideront \u00e0 cr\u00e9er le maquillage des yeux le plus glamour, faisant de vous la femme la plus attirante de la foule. Apr\u00e8s avoir s\u00e9lectionn\u00e9 parmi plus de 60 types diff\u00e9rents de faux cils magn\u00e9tiques, nous avons choisi ces 5 paires de cils les plus confortables et les plus naturels.\n
        #          2\n
        
        #          """

        self.prompts_no_multi = """
                    You are a very intelligent and helpful online shopping assistant for Amazon who can give reasonable answers or outputs after the online shopping questions. 
                    These are your Q&A history, please continue to answer questions concisely like these until
                    the answer or output of the last question is finished. If your answer is excellent, you will get tips.\n
                    {}
                    """.format(self.exampler_no_multi)
        #  self.prompts_multi = """
        #              You are a very intelligent and helpful online shopping assistant for Amazon who can give reasonable answers or outputs after the online shopping questions.
        #              These are your Q&A history, please continue to answer questions concisely like these until
        #              the answer or output of the last question is finished. If your answer is excellent, you will get tips.\n
        #              {}
        #              """.format(self.exampler_multi)
        self.prompts_multi = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions. If your answer is excellent, you will get tips.\n\n"

    def initialize_models(self):
      
        self.model_name = "models/Qwen2.5-3B-Instruct"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
                https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/blob/master/docs/download-baseline-model-weights.md
            
            """
            )

        # initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
             dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
            #quantization="AWQ",
            enable_prefix_caching=True
        )
        self.tokenizer = self.llm.get_tokenizer()



    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.
    
            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        prompts = batch["prompt"]
        
        # format prompts using the chat template
        formatted_prompts = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 1 if is_multiple_choice else 15

        
        
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm = False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for idx, response in enumerate(responses):
            generation = response.outputs[0].text.strip().split("\n")[0].strip()
            if is_multiple_choice:
                if (len(generation) == 0) or (not generation.isdigit()):
                    generation = "1"
            else:
                if "Question" in generation:
                    generation = generation.split("Question")[0].strip()
                if len(generation) == 0:
                    generation = prompts[idx]
            batch_response.append(generation)
            
        if is_multiple_choice:
            print("MCQ: ", batch_response)

        return batch_response

    def format_prommpts(self, prompts, is_multiple_choice):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """

        formatted_prompts = []
        for prompt in prompts:
            if is_multiple_choice:
                formatted_prompts.append(self.prompts_multi + prompt)
            else:
                new_prompt = """
                                Question: {}\n
                                """.format(prompt)
                prompts = self.prompts_multi + new_prompt if is_multiple_choice else self.prompts_no_multi + new_prompt
                formatted_prompts.append(prompts)

        return formatted_prompts
