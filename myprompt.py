import argparse
import torch
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import functions
from prompter import PromptManager
from validator import validate_function_call_schema

from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)

class ModelInference:
    def __init__(self, model_path, chat_template, load_in_4bit):
        inference_logger.info(print_nous_text_art())
        self.chat_history = []
        self.first_turn = True
        self.prompter = PromptManager()
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = get_chat_template(chat_template)
        
        inference_logger.info(self.model.config)
        inference_logger.info(self.model.generation_config)
        inference_logger.info(self.tokenizer.special_tokens_map)

    def process_completion_and_validate(self, completion, chat_template):

        assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)
        
        if assistant_message:
            validation, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)

            if validation:
                inference_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
                # jspark
                additional_message = assistant_message.split("<tool_call>")[0].strip()
                if additional_message != '':
                    self.chat_history.append({"role": "assistant", "content": additional_message})

                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                self.chat_history.append({"role": "assistant", "content": assistant_message})

                return tool_calls, assistant_message, error_message
        else:
            inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")
        
    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(functions, function_name, None)
        function_args = tool_call.get("arguments", {})

        inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict
    
    def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)
        return completion

    def generate_function_call(self, query, chat_template, num_fewshot, max_depth=5):
        try:
            depth = 0
            # user_message = f"{query}\nPlease refer the <tool_results> to analyze if you can"
            if self.first_turn:
                user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
                self.first_turn = False
            else:
                user_message = f"{query}"
            self.chat_history.append({"role": "user", "content": user_message})
            chat = self.chat_history.copy()
            tools = functions.get_openai_tools()
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            completion = self.run_inference(prompt)

            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                tool_calls, assistant_message, error_message = self.process_completion_and_validate(completion, chat_template)
                prompt.append({"role": "assistant", "content": assistant_message})
                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                                
                if tool_calls:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

                    for tool_call in tool_calls:
                        validation, message = validate_function_call_schema(tool_call, tools)
                        if validation:
                            try:
                                function_response = self.execute_function_call(tool_call)
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                inference_logger.info(f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}")
                            except Exception as e:
                                inference_logger.info(f"Could not execute function: {e}")
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return 

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                elif error_message:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return 

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    # self.chat_history.append({"role": "assistant", "content": assistant_message})

            recursive_loop(prompt, completion, depth)            

        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

model_path = '/home/jspark/projects/LLM-tool-calling/model/Hermes-2-Pro-Llama-3-8B'
inference = ModelInference(model_path, 'chatml', 'False')

"""
inference.generate_function_call('hello, my name is Jongseo Park', 'chatml', None)
inference.generate_function_call('hello, what is your name?', 'chatml', None)
inference.generate_function_call('good. you are chemical expert assistant. ok?', 'chatml', None)
inference.generate_function_call('yeah, do you remember my name?', 'chatml', None)
inference.generate_function_call("good. I want to know SMILES of 80-05-7\nThis is the first turn and you don't have <tool_results> to analyze yet", 'chatml', None)
inference.chat_history
inference.prompter.generate_prompt(inference.chat_history, functions.get_openai_tools(),1)
inference.generate_function_call('do you have any additional information of this chemical?', 'chatml', None)

inference.generate_function_call('recommend me for lunch menu which is trendy', 'chatml', None)
inference.generate_function_call('I want to know the name of the chemical', 'chatml', None)
inference.generate_function_call('It is 80-05-7', 'chatml', None)
inference.generate_function_call('It is not isoproturon!', 'chatml', None)
inference.generate_function_call('no it is wrong. please find it', 'chatml', None)


inference.generate_function_call('find name of 50-00-0', 'chatml', None)
inference.generate_function_call('ok start', 'chatml', None)
inference.generate_function_call('I want to know the name of the chemical', 'chatml', None)
inference.generate_function_call('CAS number is 50-00-0', 'chatml', None)
inference.generate_function_call('I wonder that it is harmful.', 'chatml', None)
inference.generate_function_call('is there are any GHS hazardments related with the chemical?', 'chatml', None)
inference.generate_function_call('I think you gave me wrong information. please do google search about it', 'chatml', None)

inference.generate_function_call('recommed the refrigerator of SAMSUNG', 'chatml', None)
inference.generate_function_call('I want to know now finance price of TESLA', 'chatml', None)



inference.generate_function_call('find name of 50-00-0', 'chatml', None)
inference.generate_function_call('continue plz', 'chatml', None)
inference.generate_function_call('I want to know functional use of it', 'chatml', None)

inference.generate_function_call('I want to know functional use of 80-05-7', 'chatml', None)

inference.generate_function_call('안녕? 한국말로 대답해줘', 'chatml', None)
inference.generate_function_call('80-05-7의 이름을 알려줘', 'chatml', None)
"""

inference.generate_function_call('give me SMILES of 80-05-7 and 50-00-0', 'chatml', 1)
# inference.generate_function_call('give me SMILES of 50-00-0', 'chatml', None)
inference.generate_function_call('I want to know the functional use of them', 'chatml', 1)

inference.generate_function_call('Thank you! Do you have any tools to analysis these results?', 'chatml', 1)

inference.generate_function_call('What were SMILES of my chemicals?', 'chatml', 1)

#inference.generate_function_call('', 'chatml', 1)
