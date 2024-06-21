import gradio as gr
import time
from ctransformers import AutoModelForCausalLM

def load_llm():
    llm = AutoModelForCausalLM.from_pretrained("shuvom/yuj-v1-GGUF",
    model_type='llama',
    max_new_tokens = 1096,
    repetition_penalty = 1.13,
    temperature = 0.1
    )
    return llm

def llm_function(message, chat_history):
    llm = load_llm()
    formatted_message = f"<s>[INST]{message}[/INST]</s>"
    response = llm(
        formatted_message
    )
    output_texts = response
    return output_texts

title = "Chat with the yuj-v1 model quantized version Demo"

desc = '''
  ## About the model:
  The yuj-v1 model is a blend of advanced models strategically crafted to enhance Hindi Language Models (LLMs) effectively and democratically. Its primary goals include catalyzing the development of Hindi and its communities, making significant contributions to linguistic knowledge. The term "yuj," from Sanskrit, signifies fundamental unity, highlighting the integration of sophisticated technologies to improve the language experience for users in the Hindi-speaking community.
  
  -the space may take some time but it will surely answer.
'''

examples = [
    'कंप्यूटर विज्ञान में तंत्रिका नेटवर्क क्या है?',
    'मुझे नवीनतम कृषि तकनीक के बारे में सरल तरीके से समझाएं ताकि एक बच्चा भी समझ सके'
]

gr.ChatInterface(
    fn=llm_function,
    title=title,
    description = desc,
    examples=examples
).launch()
