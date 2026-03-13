from openai import OpenAI
import os
from dotenv import load_dotenv

class LLM():
    def __init__(self, 
                 model: str = "Qwen2.5-14B-Instruct", 
                 base_url: str = "http://localhost:22999/v1", 
                 api_key: str = "EMPTY", 
                 reasoning_effort: str = None,
                 temperature: float = 0,
                 max_gen_len: int = 40960):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_gen_len = max_gen_len

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=3, timeout=240)

    def infer(self, prompt: str):
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            stream=True
        )

        # 流式输出到命令行并收集完整响应
        full_content = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        print()  # 输出完成后换行

        return full_content
    
# openai模型类，添加了对reasoning summary的支持
class OpenAILLM(LLM):
    def __init__(self, model = "Qwen2.5-14B-Instruct", base_url = "http://localhost:22999/v1", api_key = "EMPTY", reasoning_effort = None, temperature = 0, max_gen_len = 40960):
        super().__init__(model, base_url, api_key, reasoning_effort, temperature, max_gen_len)

    def infer(self, prompt: str):
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            reasoning={
                "effort": self.reasoning_effort,
                "summary": "auto"
                },
            stream = False
        )

        print(response)
        
        if response.output[0].hasattr(summary):
            resoning_summary = response.output[0].summary[0].text
            output_text = response.output[1].content[0].text

            return resoning_summary + "\n" + output_text
        else:
            return output_text

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
qwen = LLM(model="Qwen/Qwen3-32B", base_url="http://localhost:8000/v1")
gemini = LLM(model="gemini-3-flash-preview", base_url="https://api.sychatplus.cc/v1", api_key=OPENAI_API_KEY)
gpt = OpenAILLM(model="gpt-4o", base_url="https://api.sychatplus.cc/v1", api_key=GEMINI_API_KEY)

models = {"qwen":qwen, "gemini":gemini, "gpt":gpt}