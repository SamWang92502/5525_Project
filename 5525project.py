import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig  # Import GenerationConfig

class TrollingCommentDetector:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash", output_dir: str = "results"):
        # Get the API key from the argument or from the environment variable.
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: API key not provided. Please set your GEMINI_API_KEY environment variable or provide it as an argument.")
            # Replace below with your valid API key or handle it however is appropriate.
            self.api_key = "YOUR_API_KEY"  # Replace with your actual API key

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load hard-coded few-shot examples.
        self.few_shot_examples = self.load_few_shot_examples()

    def load_few_shot_examples(self):
        # A couple of sample examples as context
        return [
            {
                "comment": "这个评论充满了夸大其词和矛盾的论调，明显是在挑衅。",
                "response": {"classification": 1, "highlights": ["夸大其词", "矛盾的论调"]}
            },
            {
                "comment": "这个人说话温和，没有任何挑衅的意味。",
                "response": {"classification": 0, "highlights": []}
            }
        ]

    def generate_response(self, input_comment: str) -> dict:
        few_shot_prompt = ""
        for ex in self.few_shot_examples:
            few_shot_prompt += f"""
评论: "{ex['comment']}"
返回: {json.dumps(ex['response'], ensure_ascii=False)}
"""
        prompt_text = f"""
你是一位政治文本分析专家。请分析以下中文评论，判断该评论是否属于钓鱼评论（trolling）。
如果是，请列出能指示钓鱼评论的关键词或短语。
请返回一个JSON对象，包含以下键：
 - "classification": 如果检测到钓鱼评论返回1，否则返回0。
 - "highlights": 一个关键词或短语的列表，用以支持你的判断。

以下是一些示例：
{few_shot_prompt}

现在请分析这条评论：
评论: "{input_comment}"
"""

        try:
            generation_config = GenerationConfig(temperature=0.2)  # Create GenerationConfig
            response = self.model.generate_content(
                prompt_text,
                generation_config=generation_config  # Pass GenerationConfig
            )
            if not response or not hasattr(response, "parts") or not response.parts:
                print("No valid response returned from generate_content().")
                return None

            response_text = response.parts[0].text
            print(f"Raw Response Text: '{response_text}'")

            # Refined JSON extraction
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')

            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_string = response_text[json_start:json_end + 1]
                result = json.loads(json_string)
            else:
                print("Could not find valid JSON within the response.")
                result = None

        except Exception as e:
            print("Error generating or parsing response:", e)
            result = None
        return result

    def process_comment(self, input_comment: str) -> dict:
        return self.generate_response(input_comment)

def main():
    detector = TrollingCommentDetector(output_dir="evaluation_results")
    input_comment = input("请输入中文评论: ")
    result = detector.process_comment(input_comment)

    if result:
        print("\n分析结果:")
        print("高亮关键词:", result.get("highlights", []))
        print("分类标签:", result.get("classification", None))
    else:
        print("无法解析模型的响应。")

if __name__ == "__main__":
    main()