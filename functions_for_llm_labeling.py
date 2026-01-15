import transformers
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

def format_value(value):
    """
    格式化数值，如果是浮点数则保留两位小数，如果没有小数位则用整数描述。

    参数:
    value: 需要格式化的数值。

    返回:
    str: 格式化后的字符串。
    """
    if isinstance(value, float):
        return f"{value:.2f}".rstrip('0').rstrip('.')
    return str(value)

def convert_behavior_seq_to_str(behavior_seq):
    """
    将行为序列转化为字符串形式。

    参数:
    behavior_seq (list): 行为序列列表，每个元素为一个字典。

    返回:
    list: 转化后的字符串列表，每个元素为一个时间步的字符串表示。
    """
    str_list = []
    for step in behavior_seq:
        step_values = [format_value(v) for v in step.values()]
        step_str = ", ".join(step_values)
        str_list.append(step_str)
    return "\n".join(str_list)

def save_raw_timeseries_samples(training_ts_samples, save_path="behavior_sequences.txt"):
    """
    将样本中的行为序列转化为多元时间序列，并记录执行时间。

    参数:
    samples (list): 样本列表，每个样本包含行为序列。
    save_path (str): 保存行为序列的文件路径。
    """

    with open(save_path, "w") as f:
        for i, sample in enumerate(training_ts_samples):
            f.write(f"Sample {i}\n")
            sample_label = sample["label"]
            f.write(f"Label: {sample_label}\n")
            behavior_seq = sample["behaviour_sequence"]
            behavior_seq_str = convert_behavior_seq_to_str(behavior_seq)
            f.write(behavior_seq_str + "\n\n")  # 用空行分隔每个样本
            

def get_sample_prompt(samples, number_samples_for_one_prompt):
    # 返回要询问的samples组成的prompt，连续number_samples_for_one_prompt个样本在同一个prompt中询问
    current_sample_number_in_prompt = 0
    prompt_for_samples = ""
    sample_prompt_list = []
    
    for i, sample in enumerate(samples):
        
        behavior_seq = sample["behaviour_sequence"]
        behavior_seq_str = convert_behavior_seq_to_str(behavior_seq)
        prompt_for_samples += f"sample_{(i%number_samples_for_one_prompt)+1}:\n"
        prompt_for_samples += behavior_seq_str + "\n\n"  # 用空行分隔每个样本
        current_sample_number_in_prompt += 1
        
        if current_sample_number_in_prompt == number_samples_for_one_prompt:
            sample_prompt_list.append(prompt_for_samples)
            current_sample_number_in_prompt = 0
            prompt_for_samples = ""

    if current_sample_number_in_prompt != 0:
        sample_prompt_list.append(prompt_for_samples)
    
    return sample_prompt_list


def call_llm_api(prompt, aux_prompt, client, temperature=0):
    """
    调用 LLM API，发送系统提示（system）和用户输入（user）消息。

    参数：
        prompt (str): 用户输入部分（即某个样本组）。
        aux_prompt (str): 系统提示部分。
        client (OpenAI): 已初始化的 OpenAI 客户端对象。

    返回：
        dict: 包含模型输出的结果字符串，以及请求中使用的 Token 统计信息。
    """
    response = client.chat.completions.create(
        # model="qwen-plus",  # 你可以根据需要切换模型
        model="deepseek-v3",
        # model="gpt-4o-mini",
        # model="gpt-5.1",
        messages=[
            {"role": "system", "content": aux_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content
    # token_usage = {
    #     "prompt_tokens": response.usage.prompt_tokens,
    #     "completion_tokens": response.usage.completion_tokens,
    #     "total_tokens": response.usage.total_tokens,
    #     "cached_tokens": response.usage.prompt_tokens_details.cached_tokens
    # }

    # return {"result": result, "tokens": token_usage}

    return result

def call_prompts_with_rate_limit(prompt_list, aux_prompt, client,
                                  temperature=0,
                                  max_workers = 15,
                                  max_requests_per_min=15000,
                                  max_tokens_per_min=1200000):
    """
    多线程发送多个 prompt 到 LLM API，自动管理速率限制，并显示进度条。

    参数：
        prompt_list (List[str]): 多个用户输入部分，每个作为一次请求的主体。
        aux_prompt (str): 系统提示部分，将与每个 prompt 组合。
        client (OpenAI): 初始化的 OpenAI 客户端。
        max_requests_per_min (int): 每分钟请求次数限制。
        max_tokens_per_min (int): 每分钟 token 使用限制。

    返回：
        List[str]: 与 prompt_list 对应的模型输出结果。
    """

    results = [None] * len(prompt_list)
    lock = threading.Lock()
    request_count = 0
    token_count = 0
    start_time = time.time()

    progress_bar = tqdm(total=len(prompt_list), desc="Processing prompts", ncols=80)

    def worker(prompt_index, prompt):
        nonlocal request_count, token_count, start_time

        while True:
            with lock:
                current_time = time.time()
                elapsed = current_time - start_time

                # 每分钟重置计数
                if elapsed >= 60:
                    start_time = current_time
                    request_count = 0
                    token_count = 0

                # 粗略估算 token 数量
                est_prompt_tokens = len(prompt.split()) + len(aux_prompt.split())

                if (request_count + 1 <= max_requests_per_min) and (token_count + est_prompt_tokens <= max_tokens_per_min):
                    request_count += 1
                    token_count += est_prompt_tokens
                    break
                else:
                    time.sleep(max(0, 60 - elapsed))

        try:
            response = call_llm_api(prompt, aux_prompt, client, temperature)
            
            results[prompt_index] = response
        except Exception as e:
            results[prompt_index] = f"ERROR: {str(e)}"
        finally:
            progress_bar.update(1)  # 每完成一个任务就更新进度条

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, prompt in enumerate(prompt_list):
            futures.append(executor.submit(worker, idx, prompt))

        for future in as_completed(futures):
            pass

    progress_bar.close()
    return results


import json
import re

def _strip_markdown_json(text: str) -> str:
    """
    去除 LLM 返回结果中可能存在的 Markdown JSON 代码块标记
    """
    if not isinstance(text, str):
        return text

    text = text.strip()

    # 匹配 ```json ... ``` 或 ``` ... ```
    if text.startswith("```"):
        # 去掉开头 ```json 或 ```
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        # 去掉结尾 ```
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_llm_result(result_str):
    """
    将 LLM 返回的 JSON 字符串解析为有序列表，每个元素为一个 sample 的判断结果。

    参数：
        result_str (str): LLM API 返回的 JSON 格式字符串（可能被 ```json 包裹）

    返回：
        List[Dict]: 按照 sample 顺序排序的列表，每个元素是一个 dict，包含判断结果。
    """
    try:
        # 1. 预处理，去掉 Markdown 包装
        cleaned_str = _strip_markdown_json(result_str)

        # 2. JSON 解析
        data = json.loads(cleaned_str)

        # 3. 提取 sample_n 的键并排序
        sample_keys = sorted(
            data.keys(),
            key=lambda x: int(re.findall(r"\d+", x)[0])
        )

        # 4. 按顺序返回每个 sample 的判断结果
        return [data[sample] for sample in sample_keys]

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw result:\n{result_str}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Raw result:\n{result_str}")
        return []






    
         