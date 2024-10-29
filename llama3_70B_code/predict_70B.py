import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from rouge_score import rouge_scorer
from rouge_chinese import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

model_name_or_path = "/home/chl/LLaMA-Factory/meta-llama/Meta-Llama-3.1-70B-Instruct"
eval_dataset_path = '/home/chl/LLaMA-Factory/data/ptsgi/json/繁中翻英法律1.2N_test_10.json'
output_dir = '/home/chl/LLaMA-Factory/saves/llama3.1-70b/lora/predict/102901_noft_law_test'
# 計時器開始
model_start_time = time.time()

# 初始化模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))

# 計時器結束
model_end_time = time.time()
model_loading_time = model_end_time - model_start_time  # 計算模型載入時間
print("predict_model_preparation_time:", model_loading_time)

# 定義模板
alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 分數計算函式
def calculate_scores(reference, hypothesis):
    bleu_score = sentence_bleu(
        [reference.split()],
        hypothesis.split(),
        smoothing_function=SmoothingFunction().method3
    )
    
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    
    rouge_1 = scores[0]['rouge-1']['f']  # F1-score
    rouge_2 = scores[0]['rouge-2']['f']  # F1-score
    rouge_l = scores[0]['rouge-l']['f']  # F1-score
    
    return bleu_score, rouge_1, rouge_2, rouge_l


# 初始化計時器和分數累加變數
start_time = time.time()
total_bleu = 0
total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
num_samples = 0

# 定義終止符
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 打開資料文件並加上進度條
predictions = []  # 用來儲存預測結果
with open(eval_dataset_path, 'r') as file:
    json_data = json.load(file)
    
    # 加入進度條，設定 total 為資料長度
    for row in tqdm(json_data, desc="Generating Responses", total=len(json_data)):
        inp = row['input']
        reference = row.get('output', '')  # 參考答案
        
        # 格式化輸入
        inputs = tokenizer(
            alpaca_prompt.format(row['instruction'], row['input'], ""), 
            return_tensors="pt"
            ).to("cuda")
        
        # 生成輸出，計時生成時間
        generation_start_time = time.time()
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            use_cache=True, 
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id)
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        
        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 計算分數
        bleu, rouge_1, rouge_2, rouge_l = calculate_scores(reference, hypothesis)
        
        # 累加分數
        total_bleu += bleu
        total_rouge_1 += rouge_1
        total_rouge_2 += rouge_2
        total_rouge_l += rouge_l
        num_samples += 1
        
        # 儲存預測結果
        predictions.append({
            'input': inp,
            'output': hypothesis,
            'bleu': bleu,
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l
        })

# 結束計時器
end_time = time.time()

# 計算平均分數
avg_bleu = total_bleu / num_samples if num_samples > 0 else 0
avg_rouge_1 = total_rouge_1 / num_samples if num_samples > 0 else 0
avg_rouge_2 = total_rouge_2 / num_samples if num_samples > 0 else 0
avg_rouge_l = total_rouge_l / num_samples if num_samples > 0 else 0

# 計算總執行時間和速度
predict_runtime = end_time - start_time
predict_samples_per_second = num_samples / predict_runtime
predict_steps_per_second = predict_samples_per_second  # 假設每個樣本為一步

# 儲存預測結果到 JSON 文件
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir,'generated_predictions.json'), 'x', ) as results_file:
    json.dump(predictions, results_file, ensure_ascii=False, indent=4)

# 準備結果字典
results = {
    "predict_bleu-4": avg_bleu,
    "predict_model_preparation_time": model_loading_time,
    "predict_rouge-1": avg_rouge_1,
    "predict_rouge-2": avg_rouge_2,
    "predict_rouge-l": avg_rouge_l,
    "predict_runtime": predict_runtime,
    "predict_samples_per_second": predict_samples_per_second,
    "predict_steps_per_second": predict_steps_per_second,
}

# 儲存總分數到 JSON 文件
with open(os.path.join(output_dir, 'predict_results.json'), 'w') as results_file:
    json.dump(results, results_file, ensure_ascii=False, indent=4)

print("Prediction results saved to generated_predictions.json")
print("Predict scores saved to predict_results.json")

