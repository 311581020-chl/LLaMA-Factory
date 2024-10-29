from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import DatasetDict
from transformers import TrainingArguments
import torch
max_seq_length =400
token='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'
tokenizer = AutoTokenizer.from_pretrained("/home/chl/LLaMA-Factory/meta-llama/Meta-Llama-3.1-70B-Instruct", device_map="auto", load_in_8bit= True)
model = AutoModelForCausalLM.from_pretrained("/home/chl/LLaMA-Factory/meta-llama/Meta-Llama-3.1-70B-Instruct", device_map="auto", load_in_8bit= True)


# 准备模型进行INT8量化训练
model = prepare_model_for_kbit_training(model)

# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],  # 根据需要调整目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # 或者其他任务类型，例如 "SEQ2SEQ_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)

alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

ds = DatasetDict.from_json({ 'train' : 'A2.json' }) #load your dataset by hugginface dataloader https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.IterableDataset
dataset=ds.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 6,
    packing = True, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 10,
        warmup_steps = 1000,
        max_steps = 1000,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "output70B",
    ),
    peft_config=lora_config ,
)

trainer_stats = trainer.train()

# import torch
# from trl import SFTTrainer
# from transformers import TrainingArguments
# import json
# from datasets import DatasetDict
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig

# import os
# # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# max_seq_length =400 # Choose any! We auto support RoPE Scaling internally!
# dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for
# token='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'
# tokenizer = AutoTokenizer.from_pretrained("yentinglin/Llama-3-Taiwan-70B-Instruct", token="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", device_map="auto")
# model = AutoModelForCausalLM.from_pretrained("yentinglin/Llama-3-Taiwan-70B-Instruct", token="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", device_map="auto")
# # model = model.to(device)
# #lora

# alpaca_prompt = """
# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs       = examples["input"]
#     outputs      = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

# ds = DatasetDict.from_json({ 'train' : 'A2.json' }) #load your dataset by hugginface dataloader https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.IterableDataset
# dataset=ds.map(formatting_prompts_func, batched = True,)
# print(ds)
# print(dataset['train'])
# peft_config = LoraConfig(
#     r=4,
#     lora_alpha=16,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset['train'],
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 4,
#     packing = True, # Can make training 5x faster for short sequences.
#     args = TrainingArguments(
#         per_device_train_batch_size = 1,
#         gradient_accumulation_steps = 1,
#         warmup_steps = 1000,
#         max_steps = 100,
#         learning_rate = 2e-4,
#         fp16 = not torch.cuda.is_bf16_supported(),
#         bf16 = torch.cuda.is_bf16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "output70B",
#     ),
#     peft_config=peft_config ,
# )


# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

# trainer_stats = trainer.train()

# model.save_pretrained("model70") # Local saving
# tokenizer.save_pretrained("model70")
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving