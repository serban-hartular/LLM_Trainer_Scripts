import unsloth
import torch
import datasets
from trl import SFTTrainer
from unsloth import FastLanguageModel
from transformers import TrainingArguments

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower
orig_model_path = 'OpenLLM-Ro/RoLlama3.1-8b-Instruct'
mask = 0b1101101101
out_model_name = f'roLlama3-Instruct-Grammar-{mask:04X}'

COUNT = 10000

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = orig_model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 1,
)

ds_orig = datasets.load_dataset('hartular/rrt-grammatical_errors-split')

ds_orig = ds_orig.filter(lambda ex: (0x01 << ex['error_class']) & mask)

# transform to good_good and good_bad pairs
ds_dict = datasets.DatasetDict()
for split in ds_orig.keys():
        orig_data = ds_orig[split].to_list()
        data_list = []
        for d in orig_data:
            data_list.extend([{'input':d['good_text' if is_good else 'bad_text'],
                               'response':d['good_text']} for is_good in (False, True)])
        ds_dict[split] = datasets.Dataset.from_list(data_list)

def preprocess_function(examples : list[dict]) -> list[dict]:
    return [{'text':tokenizer.apply_chat_template(
        conversation=[
            {"role":"user", "content":ex['input']},
            {"role":"assistant", "content":ex['response']},
        ], tokenize=False)}
            for ex in examples]


ds_train = ds_dict['train'].shuffle().select(range(COUNT))

args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not unsloth.is_bfloat16_supported(),
        bf16=unsloth.is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir=out_model_name,
        seed=0,
)

trainer=SFTTrainer(model=model,
                   tokenizer=tokenizer,
                   train_dataset=ds_train,
                   dataset_text_field="text",
                   max_seq_length=max_seq_length,
                   args=args,
    # dataset_num_proc=2,
    # packing=True,
)

trainer.train()

model = FastLanguageModel.for_inference(model)

model.save_pretrained_merged(out_model_name+'-LORA', tokenizer, save_method="lora")
model.push_to_hub_merged(out_model_name+'-LORA', tokenizer, save_method="lora")

model.save_pretrained_merged(out_model_name, tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(out_model_name, tokenizer, save_method="merged_16bit")

model.push_to_hub_gguf(out_model_name+"-GGUF", tokenizer, 'q4_k_m')
