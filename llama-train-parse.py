import unsloth
import torch
import datasets
from trl import SFTTrainer
from unsloth import FastLanguageModel
from transformers import TrainingArguments

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower
orig_model_path = 'OpenLLM-Ro/RoLlama3.1-8b-Instruct'
out_model_name = 'roLlama3-Instruct-Parse-v0'

NUM_EPOCHS = 2

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

ds_orig = datasets.load_dataset('hartular/rrt-parse0-v0')

ds_train = ds_orig['train']


def preprocess_function(ex) -> list[str]:
    return [tokenizer.apply_chat_template(
        conversation=[
            {"role":"user", "content":in_str},
            {"role":"assistant", "content":res_str},
        ], tokenize=False) for in_str, res_str in zip(ex['input'], ex['response'])]

# ds_train = ds_dict['train'].shuffle().select(range(COUNT))

args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,
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
                   formatting_func=preprocess_function,
                   train_dataset=ds_train,
                   #dataset_text_field="text",
                   max_seq_length=max_seq_length,
                   args=args,
    # dataset_num_proc=2,
    # packing=True,
)

trainer.train()

# model = FastLanguageModel.for_inference(model)

# model.save_pretrained_merged(out_model_name+'-LORA', tokenizer, save_method="lora")
# model.push_to_hub_merged(out_model_name+'-LORA', tokenizer, save_method="lora")

model.save_pretrained_merged(out_model_name, tokenizer, save_method="merged_16bit")
model.push_to_hub_merged('hartular/'+out_model_name, tokenizer, save_method="merged_16bit")
