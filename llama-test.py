from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("OpenLLM-Ro/RoLlama2-7b-Instruct")



model = AutoModelForCausalLM.from_pretrained("OpenLLM-Ro/RoLlama2-7b-Instruct")

instruction = "Care este cel mai înalt vârf muntos din România?"
chat = [
        {"role": "system", "content": "Ești un asistent folositor, respectuos și onest. Încearcă să ajuți cât mai mult prin informațiile oferite, excluzând răspunsuri toxice, rasiste, sexiste, periculoase și ilegale."},
        {"role": "user", "content": instruction},
        ]
prompt = tokenizer.apply_chat_template(chat, tokenize=False)

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

def get_response(msg : str) -> str:
    msg = [{'role':'user', 'content':msg}]
    inputs = tokenizer.apply_chat_template(msg, tokenize=True, return_tensors="pt",).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
    return tokenizer.decode(out[0])