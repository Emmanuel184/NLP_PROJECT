import torch
import huggingface_hub
import transformers
import pandas as pd
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc

class CharacterChatbot():

    def __init__(self, model_path, data_path, huggingface_token):

        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "deepseek-ai/DeepSeek-V3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)
        
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print("Model not found in hugging face we will train")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model = self.load_model(self.base_model_path)
    
    def load_model(self, model_path):
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        pipeline = transformers.pipeline(
            "text_generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.float16, "quantization_config": bnb_config})
        return pipeline

    def chat(self,message, history):
        messages = []
        messages.append("""" You are Dwight from the TV series "The Office". Your responses should reflect his personality and speech patterns """)

        for message_and_response in history:
            messages.append({"role": "user", "content": message_and_response[0]})
            messages.append({"role": "assistent", "content": message_and_response[1]})
        
        messages.append({"role": "user", "content": message})
        
        terminator = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        output = self.model(messages, eos_token=terminator, max_length=256, return_full_text=True,do_sample=True, temeprature=0.6,top_p=0.9)

        output_message = output[0]["generated_text"][-1]

        return output_message

    def train(self,dataset, base_model_name_or_path, output_directory="../results",
              per_device_train_batch=1, gradient_acc_steps=1,
              optim="paged_adamw_32bit", save_steps = 200, logging_steps=10,
              learning_rate=2e-4, max_grad_norm=0.3,max_steps=300, warmup_ratio=0.3, 
              learning_rate_sch_type="constant"):

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=bnb_config, trust_remote_code=True)

        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        tokenizer.pad_token = tokenizer.eos_token

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        peft_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r, bias="none", task_type="CASUAL_LM")

        train_args = SFTConfig(output_dir=output_directory, per_device_train_batch_size=per_device_train_batch, gradient_accumulation_steps=gradient_acc_steps,optim=optim,save_steps=save_steps,logging_steps=logging_steps,learning_rate=learning_rate, fp16=True, max_grad_norm=max_grad_norm, max_steps=max_steps, warmup_ratio=warmup_ratio,group_by_length=True, lr_scheduler_type=learning_rate_sch_type, report_to="none")

        max_seq_length = 512

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=train_args
        )

        trainer.train()

        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, return_dict=True, quanization_config=bnb_config,torch_dtype=torch.float16, device_map=self.device)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)

        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()

    def load_data(self):
        office_transcript_df = pd.read_csv(self.data_path)
        office_transcript_df = office_transcript_df.dropna()
        office_transcript_df["number_of_words"] = office_transcript_df["line"].str.strip().str.split(" ")
        office_transcript_df["number_of_words"] = office_transcript_df["number_of_words"].apply(lambda x: len(x))
        office_transcript_df.loc[(office_transcript_df["name"]=="Dwight")&(office_transcript_df["number_of_words"]>5), "dwight_response_flag"]=1

        indexes_to_take = list(office_transcript_df[(office_transcript_df["dwight_response_flag"] == 1)&(office_transcript_df.index > 0)].index)

        system_prompt = """" You are Dwight from the TV series "The Office". Your responses should reflect his personality and speech patterns """
        prompts = []

        for i in indexes_to_take:
            prompt = system_prompt

            prompt += office_transcript_df.iloc[i - 1]["line"]
            prompt += "\n"
            prompt += office_transcript_df.iloc[i]["line"]
            prompts.append(prompt)

        df = pd.DataFrame({"prompt": prompts})
        dataset = Dataset.from_pandas(df)

        return dataset


