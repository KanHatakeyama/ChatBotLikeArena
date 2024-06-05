from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

class Bot:
    def __init__(self,model_id="mistral-community/Mixtral-8x22B-v0.1",
                peft_path=None,
                 max_new_tokens=400) -> None:
        print("loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                    device_map="auto",
                                                    #use_flash_attention_2=True
                                                    )
        if peft_path:
            print("loading adapter")
            self.model.load_adapter(peft_path)

        self.pipeline=pipeline("text-generation",
                               model=self.model,tokenizer=self.tokenizer,
                               max_new_tokens=max_new_tokens,
                               repetition_penalty=1.5,)
        

    def ask(self,question):
        return self.pipeline(question)[0]['generated_text'][len(question):]
