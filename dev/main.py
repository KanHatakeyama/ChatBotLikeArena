# %%
#
import random
import time
import datetime
from src.GeneralBot import GeneralBot
from src.TanukiBot import TanukiBot
from src.Client import Client

# %%

with open("env/url.txt") as f:
    url = f.read().strip()

# apiクライアントとchatbotを起動
client = Client(url)


# %%
model_dict = {
    "microsoft/Phi-3-medium-128k-instruct": GeneralBot("microsoft/Phi-3-medium-128k-instruct", gpu_memory_utilization=0.4),
    "hatakeyama-llm-team/Tanuki-8B-Instruct": TanukiBot(gpu_memory_utilization=0.2),
}


# %%
model_list = list(model_dict.keys())


# %%
while True:
    row_id, question, inst = client.get_unanswered_question()
    if question == "":
        print("no question to answer")
        time.sleep(10)
        continue

    print(question)
    model_name_A, model_name_B = random.sample(model_list, 2)
    botA = model_dict[model_name_A]
    botB = model_dict[model_name_B]

    responseA = botA.ask(question)
    responseB = botB.ask(question)
    meta1 = model_name_A
    meta2 = model_name_B
    meta3 = datetime.datetime.now().isoformat()

    client.answer(row_id, responseA, responseB, metainfo1=meta1,
                  metainfo2=meta2, metainfo3=meta3)

# %%
row_id

# %%


# %%
row_id

# %%
