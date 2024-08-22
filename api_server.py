# %%
from src.Client import Client
import datetime
import time
import random
from src.VLLMServer import launch_command, get_client_dict, ask_llm_prompt
from collections import Counter

# %%
import yaml

# %%
conf_path = "config.yaml"

with open(conf_path, "r") as f:
    conf = yaml.safe_load(f.read())

print("run the following llm servers!!")
print(launch_command(conf))
print("---------------")

# %%
client_dict = get_client_dict(conf)
# %%

# %%

with open("env/url.txt") as f:
    url = f.read().strip()

# apiクライアントとchatbotを起動
client = Client(url)


# %%
model_list = list(client_dict.keys())

question_count = 0

# %%
while True:
    if question_count > 2000:
        break
    try:
        row_id, question, inst = client.get_unanswered_question()
    except Exception as e:
        print(e)
        time.sleep(5)
        continue
    if question == "":
        print("no question to answer")
        time.sleep(5)
        continue

    print(question)
    try:
        # 戦わせるモデルを選ぶ
        # 単純にランダムで選択
        # model_name_A, model_name_B = random.sample(model_list, 2)

        # 対戦経験が少ないモデルを選択
        win_models = [i[8] for i in client.values if len(i[8]) > 4]
        lose_models = [i[9] for i in client.values if len(i[9]) > 4]
        model_history = win_models+lose_models
        model_history += model_list

        model_counter = Counter(model_history)

        # もっとも少ないモデルを選択
        # target_model_name = model_counter.most_common()[-1][0]

        # 選択確率の重みを1/出現頻度にして､確率的に選択
        target_model_name = random.choices([i for i in model_counter.keys()], weights=[
                                           1/i[1]/i[1] for i in model_counter.items()])[0]

        other_models = [i for i in client_dict.keys() if i !=
                        target_model_name]

        enemy_model_name = random.choice(other_models)

        if random.random() < 0.5:
            target_model_name, enemy_model_name = enemy_model_name, target_model_name
        model_name_A = target_model_name
        model_name_B = enemy_model_name

        print("asking ", model_name_A, model_name_B)
        responseA = ask_llm_prompt(client_dict, model_name_A, question)
        responseB = ask_llm_prompt(client_dict, model_name_B, question)
        meta1 = model_name_A
        meta2 = model_name_B
        meta3 = datetime.datetime.now().isoformat()

        client.answer(row_id, responseA, responseB, metainfo1=meta1,
                      metainfo2=meta2, metainfo3=meta3)
        question_count += 1
    except Exception as e:
        print(e)
        time.sleep(5)


# %%
