from openai import OpenAI
import numpy as np
import pandas as pd
import os

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = '自分のAPIのキーを入れて'

client = OpenAI()

def vectorize_texts(text):
    """ 文書リストをベクトル化して返します """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def find_most_similar(question_vector, vectors, documents):
    """ 類似度が最も高い文書を見つけます """
    vectors = np.vstack(vectors)  # ドキュメントベクトルを垂直に積み重ねて2次元配列にする
    similarities = np.dot(vectors, question_vector.T).flatten() / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vector))
    print("similarities", similarities)
    most_similar_index = np.argmax(similarities)
    return documents[most_similar_index]

def ask_question(question, context):
    """ GPTを使って質問に答えます """
    prompt = f'''以下の質問に以下の情報をベースにして答えてください。
            [ユーザーの質問]
            {question}

            [情報]
            {context}
            '''
    print("prompt", prompt)
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text

# 情報
documents = [
    "この人物は毎朝6時に起き、コーヒーを一杯飲んだ後、パソコンを開きます",
    "休みの日は友人とハイキングに行きます",
    "この人物が見る映画はほとんどがノワール映画です"
]

# 文書をベクトル化
vectors = [vectorize_texts(doc) for doc in documents]

# 質問
question = "この人物のプロフィールは何でしょう？"

# 質問をベクトル化
question_vector = vectorize_texts(question)

# 最も類似した文書を見つける
similar_document = find_most_similar(question_vector, vectors, documents)

# GPTモデルに質問
answer = ask_question(question, similar_document)
print(answer)

