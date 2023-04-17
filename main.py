# openai.api_key = "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"


import os
import openai
from pydub import AudioSegment
from moviepy.editor import *

# If else for video/audio checking

import streamlit as st
from PIL import Image

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS




def fetch_answer(question,reader):
# Text completion API
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=reader + question,
    max_tokens=3000,
    temperature=0
    )
    print(reader + question)
    return response.choices[0].text

reader=""
@st.cache_data
def func():
    video = VideoFileClip(uploaded_file.name)
    audio = video.audio
    audio.write_audiofile('test.wav')

    file_path = 'test.wav'

    segment_length = 30000

    audio = AudioSegment.from_file(file_path,format="wav")

    segments = list(range(0,len(audio),segment_length))
    k=0

    for i,start in enumerate(segments):
        end = segments[i+1] if i<len(segments) -1 else len(audio)
        segment = audio[start:end]

        segment_path = os.path.splitext(file_path)[0] + "_segment{}.wav".format(i)
        segment.export(segment_path,format="wav")
        k=k+1

    openai.api_key = "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"
    with open('Test.txt','w') as f:
        f.write("")

    for i in range(0,k):
        file_path = 'Test_segment{}.wav'.format(i)
        audio_file = open(file_path,"rb")
        transcript = openai.Audio.transcribe("whisper-1",audio_file)

        with open('Test.txt','a') as f:
            f.write(transcript.text)
            f.write(" ")
    print("Text written to file")



    with open('Test.txt','r') as f:
        reader = f.read()
    return reader


# Set page title
st.set_page_config(page_title="Video Summarization using GPT-3")

# Display title and description
st.title("Video summarization using GPT-3")
st.write("Upload a video and ask question")


def display():
    # Display video
    if uploaded_file is not None:
        # Read the video file as bytes
        video_bytes = uploaded_file.read()

        # Display video using Streamlit's video widget
        st.video(video_bytes, format=uploaded_file.type)
        return func()

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv", "mov"])
reader=display()
question = st.text_input("Ask Question: ", "")
submit = st.button("Submit")
if submit:
    st.write("",fetch_answer(question,reader))

# os.environ["OPENAI_API_KEY"] = "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"

# text_splitter = CharacterTextSplitter(
#     seperator = "\n",
#     chunk_size = 1000,
#     chunk_overlap = 200,
#     length_function = len
# )
# texts = text_splitter.split_text(reader)
# chunk_size = 1000
# overlap = 200
# texts = []
# start = 0
# while start < len(reader):
#     chunk = reader[start:start + chunk_size]
#     texts.append(chunk)
#     start += chunk_size - overlap



# embeddings = OpenAIEmbeddings()
# docsearch = FAISS.from_texts(texts,embeddings)

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI

# chain = load_qa_chain(OpenAI(),chain_type = "stuff")
# query = input("Ask Question: ")

# docs = docsearch.similarity(query)

# print(chain.run(input_documents = docs,question=query))