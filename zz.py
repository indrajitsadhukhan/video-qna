import os
import openai
from pydub import AudioSegment
from moviepy.editor import *

# If else for video/audio checking

video = VideoFileClip('video.mp4')
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

for i in range(0,k):
    file_path = 'Test_segment{}.wav'.format(i)
    audio_file = open('file_path',"rb")
    transcript = openai.Audio.transcribe("whisper-1",audio_file)

    with open('Test.txt','a') as f:
        f.write(transcript.text)
        f.write(" ")
print("Text written to file")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weavite, FAISS

with open('Test.txt','r') as f:
    reader = f.read()

os.environ["OPENAI_API_KEY"] = "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"

text_splitter = CharacterTextSplitter(
    seperator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)

texts = text_splitter.split_text(reader)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts,embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(),chain_type = "stuff")
query = input("Ask Question: ")

docs = docsearch.similarity(query)

print(chain.run(input_documents = docs,question=query))