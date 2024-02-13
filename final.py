# gemini_app.py

import os
import streamlit as st
import google.generativeai as genai  # Import your Gemini model library
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import datetime
import re
from copy import deepcopy
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import amax
import keras
import json
import requests

from collections import namedtuple
import calendar


os.environ["GOOGLE_API_KEY"] = "AIzaSyDpRJuB4YCPNecreUFfp37qy-nB3QIL0iw"
v=os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=v)
# Model Configuration

MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}


## Safety Settings of Model
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]


# Initialize your Gemini model (replace with actual initialization)
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings,
)


class NERModel(keras.Model):
    
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super().__init__()
        
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")
        

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


# def image_format_path(image_path):
    
#     img = Path(image_path)

#     if not img.exists():
        
#         raise FileNotFoundError(f"Could not find image: {img}")

#     image_parts = [
#         {
#             "mime_type": "image/png",
             
#             "data": img.read_bytes()
#         }
#     ]
#     return image_parts



def image_format(uploaded_image):
    
    img_bytes = uploaded_image.read()

    image_parts = [
        {
            "mime_type": "image/png",
            "data": img_bytes
        }
    ]
    return image_parts


def gemini_output(image_path, system_prompt, user_prompt):
  
    image_info = image_format(image_path)
    
    input_prompt = [system_prompt, image_info[0], user_prompt]
    
    response = model.generate_content(input_prompt)
    
    return response.text



# STYLE = """
# <style>
# img {
#     max-width: 100%;
# }
# </style>
# """

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")


model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")

model2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

specifier = 0



# def make_tag_lookup_table():
#     iob_labels = ["B", "I"]
#     ner_labels = ["PER", "ORG", "LOC", "MISC"]
#     all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
#     all_labels = ["-".join([a, b]) for a, b in all_labels]
#     all_labels = ["[PAD]", "O"] + all_labels
#     return dict(zip(range(0, len(all_labels) + 1), all_labels))


# mapping = make_tag_lookup_table()



# def lowercase_and_convert_to_ids(tokens):
#     tokens = tf.strings.lower(tokens)
#     return lookup_layer(tokens)


def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags


def overlapping(dates, existingleaves):
  
  Range = namedtuple('Range', ['start', 'end'])
  newrange = Range(start=dates[0], end=dates[1])

  for elv in existingleaves:
    if elv["status"] == "Approved" or elv["status"] == "Pending":
      datetemp = elv["leaveFrom"]
      fromdate = datetime.date(int(datetemp[11:15]), getmonth(datetemp, today), int(datetemp[8:10]))
      datetemp = elv["leaveTo"]
      todate = datetime.date(int(datetemp[11:15]), getmonth(datetemp, today), int(datetemp[8:10]))

      daterange = Range(start=fromdate, end=todate)

      latest_start = max(newrange.start, daterange.start)
      
      earliest_end = min(newrange.end, daterange.end)
      
      delta = (earliest_end - latest_start).days + 1
      
      overlap = max(0, delta)
      
      if overlap != 0:
        return True
      
    return False
  


def distance(s, w1, w2):
        if w1 == w2 :
          return 0
        words = s.split(" ")
        n = len(words)
        min_dist = n+1
        for i in range(n):
            if words[i] == w1 or words[i] == w2:
                prev = i
                break
        while i < n:
            if words[i] == w1 or words[i] == w2:
                if words[prev] != words[i] and (i - prev) < min_dist :
                    min_dist = i - prev - 1
                    prev = i
                else:
                    prev = i
            i += 1
        return min_dist



def getmonth(date, today):
        date = date.lower()
        x = re.search("jan", date)
        if x is not None:
            return 1
        x = re.search("feb", date)
        if x is not None:
            return 2
        x = re.search("mar", date)
        if x is not None:
            return 3
        x = re.search("apr", date)
        if x is not None:
            return 4
        x = re.search("may", date)
        if x is not None:
            return 5
        x = re.search("jun", date)
        if x is not None:
            return 6
        x = re.search("jul", date)
        if x is not None:
            return 7
        x = re.search("aug", date)
        if x is not None:
            return 8
        x = re.search("sep", date)
        if x is not None:
            return 9
        x = re.search("oct", date)
        if x is not None:
            return 10
        x = re.search("nov", date)
        if x is not None:
            return 11
        x = re.search("dec", date)
        if x is not None:
            return 12
        return today.month   



def getyear(date, today):
    x = re.search("[1-9][0-9][0-9][0-9]", date)
    if x is None:
        y = re.search("'[0-9][0-9]", date)
        if y is None:

            return today.year
        else:
            return int(str(today.year)[:2] + y.group()[1:])
    else:
        return int(x.group())



system_prompt = """
               You are a specialist in comprehending receipts.
               Input images in the form of receipts will be provided to you,
               and your task is to respond to questions based on the content of the input image.
               """

image_path = "/content/bill.png"


user_prompt = """
              Convert Invoice data into json format with appropriate 
              json tags as required for the data in image 
              """

def main():
    st.title("Named Entity Recognition (NER)")
    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload file", type=["pdf", "png", "jpg"])
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["pdf", "png", "jpg"]))
        return
        # content = file.getvalue()
        # if isinstance(file, BytesIO):
    

    show_file.image(file)
    output =gemini_output(file,system_prompt,user_prompt)
    st.write(output)


    # uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
    # # st.write(uploaded_image)
    # if uploaded_image:
    #     image_parts = image_format(uploaded_image)
    #     st.image(image_parts)
    #     gemini_output(image_parts, "System says:", "User says:")


       

   
    # show_file.image(uploaded_image)

        # Display the uploaded image
        # st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Get user prompts
        # system_prompt = st.text_input("System Prompt", "System says:")
        # user_prompt = st.text_input("User Prompt", "User says:")

        # # Generate content
        # if st.button("Generate Content"):
        #     response_text = gemini_output(uploaded_image, system_prompt, user_prompt)
        #     st.success("Generated Content:")
        #     st.write(response_text)
        
    # file.close()

if __name__ == "__main__":
    main()
