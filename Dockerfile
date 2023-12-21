FROM python:3.10
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt
# RUN python3 -c "import os;import pandas as pd;from sentence_transformers import SentenceTransformer;from sklearn import svm; SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1');"
# RUN python3 -c "from PIL import Image;import pandas as pd;from transformers import AutoProcessor, BlipVisionModel;BlipVisionModel.from_pretrained('Salesforce/blip-image-captioning-base');AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base');"

COPY . /app

VOLUME /app/data

RUN chmod +x /app/solution.py
CMD ["python3","/app/solution.py"]
