## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1:
Load the Hugging Face API keys and endpoints, then create a function to send text to the NER model and receive entity predictions.
#### STEP 2:
Build a Gradio interface where users enter text and the NER model identifies entities such as names, organizations, and locations.
#### STEP 3:
Merge subword tokens and display the final entities using HighlightedText, then launch the Gradio app with a shareable link.
### PROGRAM:
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
import requests, json
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": inputs}
    
    if parameters is not None:
        data.update({"parameters": parameters})
        
    response = requests.request("POST",ENDPOINT_URL,headers=headers,data=json.dumps(data))
    
    return json.loads(response.content.decode("utf-8"))
API_URL = os.environ['HF_API_NER_BASE'] # NER endpoint

text = "My name is Ashwini. I am studying AIDS. I am interested in Machine Learning"

get_completion(text,parameters=None,ENDPOINT_URL=API_URL)
import gradio as gr

def ner(input):
    output = get_completion(input,parameters=None,ENDPOINT_URL=API_URL)
    
    return {"text": input, "entities": output}

gr.close_all()

demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the dslim/bert-base-NER model under the hood!",
    allow_flagging="never",
    examples=[
        "My name is Ashwini. I am studying AIDS. I am interested in Machine Learning",
        "My name is Karthik, I work at ECE, and I am interested in VLSI."
    ]
)

demo.launch(share=True,server_port=int(os.environ['PORT3']))
def merge_tokens(tokens):
    merged_tokens = []
    
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens


def ner(input):
    output = get_completion(
        input,
        parameters=None,
        ENDPOINT_URL=API_URL
    )
    
    merged_tokens = merge_tokens(output)
    
    return {"text": input, "entities": merged_tokens}


gr.close_all()

demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the dslim/bert-base-NER model under the hood!",
    allow_flagging="never",
    examples=[
        "My name is Ashwini. I am studying AIDS. I am interested in Machine Learning",
        "My name is Karthik, I work at ECE, and I am interested in VLSI."
    ]
)

demo.launch(share=True,server_port=int(os.environ['PORT4']))
```

### OUTPUT:
<img width="1317" height="542" alt="image" src="https://github.com/user-attachments/assets/518388ae-454e-418d-8798-7e8c274468cf" />


### RESULT:
Therefore,the program to design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation is executed and the output is verified.


