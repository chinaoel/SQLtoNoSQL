import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
#nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")
import os

# load harry potter data
import requests
res = requests.get("https://raw.githubusercontent.com/gastonstat/harry-potter-data/main/csv-data-file/harry_potter_books.csv")

from io import StringIO

import pandas as pd

TESTDATA = StringIO(res.text)

df = pd.read_csv(TESTDATA, sep=",")
df.head()

agg_df = df.groupby(['book','chapter']).text.apply(lambda x:" ".join(x))

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", device_map="auto",quantization_config=quantization_config,)

pipeline_inst = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline_inst)

template = """Given a prompt delimited by triple backticks, perform the following actions:
- identify as many relations among entities as possible
- output a list in the format ["ENTITY 1", "TYPE of ENTITY 1", "RELATION", "ENTITY 2", "TYPE of ENTITY 2"].

The most important entity types are:
Object 
Person (e.g., Harry Potter, Hermione Granger)
Place (e.g., Hogwarts, Diagon Alley, Forbidden Forest)
Event (e.g., Triwizard Tournament, Battle of Hogwarts)
Spell (e.g., Expelliarmus, Avada Kedavra)
Potion (e.g., Polyjuice Potion, Felix Felicis)
Magical Creature (e.g., Hippogriff, Thestral)
House (e.g., Gryffindor, Slytherin)
Item (e.g., Elder Wand, Invisibility Cloak)
Organization (e.g., Order of the Phoenix, Death Eaters)
Role (e.g., Student, Professor, Auror)


The most important relations are: 
KNOWS 
ATTENDED 
PARTICIPATED_IN 
CAST 
BREWED 
BELONGS_TO
OWNED 
MEMBER_OF 
TEACHES
OCCURS_AT 
PARENT_OF 
SIBLING_OF
SPOUSE_OF
ENEMY_OF

You can create new name entity types and relations as needed.
However, make sure to merge similar entities and relations into the same type.
eg. “Harry Potter” and “Harry” should be the same entity.
eg. "ENEMY_OF" and "OPPONENT_OF" should be the same relation if they carry the same semantic meaning.
Also, if the name entity or relationship is suggested above, use the suggested type.
The triplet must be short and to the point.

Example:
###prompt: “Professor Dumbledore, the principal of Hogwart, gave Gryffindor House 10 points.”
###output:
["Dumbledore", "person", "principal of", "Hogwart", "school"]
["Dumbledore", "person", "award", "Gryffindor", "house"]
Generate as many as possible and accurate triplets from the prompt.
You will be punished if you create triplets that are not in the prompt.
Text before triple backticks must not be interpreted as prompt.
YOU CAN ONLY USE THE PROMPT PROVIDED TO GENERATE THE TRIPLETS.

###prompt: ```{question}```
###output:
"""

def generate_response(question):
  prompt = PromptTemplate(template=template, input_variables=["question","context"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"question":question})
  return response

def get_chunk(temp,chunk_size=1000):
  temp_text = ""
  chunks = []
  while len(temp) > chunk_size:
    temp_text = ""
    while len(temp_text) < chunk_size:

      slicer = temp.index('.') + 1
      temp_text += temp[:slicer]
      temp = temp[slicer:]
    chunks.append(temp_text)
  # last chunk
  chunks.append(temp)
  return chunks

def parseTriplet(sent,resp):
  print(resp)
  #resp = resp.split('output')[-1]
  output = []
  triples = re.findall('\[(.*?)\]\n',resp)
  
  for triple in triples:
    temp = {}
    if len(triple.split(',')) == 5:
      temp['entity1'],temp['entityType1'],temp['relationship'],temp['entity2'],temp['entityType2'] = triple.split(',')
      temp['originalSentence'] = sent
      '''
      inference = temp['entity1'] + 'is' + temp['relationship'].rstrip('of') + 'of' + temp['entity2']
      
      premise = sent
      for premise in sent.split('.'):
        inputs = f"{premise} {nli_pipeline.tokenizer.sep_token} {inference}"
        
        # Get the prediction
        result = nli_pipeline(inputs)
        if result[0]['label'] == 'ENTAILMENT' and result[0]['score'] > 0.7:
          temp['NLIlabel'] = result[0]['label']
          temp['NLIscore'] = result[0]['score']
      '''
      output.append(temp)
          
    else:
      print("Failed To Parse",triple,"Sentence",sent)
  return output


indexes = agg_df.index
for ind in indexes:

  if f"{ind}.csv" in os.listdir():
    print(ind,"Already Done")
    continue

  all_triplets = []

  text = agg_df[ind]
  doc = nlp(agg_df[ind])
  input_sent = ""
  word_counts = len(doc.text)
  current_counts = 0
  for i,sent in enumerate(doc.sents):
    
    input_sent += sent.text
    current_counts += len(sent.text)
    if (len(input_sent) <= 2000):
      continue
    
    
    resp = generate_response(input_sent)
    output = parseTriplet(input_sent,resp)
    
    for triplet in output:
        all_triplets.append(triplet)
    input_sent = ""
    print(f"Total Words: {word_counts} Current Words: {current_counts}")
  print(f'Chapter : {ind}')
  
  
  
  with open(f'{ind}.csv', 'w') as f:
      for key in ("entity1","entityType1","relationship","entity2","entityType2","originalSentence"):
          f.write("%s," % key)
      f.write("\n")
      for item in all_triplets:
          for key in item.keys():
              f.write("%s," % item[key])
          f.write("\n")
  print(f'Chapter {ind} Done')