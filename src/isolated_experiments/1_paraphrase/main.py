"""
Generate paraphrases for the originaldata using OpenAI

Running command:
    `python -m src.isolated_experiments.1_paraphrase.main`
"""
import glob
import json
import os
import tqdm
import random
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from src.make_data_structured import line_to_dict
from src.utils import get_hash
import re
from langchain.callbacks import get_openai_callback
from nltk.tokenize import sent_tokenize

def parse_enumeration(text):
    pattern = r'^(?:\d+\.\s*)?(.*)$'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


# def do_paraphrase():
    


if __name__ == "__main__":
    with open('api2.key') as fin:
        os.environ["OPENAI_API_KEY"] = fin.readlines()[0].strip()
        # chat_model = ChatOpenAI(temperature=1.0, model_name="gpt-4-0125-preview")
        chat_model = ChatOpenAI(temperature=1.0, model_name="gpt-4-0125-preview")
        # chat_model = ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo-0125")

    template_location = """Please give me a location that is either close or similar in nature with: `{location}`. Please do not return any additional information."""
    location_prompt = PromptTemplate(
        input_variables=["location"],
        template=template_location,
    )


    template_date = """Please give me a date that is either close or similar in nature with: `{date}`. Please do not return any additional information."""
    date_prompt = PromptTemplate(
        input_variables=["date"],
        template=template_date,
    )


    template_rephrase = """Please rephrase the following text, while keeping the following the following phrase fixed:`{phrase}` and maintaining the overall message and length\n\n{text}"""
    rephrase_prompt = PromptTemplate(
        input_variables=["phrase", "text"],
        template=template_rephrase,
    )


    template_rephrase2 = """Please rephrase the following text, maintaining the overall message and length\n\n{text}"""
    rephrase_prompt2 = PromptTemplate(
        input_variables=["text"],
        template=template_rephrase2,
    )

    template_replace = """Please replace word `{word}` and its derivatives with the word `{replacement}` and its appropriate derivatives the following text:\n{text}"""
    replace_prompt = PromptTemplate(
        input_variables=["word", "replacement", "text"],
        template=template_replace,
    )

    data = []
    for f in glob.glob('data/original/*.json'):
        with open(f) as fin:
            data += json.load(fin)

    data = [x for x in data if all(y in x.keys() for y in ['contents', 'text', 'pre_context', 'post_context'])]
    
    banned_contents_words = ['AUTHOR_INST', 'DATE', 'STRENGTHS', 'DESCRIPTION', 'USAGE', 'ASSUMPTIONS', 'All these instances are very', 'is relative temporal', 'AUTHOR_AUTHOR', 'AUTHOR', 'DATASET', 'DATASET', 'SCHEMA', "variation"]
    for w in banned_contents_words:
        data = [x for x in data if w not in x['contents']]

    # x = data[0]
    # contents = line_to_dict(x['contents'])
    # print("\n\n\n")
    # print('#'*100)
    # print(chat_model.call_as_llm(template_location.format(location=contents['location'][0])))
    # print('#'*100)
    # print("\n\n\n")
    # print('%'*100)
    # print(chat_model.call_as_llm(template_date.format(date=contents['time'][0])))
    # print('%'*100)
    # print("\n\n\n")
    # print('^'*100)
    # print(chat_model.call_as_llm(template_rephrase.format(phrase=x['text'], text=x['pre_context'] + x['text'] + x['post_context'])))
    # print('^'*100)
    # print("\n\n\n")
    # print('@'*100)
    # print(x['pre_context'] + x['text'] + x['post_context'])
    # print('@'*100)
    # print("\n\n\n")
    # exit()
    results = []
    with get_openai_callback() as cb:
        for i in range(1):
            # Iterate over the data 
            for x in tqdm.tqdm([x for x in data if 'time' in x['contents'] or 'location' in x['contents']]):
                # Monitor whether the paraphrase becomes invalid
                invalid_paraphrase = False
                
                contents = line_to_dict(x['contents'])
                pre_context_inpt_tokenized  = sent_tokenize(x['pre_context'])
                post_context_inpt_tokenized = sent_tokenize(x['post_context'])

                # Paraphrsae pre_context
                pre_context  = chat_model.call_as_llm(rephrase_prompt2.format(text=' '.join(pre_context_inpt_tokenized[:-4]))) + ' '.join(pre_context_inpt_tokenized[-4:])
                # Paraphrsae post_context
                post_context = post_context_inpt_tokenized[0] + chat_model.call_as_llm(rephrase_prompt2.format(text=' '.join(post_context_inpt_tokenized[1:])))
                
                # Construct original text by concatenating pre_context, text and post_context
                text             = x['pre_context'] + x['text'] + x['post_context']

                # Construct praphrase by concatenatig pre_conetxt (paraphrased), x['text'], and post_context (paraphrased)
                paraphrased_text = pre_context + x['text'] + post_context

                # Paraphrase the conents
                contents_paraphrased = []
                new_contents = ''

                new_text = paraphrased_text
                
                if 'location' in contents and invalid_paraphrase is False:
                    location_paraphrased = [chat_model.call_as_llm(location_prompt.format(location=location)).strip() for location in contents['location']]
                    new_contents += 'location: ' + ', '.join(location_paraphrased)
                    assert(len(contents['location']) == len(location_paraphrased))
                    for original, paraphrased in zip(contents['location'], location_paraphrased):
                        if original.lower() not in paraphrased_text.lower():
                            print("The paraphrase is not valid (1)")
                            invalid_paraphrase = True
                            continue
                        # Replace
                        new_text = chat_model.call_as_llm(replace_prompt.format(word=original, replacement=paraphrased, text=new_text), max_tokens=1000)

                if 'time' in contents and invalid_paraphrase is False:
                    time_paraphrased = [chat_model.call_as_llm(date_prompt.format(date=time)).strip() for time in contents['time']]
                    # If location is present, we separete the contents with a ';'
                    if 'location' in contents:
                        new_contents += '; '
                    new_contents += 'time: ' + ', '.join(time_paraphrased)
                    assert(len(contents['time']) == len(time_paraphrased))
                    for original, paraphrased in zip(contents['time'], time_paraphrased):
                        if original.lower() not in paraphrased_text.lower():
                            print("The paraphrase is not valid (2)")
                            invalid_paraphrase = True
                            continue
                        new_text = chat_model.call_as_llm(replace_prompt.format(word=original, replacement=paraphrased, text=new_text), max_tokens=1000)

                if len(new_text.split(x['text'])) != 2 or invalid_paraphrase is True:
                    print("The paraphrase is not valid (3)")
                    continue

                

                # pre_context, post_context = new_text.split(x['text'])

                hash_id = get_hash(x)

                paraphrased = {
                    **x,
                    'pre_context': new_text.split(x['text'])[0],
                    'post_context': new_text.split(x['text'])[1],
                    'contents': new_contents.lower(),
                    'original': hash_id,
                    'original_pre_context' : x['pre_context'],
                    'original_post_context': x['post_context'],
                    'original_contents'    : x['contents'],
                }
                # if 'location' in contents and 'time' in contents:
                #     print('-'*20)
                #     print(contents)
                #     print(location_paraphrased)
                #     print(time_paraphrased)
                #     print(text)
                #     print(paraphrased_text)
                #     print(new_text)
                #     print('-'*20)
                #     exit()
                # else:
                #     print(contents)
                results.append(paraphrased)

        with open(f'data/paraphrases/240423/data_paraphrased_{random.randint(10000, 20000)}.jsonl', 'a+') as fout:
            for line in results:
                _=fout.write(json.dumps(line))
                _=fout.write('\n')
        print(cb)
        # data = [{
        #     'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + '\n' + x['post_context'],
        #     'output': x['contents']
        # } for x in data]



