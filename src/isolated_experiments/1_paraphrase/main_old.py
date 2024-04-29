import glob
import json
import os
import tqdm
import random
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from make_data_structured import line_to_dict
from src.utils import get_hash
import re
from langchain.callbacks import get_openai_callback

def parse_enumeration(text):
    pattern = r'^(?:\d+\.\s*)?(.*)$'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    else:
        return None



if __name__ == "__main__":
    with open('api2.key') as fin:
        os.environ["OPENAI_API_KEY"] = fin.readlines()[0].strip()
        # chat_model = ChatOpenAI(temperature=1.0, model_name="gpt-4-0125-preview")
        chat_model = ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo-0125")

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

    template_replace = """Please replace word `{word}` and its derivatives with the word `{replacement}` and its appropriate derivatives the following text:\n{text}"""

    replace_prompt = PromptTemplate(
        input_variables=["word", "replacement", "text"],
        template=template_replace,
    )

    data = []
    for f in glob.glob('data/*.json'):
        with open(f) as fin:
            data += json.load(fin)

    data = [x for x in data if all(y in x.keys() for y in ['contents', 'text', 'pre_context', 'post_context'])]
    
    banned_contents_words = ['AUTHOR_INST', 'DATE', 'STRENGTHS', 'DESCRIPTION', 'USAGE', 'ASSUMPTIONS', 'All these instances are very', 'is relative temporal', 'AUTHOR_AUTHOR', 'AUTHOR', 'DATASET', 'DATASET', 'SCHEMA', "variation"]
    for w in banned_contents_words:
        data = [x for x in data if w not in x['contents']]

    x = data[0]
    contents = line_to_dict(x['contents'])
    print("\n\n\n")
    print(chat_model.call_as_llm(template_location.format(location=contents['location'][0])))
    print("\n\n\n")
    print(chat_model.call_as_llm(template_date.format(date=contents['time'][0])))
    print("\n\n\n")
    print(chat_model.call_as_llm(template_rephrase.format(phrase=x['text'], text=x['pre_context'] + x['text'] + x['post_context'])))
    print("\n\n\n")

    results = []
    with get_openai_callback() as cb:
        for i in range(1):
            for x in tqdm.tqdm(data):
                contents = line_to_dict(x['contents'])
                text = x['pre_context'] + x['text'] + x['post_context']
                paraphrased_text = chat_model.call_as_llm(rephrase_prompt.format(phrase=x['text'], text=text))
                if x['text'] not in paraphrased_text:
                    print("The paraphrase is not valid")
                    continue


                contents_paraphrased = []
                new_contents = ''

                new_text = paraphrased_text
                
                if 'location' in contents:
                    location_paraphrased = [chat_model.call_as_llm(location_prompt.format(location=location)).strip() for location in contents['location']]
                    new_contents += 'location: ' + ', '.join(location_paraphrased)
                    assert(len(contents['location']) == len(location_paraphrased))
                    for original, paraphrased in zip(contents['location'], location_paraphrased):
                        if original.lower() not in paraphrased_text.lower():
                            print("The paraphrase is not valid")
                            continue
                        new_text = chat_model.call_as_llm(replace_prompt.format(word=original, replacement=paraphrased, text=new_text), max_tokens=1000)

                if 'time' in contents:
                    time_paraphrased = [chat_model.call_as_llm(date_prompt.format(date=time)).strip() for time in contents['time']]
                    if 'location' in contents:
                        new_contents += '; '
                    new_contents += 'time: ' + ', '.join(time_paraphrased)
                    assert(len(contents['time']) == len(time_paraphrased))
                    for original, paraphrased in zip(contents['time'], time_paraphrased):
                        if original.lower() not in paraphrased_text.lower():
                            print("The paraphrase is not valid")
                            continue
                        new_text = chat_model.call_as_llm(replace_prompt.format(word=original, replacement=paraphrased, text=new_text), max_tokens=1000)

                if len(new_text.split(x['text'])) != 2:
                    print("The paraphrase is not valid")
                    continue

                

                pre_context, post_context = new_text.split(x['text'])

                hash_id = get_hash(x)

                paraphrased = {
                    **x,
                    'pre_context': pre_context,
                    'post_context': post_context,
                    'contents': new_contents.lower(),
                    'original': hash_id
                }
                if 'location' in contents and 'time' in contents:
                    print(contents)
                    print(location_paraphrased)
                    print(time_paraphrased)
                    print(text)
                    print(paraphrased_text)
                    print(new_text)
                    exit()
                results.append(paraphrased)

        with open(f'data_paraphrased_{random.randint(0, 10000)}.jsonl', 'a+') as fout:
            for line in results:
                _=fout.write(json.dumps(line))
                _=fout.write('\n')
        print(cb)
        # data = [{
        #     'input': 'Text:\n' + x['text'] + '\n\nContext:\n' + x['pre_context'] + '\n' + x['post_context'],
        #     'output': x['contents']
        # } for x in data]



