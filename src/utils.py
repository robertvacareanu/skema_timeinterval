import hashlib
from bs4 import BeautifulSoup
import re


def get_hash(line):
    text = line['text'] + line['contents'] + line['pre_context'] + line['post_context']
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def remove_unnecessary_content(line, necessary_content = ['location', 'time']):
    """
    Remove unnecessary content (e.g., `variable`)

    Example input
    `location: china; time: now, tomorrow; variable: confirmation rate of China excluding Hubei`

    Example output
    `location: china; time: now, tomorrow`
    """
    contents = line['contents']
    contents = contents.split('; ')
    contents = [x for x in contents if x.split(':')[0] in necessary_content]

    return {
        **line,
        'contents': '; '.join(contents)
    }

def parse_synthetic_data(line):
    # Parse the text using BeautifulSoup  
    soup = BeautifulSoup(line['content'], "html.parser")    
    # Extracting the words wrapped in tags  
    extracted_data = {      
        "loc": soup.find_all("loc"),
        "tmp": soup.find_all("tmp"),
        "nloc": soup.find_all("nloc"),
        "ntmp": soup.find_all("ntmp"),
        "evt": soup.find_all("evt")
    }
    
    structured_data = {tag: list(dict.fromkeys([elem.get_text() for elem in elems])) for tag, elems in extracted_data.items()}

    contents = ''
    location = structured_data.get('loc', [])
    if len(location) != 0:
        location = 'location: ' + ', '.join(location)
    else:
        location = ''
    
    contents += location
    
    time = structured_data.get('tmp', [])
    if len(time) != 0:
        time = 'time: ' + ', '.join(time)
        if location != '':
            contents += '; '
    else:
        time = ''

    contents += time
    
    untagged_text = soup.get_text()

    split = untagged_text.split(line['event'])

    if len(split) != 2:
        return None

    pre_context  = split[0]
    post_context = split[1]

    return {
        'text': line['event'],
        'pre_context': pre_context,
        'post_context': post_context,
        'contents': contents,
    }

def normalize_date_heuristics(date: str) -> str:
    """
    Normalize dates by dropping:
    - drop prepositions
    - drop ordinal affixes
    - make the order sequence day/month/years
    """
    date = date.lower()
    for preposition in ['on', 'at', 'in', 'the']:
        date = date.replace(preposition, '')
    for punctuation in [',', '.']:
        date = date.replace(punctuation, '')
    # Replace ordinal, only if right before the ordinal is a digit
    
    date = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date)
    
    return date.strip()

def normalize_date(date: str, use_examples: bool = False, how_many_examples: int = 50) -> str:
    import os
    from langchain import PromptTemplate, FewShotPromptTemplate
    from langchain.chat_models import ChatOpenAI

    with open('api2.key') as fin:
        os.environ["OPENAI_API_KEY"] = fin.readlines()[0].strip()
        chat_model = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-0125")

    prefix = """Please transform the following date in the ISO extended format (YYYY-MM-DD)."""
    # Examples of input (date, free text), output (ISO date)
    examples = [
        {'input': '21 may 2022', 'output': '2022-05-21'},
        {'input': '15 june 2020', 'output': '2020-06-15'},
        {'input': 'two days ago', 'output': 'two days ago'},
        {'input': 'yesterday', 'output': 'yesterday'},
        {'input': '10 april 2023', 'output': '2023-04-10'},
        {'input': 'next week', 'output': 'next week'},
        {'input': 'today', 'output': 'today'},
        {'input': '25 december 2024', 'output': '2024-12-25'},
        {'input': 'last month', 'output': 'last month'},
        {'input': 'march 10th, 2020', 'output': '2020-03-10'},
        {'input': 'april 1st, 2022', 'output': '2022-04-01'},
        {'input': 'next month', 'output': 'next month'},
        {'input': '10th of August 2022', 'output': '2022-08-10'},
        {'input': 'December 31, 2023', 'output': '2023-12-31'},
        {'input': 'last year', 'output': 'last year'},
        {'input': 'November 11, 2024', 'output': '2024-11-11'},
        # {'input': 'Februray 5th, 2023', 'output': '2023-02-05'},
        {'input': 'February 5th, 2023', 'output': '2023-02-05'},
        {'input': 'May 6th, 2020', 'output': '2020-05-06'},
        {'input': '3rd of July, 2020', 'output': '2020-07-10'},
        {'input': '1st of August, 2005', 'output': '2005-08-01'},
        {'input': 'first day of october, 2016', 'output': '2016-10-01'},
        {'input': 'last day of december, 2015', 'output': '2015-12-31'},
    ]

    prompt = FewShotPromptTemplate(
        examples=examples[:max(1, how_many_examples)] if use_examples else [],
        example_prompt=PromptTemplate(template="Input: {input}\nOutput: {output}", input_variables=['input', 'output']),
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
        prefix=prefix
    )

    output = chat_model.call_as_llm(prompt.format(input=date)).strip()

    return output


if __name__== "__main__":
    from langchain_community.callbacks import get_openai_callback
    # with get_openai_callback() as cb1:
    #     print(normalize_date('3rd of May, 1999', use_examples=True))
    # print(cb1)
    # with get_openai_callback() as cb2:
    #     print(normalize_date('3rd of May, 1999', use_examples=False))
    # print(cb2)
    with get_openai_callback() as cb3:
        print(normalize_date('Thanksgiving 2021', use_examples=False))
    print(cb3)
    # with get_openai_callback() as cb4:
        # print(normalize_date('Thanksgiving 2020', use_examples=True))
    # print(cb3)
    # print(remove_unnecessary_content({'contents': 'location: china; time: now, tomorrow; variable: confirmation rate of China excluding Hubei'}))
    # import json
    # with open('data/synthetic_original/initial_data.jsonl') as fin:
    #     data = json.load(fin)
    
    # data_parsed = [parse_synthetic_data(x) for x in data]
    # data_parsed = [x for x in data_parsed if x]

    # with open('data/synthetic/initial_data.jsonl', 'w+') as fout:
    #     json.dump(data_parsed, fout, indent=4)
