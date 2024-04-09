import hashlib

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

if __name__== "__main__":
    print(remove_unnecessary_content({'contents': 'location: china; time: now, tomorrow; variable: confirmation rate of China excluding Hubei'}))