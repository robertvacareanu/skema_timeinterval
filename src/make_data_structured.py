from typing import Dict, Any
from src.utils import normalize_date


def line_to_dict(line: str, should_normalize_date: bool = False):
    # Split the input string by semicolon to separate key-value pairs
    pairs = line.strip().split(';')
    result = {}
    # Iterate through each pair
    for pair in pairs:
        # Split each pair by colon to separate key and value
        split = pair.strip().split(':')
        if len(split) % 2 != 0:
            continue
        if len(split) == 0:
            continue
        if len(split) != 2:
            continue
        key, value = split
        # Strip any leading or trailing whitespace from key and value
        key = key.strip()
        value = value.strip().split(',')
        value = [v.strip() for v in value]
        if key in result:
            result[key] += value
        else:
            result[key] = value    
    
    
    if should_normalize_date:
        if 'time' in result:
            result['time'] = [normalize_date(date) for date in result['time']]
    
    return result


def get_content_sorted(line: Dict[str, Any]):
    content = line_to_dict(line['contents'])
    result  = {}

    pre_context  = line['pre_context'].lower()
    post_context = line['post_context'].lower()
    max_len      = len(pre_context) + len(post_context) + 1
    
    for key in sorted(content.keys()):
        items = [(pre_context.index(i.lower().strip()) if i.lower().strip() in pre_context else max_len, post_context.index(i.lower().strip()) if i.lower().strip() in post_context else max_len, i) for i in content[key]]
        items = sorted(items, key=lambda x: (x[0], x[1]))
        result[key] = [x[-1] for x in items]

    return result
        


if __name__ == "__main__":
    # Example usage:
    line = {
        "type": "Highlight",
        "page": 5,
        "page_label": "275",
        "start_xy": [
        398.034,
        168.968
        ],
        "prior_outline": "3.2. Model calibration from the data",
        "text": "diagnosis rate r",
        "contents": "location: Wuhan; time: january 21, february 4, 2020",
        "author": "chunwei",
        "created": "2023-10-10T22:53:09",
        "color": "#ffd100",
        "pre_context": " onset of symptoms WHO (2003); so, the SARS latent period is on average longer than the incubation period. For COVID-19, evidence has shown that infected individuals can be infectious before the onset of symptoms (Bai et al., 2020), but the length of the latent period is largely unknown. In comparison to the SIR model, the SEIR model has the strength of being more biologically realistic, but the SEIR model has the drawback of having two additional unknown parameters: the latent period and the initial latent population. The transfer diagrams for both models are shown in Fig. 1. The biological meaning of all model parameters are given in Table 1 and Table 2. A key assumption in both models is that deaths occurring in the S, E, and I compartments are negligible during the period of model predictions. (4 months). Since we use the newly confirmed case data for model calibration, which is matched to the rI term in both models, the death term in the R compartment has no effect on our model fitting. The systems of differential equations for each model is given below: S' ¼ (cid:1)bIS I' ¼ bIS (cid:1) ðr þ mÞI R' ¼ rI (cid:1) dR S' ¼ (cid:1)bIS E' ¼ bIS (cid:1) εE I' ¼ εE (cid:1) ðr þ mÞI R' ¼ rI (cid:1) dR (2) (3) 3.2. Model calibration from the data For data reliability, the data used for both models (2) and (3) is the newly confirmed cases in Wuhan city from the official reports from January 21 to February 4, 2020 (National Health Commission of the People's Republic of China, 2020). It is common to use a Poisson or negative binomial probability model for observed count data. When the mean of a Poisson or negative binomial distribution is large, it approximates a normal distribution. Since the newly confirmed cases are approaching large values quickly, the distribution of the count data will be approximately normal and the probability model for the observed count data in our study was assumed to a normal distribution with mean given by rI and variance given by 1= t. There are four parameters to be estimated in the SIR model from data: transmission rate b, ",
        "post_context": ", the initial population size I0 for the compartment I on January 21, 2019 (t ¼ 0), and the variance q ¼ 1=t for the noise distribution in the Fig. 1. Transfer diagrams for an SIR and an SEIR model for COVID-19 in Wuhan. "
    }
    parsed_dict = line_to_dict(line['contents'])
    print(parsed_dict)
    print(get_content_sorted(line))
