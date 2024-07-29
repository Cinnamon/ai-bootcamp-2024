import json 
import uuid
from datasets import load_dataset

def convert_ms_marco_to_qasper(data):
    full_text = [{
        "section_name": "",
        "paragraphs": data["passages"]["passage_text"]
    }]

    ground_truth = [data["passages"]["passage_text"][i] for i, is_selected in enumerate(data["passages"]["is_selected"]) if is_selected == 1]
    qas = [{
        "question": data["query"],
        "question_id": str(uuid.uuid4()),
        "answers": [{
            "answer": {
                "unanswerable": True if len(data["answers"]) == 0 else False,
                'extractive_spans': [],
                'yes_no': None,
                "free_form_answer": "" if len(data["answers"]) == 0 else data["answers"][0],
                "evidence": ground_truth
            }
        }],
    }]

    return {
        "full_text": full_text,
        "qas": qas
    }

ds = load_dataset(path = "microsoft/ms_marco", name = "v1.1", split="test")     
ds.to_json("ms_marco_raw.json") 

converted_data = {}

for i, line in enumerate(open("ms_marco_raw.json")):
    data = json.loads(line)
    converted_data[i] = convert_ms_marco_to_qasper(data)

json.dump(converted_data, open("ms_marco.json", "w"))