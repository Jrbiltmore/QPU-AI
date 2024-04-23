
# /AI_ML_Modules/natural_language_processor.py

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch

def train_spacy_model(training_data, model='en_core_web_sm', iterations=20):
    nlp = spacy.load(model)  # Load existing Spacy model
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=8)
            for batch in batches:
                texts, annotations = zip(*batch)
                docs = [nlp.make_doc(text) for text in texts]
                examples = [Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)]
                nlp.update(examples, drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")

    return nlp

# Example training data and usage
if __name__ == '__main__':
    TRAIN_DATA = [
        ("Who is Shaka Khan?", {"entities": [(7, 16, "PERSON")]}),
        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]})
    ]
    nlp_model = train_spacy_model(TRAIN_DATA)
    test_text = "Shaka Khan remains influential."
    doc = nlp_model(test_text)
    for ent in doc.ents:
        print(f"{ent.text}, {ent.start_char}, {ent.end_char}, {ent.label_}")
