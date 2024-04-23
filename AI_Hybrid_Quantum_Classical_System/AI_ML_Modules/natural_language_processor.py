import random
import spacy
from spacy.training import Example
from spacy.util import minibatch

def train_spacy_model(training_data, model='en_core_web_sm', iterations=20, dropout=0.5, batch_size=8):
    """
    Train a SpaCy NER model using the provided training data.

    Parameters:
    - training_data (list): List of tuples containing text and annotations.
    - model (str): Name of the SpaCy model to use (default='en_core_web_sm').
    - iterations (int): Number of training iterations (default=20).
    - dropout (float): Dropout rate for regularization during training (default=0.5).
    - batch_size (int): Size of minibatches for training (default=8).

    Returns:
    - spacy.language.Language: Trained SpaCy NER model.
    """
    # Load existing SpaCy model or create new
    nlp = spacy.load(model)
    ner = nlp.get_pipe('ner') if 'ner' in nlp.pipe_names else nlp.add_pipe('ner')

    # Add entity labels from training data
    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # Disable other pipeline components during training
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            for batch in minibatch(training_data, size=batch_size):
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in zip(texts, annotations)]
                nlp.update(examples, drop=dropout, sgd=optimizer, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")

    return nlp

# Example training data and usage
if __name__ == '__main__':
    TRAIN_DATA = [
        ("Who is Shaka Khan?", {"entities": [(7, 16, "PERSON")]}),
        ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]})
    ]
    nlp_model = train_spacy_model(TRAIN_DATA, iterations=30, dropout=0.6)
    test_text = "Shaka Khan remains influential."
    doc = nlp_model(test_text)
    for ent in doc.ents:
        print(f"{ent.text}, {ent.start_char}, {ent.end_char}, {ent.label_}")
