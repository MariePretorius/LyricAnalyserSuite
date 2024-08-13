import re
import os
import requests
from transformers import pipeline
from bs4 import BeautifulSoup

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")

classifier = pipeline(task="text-classification", model='SamLowe/roberta-base-go_emotions', top_k=None)


def load_classifier():
    return classifier


def process_lyrics(text):
    text = text.replace('\\', '\n')

    lines = []
    current_line = []

    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1

        word_start = i
        while i < len(text) and not text[i].isspace():
            current_line.append(text[i])
            i += 1

        if len(current_line) > 1:
            for j in range(1, len(current_line)):
                if current_line[j].isupper():
                    current_line.insert(j, ' ')
                    break

        if current_line:
            lines.append(''.join(current_line))
            current_line = []

    if current_line:
        lines.append(''.join(current_line))

    processed_text = ' '.join(lines)

    return processed_text


def split_lyrics(text):
    sections = re.split(r'\s*(\[.*?\])\s*', text)

    all_sections = []
    for section in sections:
        if section.strip() and not re.match(r'\s*(\[.*?\])\s*', section):
            all_sections.append(section.strip())

    return all_sections


def get_highest_score_label(model_outputs):
    highest_score = -1
    highest_label = None

    for output in model_outputs:
        if output['score'] > highest_score:
            highest_score = output['score']
            highest_label = output['label']

    return highest_label


def get_lyrics_url(song_name, artist_name):
    base_url = "https://api.genius.com"
    headers = {'Authorization': 'Bearer ' + ACCESS_TOKEN}
    search_url = base_url + "/search"
    data = {'q': f"{song_name} {artist_name}"}
    response = requests.get(search_url, headers=headers, params=data)
    response_data = response.json()

    hits = response_data['response']['hits']
    if not hits:
        return None

    path = hits[0]['result']['path']
    if path:
        lyrics_url = "https://genius.com" + path
    else:
        return None

    return lyrics_url


def get_lyrics(song_name, artist_name):
    song_api_path = get_lyrics_url(song_name, artist_name)

    if song_api_path is None:
        print("No song api path")
        return None

    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }

    song_url = song_api_path

    response = requests.get(song_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve lyrics. Status code: {response.status_code}")
        return None

    html = BeautifulSoup(response.content, "html.parser")
    [h.extract() for h in html('script')]

    lyric_containers = html.find_all("div", attrs={"data-lyrics-container": "true"})

    lyrics_list = []
    for container in lyric_containers:
        lyrics_list.append(container.get_text().strip())

    return lyrics_list


def remove_brackets(text):
    pattern = r'\[.*?\]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def run_lyric_analysis(song_name, artist):
    classifier = load_classifier()
    lyrics = get_lyrics(song_name, artist)

    if lyrics is None:
        return ""

    new_lyrics = []
    for lyric in lyrics:
        processed = process_lyrics(lyric)
        new_lyrics.append(processed)

    cleaned_lyrics = []
    for lyric in new_lyrics:
        processed = split_lyrics(lyric)
        for cleaned_lyric in processed:
            cleaned_lyrics.append(cleaned_lyric)

    label_scores = {}

    for cleaned_lyric in cleaned_lyrics:
        try:
            model_outputs = classifier(cleaned_lyric)
            scores = model_outputs[0]
        except Exception as e:
            print(f"Error processing lyric: {cleaned_lyric}")
            print(f"Exception: {e}")
            return ""
        
        if not scores:
            return ""

        for output in scores:
            label = output['label']
            score = output['score']
            if label not in label_scores:
                label_scores[label] = 0
            label_scores[label] += score

    if label_scores:
        overall_highest_label = max(label_scores, key=label_scores.get)

        if overall_highest_label == "neutral":
            label_scores.pop("neutral")
            second_highest_label = max(label_scores, key=label_scores.get)
            return second_highest_label
        else:
            return overall_highest_label
    else:
        return ""
