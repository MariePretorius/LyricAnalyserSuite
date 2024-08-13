import os

from flask import (Flask, redirect, render_template, request, send_from_directory, url_for, jsonify)

app = Flask(__name__)

import os
import json

import classification
import utils

app = Flask(__name__)


@app.route('/get_similar_sentiments', methods=['POST'])
def get_songs():
    req_body = request.get_json()
    access_key = os.environ.get('ACCESS_KEY')
    provided_key = req_body.get('access_key')

    if access_key != provided_key:
        return jsonify({"error": "Unauthorized access. Invalid access key."}), 401

    try:
        song_name = req_body.get('song_name')
        artist = req_body.get('artist')

        recommended_tracks = req_body.get('recommended_tracks', [])
        
        given_emotion = classification.run_lyric_analysis(song_name, artist)

        for track in recommended_tracks:
            track_name = track.get('track_name')
            artist_name = track.get('artist_name')

            if not track_name or not artist_name:
                return jsonify({"error": "Please provide both song_name and artist_name."}), 400
        
            similar_songs = []

            for track in recommended_tracks:
                similar_emotion = False
                emotion = classification.run_lyric_analysis(track_name, artist_name)

                if emotion and given_emotion:
                    distance = utils.emotional_similarity[emotion][given_emotion]
                    if distance >= 7:
                        similar_emotion = True

                    if similar_emotion:
                        similar_songs.append(track)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"recommended_tracks": similar_songs}), 200


@app.route('/')
def index():
   print('Request for index page received')
   return jsonify({"success": "page loaded, go to /get_similar_sentiments for recommendations"}), 200


if __name__ == '__main__':
   app.run()
