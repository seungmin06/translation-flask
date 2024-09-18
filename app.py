from flask import Flask, request, send_file
import yt_dlp
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pyannote.audio import Pipeline
import os

app = Flask(__name__)
CORS(app)  # CORS 설정 추가


def download_video(url, output_video_path='downloaded_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_video_path,
        'merge_output_format': 'mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_audio(url, output_audio_mp3_path='downloaded_audio.mp3', output_audio_wav_path='downloaded_audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    os.system(f"ffmpeg -i temp_audio.mp3 {output_audio_wav_path}")
    os.rename('temp_audio.mp3', output_audio_mp3_path)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    return [(segment['start'], segment['end'], segment['text']) for segment in result['segments']]

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("ko"))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_subtitles(subtitles):
    model_name = "./facebook_model"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return [(start, end, translate_text(text, model, tokenizer)) for start, end, text in subtitles]

def create_text_image(text, width, height):
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_size = 30
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, height - text_height - 20)
    draw.text(position, text, font=font, fill="white")
    return np.array(image)

def add_subtitles(video_path, subtitles):
    video = VideoFileClip(video_path)
    subtitle_clips = []
    for subtitle in subtitles:
        start, end, text = subtitle
        width, height = video.size
        text_image = create_text_image(text, width, height)
        txt_clip = ImageClip(text_image).set_duration(end - start).set_start(start).set_pos('bottom')
        subtitle_clips.append(txt_clip)
    video_with_subtitles = CompositeVideoClip([video] + subtitle_clips)
    video_with_subtitles.write_videofile("output_with_subtitles.mp4", codec='libx264')

def diarize_audio(audio_path, output_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization = pipeline(audio_path)
    with open(output_file, "w") as rttm:
        diarization.write_rttm(rttm)

def add_end_time_to_rttm(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()
    with open(output_file, "w") as file:
        for line in lines:
            if line.startswith("SPEAKER"):
                parts = line.split()
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                file.write(f"{parts[0]} {parts[1]} {parts[2]} {start_time:.3f} {end_time:.3f} {parts[5]} {parts[6]} {parts[7]} {parts[8]}\n")

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    url = data['url']

    download_video(url, 'downloaded_video.mp4')
    download_audio(url, 'downloaded_audio.mp3', 'downloaded_audio.wav')

    subtitles = transcribe_audio('downloaded_audio.wav')
    translated_subtitles = translate_subtitles(subtitles)

    add_subtitles('downloaded_video.mp4', translated_subtitles)

    diarize_audio('downloaded_audio.wav', 'audio.rttm')
    add_end_time_to_rttm('audio.rttm', 'audio_with_end_times.rttm')

    return send_file('output_with_subtitles.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
