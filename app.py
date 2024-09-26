

from flask import Flask, request, render_template, jsonify
import yt_dlp
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from pyannote.audio import Pipeline
from utils import diarize_text
import os



app = Flask(__name__)
CORS(app)  # CORS 설정 추가


def download_audio_and_video_from_youtube(url, output_video_path='static/downloaded_video.mp4', output_audio_path='static/downloaded_audio.wav'):
    # 기존 파일이 존재하는 경우 삭제
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    # 비디오 다운로드
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_video_path,
        'merge_output_format': 'mp4',
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"An error occurred while downloading video: {e}")

    # 오디오 다운로드
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'static/temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        os.rename('static/temp_audio.wav', output_audio_path)
    except Exception as e:
        print(f"An error occurred while downloading audio: {e}")

def process_audio(audio_path):
    model = whisper.load_model("medium.en")
    asr_result = model.transcribe(audio_path)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="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")
    diarization_result = pipeline(audio_path)

    subtitles = [(segment.start, segment.end, text) for segment, _, text in diarize_text(asr_result, diarization_result) if text]
    return subtitles

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("ko"))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_subtitles(subtitles):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return [(start, end, translate_text(text, model, tokenizer)) for start, end, text in subtitles]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form['url']
        output_video_path = 'static/downloaded_video.mp4'
        output_audio_path = 'static/downloaded_audio.wav'
        
        # 유튜브 오디오와 비디오 다운로드
        download_audio_and_video_from_youtube(youtube_url, output_video_path, output_audio_path)

        # 음성 텍스트 변환 및 화자 분리
        subtitles = process_audio(output_audio_path)

        # 자막 번역
        translated_subtitles = translate_subtitles(subtitles)

        return render_template('video.html', video_path=output_video_path, subtitles=subtitles, translated_subtitles=translated_subtitles)

    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
