import yt_dlp
from pydub import AudioSegment
import re
import os
def sanitize_filename(filename):
    return re.sub(r'[^A-Za-z0-9\s.]', '', filename).strip()

def download_audio_yt_dlp(video_url, savedir):
    try:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with yt_dlp.YoutubeDL() as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            audio_title = sanitize_filename(info_dict['title'])
            print(f"Sanitized audio title: {audio_title}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(savedir, f'{audio_title}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'ffmpeg_location': r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin" 
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            print("Audio downloaded successfully!")

        audio_file_path = os.path.join(savedir, f"{audio_title}.mp3")
        temp_file_path = os.path.join(savedir, f"{audio_title}.m4a")

        if os.path.exists(audio_file_path):
            print("File found:", audio_file_path)
            return audio_file_path
        elif os.path.exists(temp_file_path):
            print("Temporary file found:", temp_file_path)
            os.rename(temp_file_path, audio_file_path)
            print("Renamed temporary file to:", audio_file_path)
            return audio_file_path
        else:
            print("The downloaded audio file was not found at the expected location.")
            return None

    except Exception as e:
        print("An error occurred while downloading audio:", str(e))
        return None


youtube_url = "https://www.youtube.com/watch?v=qrvK_KuIeJk&t=213s" 
savedir=r"use save dir" 
download_audio_yt_dlp(youtube_url,savedir)
