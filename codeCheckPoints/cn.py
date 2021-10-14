
import moviepy.editor as mp
import speech_recognition as sr


def getTranscript(video):
   video = mp.VideoFileClip(video)
   video.audio.write_audiofile(r"output.wav") 
   r = sr.Recognizer()
   with sr.AudioFile("output.wav") as source:
      audio = r.record(source) 
   s = r.recognize_google(audio)
   return s


print(getTranscript('cn.mp4'))