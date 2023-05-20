from pydub import AudioSegment
from pydub.playback import play

# # for playing wav file
# song = AudioSegment.from_wav("note.wav")
# print('playing sound using  pydub')
# play(song)

song = AudioSegment.from_wav('../ObjectDigitalProcessing/sang-amthanh.wav')
play(song)


# using playsound
# from playsound import playsound
# playsound('sang-amthanh.wav')