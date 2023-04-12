import pyttsx3
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from tqdm import tqdm
import pyaudio
import wave
import audioop

transcription = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

using_auto_silent_detection = True

# intialise chat bot 
mname = "facebook/blenderbot-400M-distill"
blenderbot = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
blenderbot.to(device)

# intialise ASR model	
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec = Wav2Vec2ForCTC.from_pretrained("w2v_trained")

# intitialise engine for chatbot to speak
engine = pyttsx3.init()

while(transcription!="goodbye" and transcription!="good bye" and transcription!="bye" and transcription!="stop" and transcription!="good night"):

######## SECTION 1: Record voice

	chunk = 1024
	format_py= pyaudio.paInt16
	channels = 1
	rate = 44100

	p = pyaudio.PyAudio()
	stream = p.open(format=format_py, channels=channels, rate=rate, input=True, frames_per_buffer= chunk)

	print('\nRecording')

	frames = []

	if(using_auto_silent_detection):
		seconds = 60
	else:
		seconds = 4

	started_speaking = False
	silent_frames = []

	for i in range(0, int(rate/chunk*seconds)):
		data = stream.read(chunk)
		frames.append(data)
		if(using_auto_silent_detection):
			rms = audioop.rms(data, 2)    # here's where you calculate the volume
			
			if(rms>2000 and i>5):
				started_speaking = True
			# print(rms)

			if(started_speaking and rms<1000):
				silent_frames.append(rms)
			else:
				silent_frames = []

			if(started_speaking and len(silent_frames)>80):
				break
	print("recording stopped")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open("user_speech.wav", 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(p.get_sample_size(format_py))
	wf.setframerate(rate)
	wf.writeframes(b''.join(frames))
	wf.close()

######## SECTION 2: Convert speech to text

	# load audio at 16hz
	audio, rate = librosa.load("user_speech.wav", sr = 16000)

	# tokenize
	input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values 

	# retrieve logits of prediction
	logits = wav2vec(input_values).logits

	# take argmax and decode
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = processor.batch_decode(predicted_ids)
	transcription = transcription[0].lower().replace(" <unk>", "").replace("<unk>", "")
	print('\n'+'----------------------------\n'+ transcription)


######## SECTION 3: Create text for reply

	if(transcription!="goodbye" and transcription!="good bye" and transcription!="bye" and transcription!="stop" and transcription!="good night"):
	#### pass to Blenderbot

		# tokenize user input
		inputs = tokenizer([transcription], return_tensors="pt")
		inputs = inputs.to(device)

		# generate reply
		reply_ids = blenderbot.generate(**inputs, max_new_tokens=50)
		reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

	else:
	#### write good bye message
		reply = "it was nice talking to you, goodbye"


######## SECTION 4: Play Audio of reply

	engine.say(reply)
	print('\n'+reply+'\n'+'----------------------------\n')
	engine.runAndWait()