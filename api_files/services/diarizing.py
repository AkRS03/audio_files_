import whisper
import numpy as np
from speechbrain.inference.speaker  import SpeakerRecognition
from sklearn.cluster import KMeans
import torchaudio
import torchaudio.transforms as T
import os
import shutil
import librosa
from constants import api_key
from pathlib import Path
from sentiment import ensemble_sentiment
import langchain
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import json
import os

# The LLM integration, we are passing values like the overall text-> summarization
# Then we are passing values like pitch-> gender determination, wpm-> speed of talking
# Based on all of the aforementioned inputs we are determining the quality of conversation and training pointers for the client handling team
os.environ['GROQ_API_KEY'] = api_key
template='''from the given {text} and {pitch} usually male pitches are under 1500 hz, formulate the summary of the conversation and then also make accurate judgements about their gender in the conversation summary itself'''
template2='''from the given {text} and {wpm} describe what things were good, could the user have faced any problem due to the operative's speed of talking or any other relevant insights which can be drawn form the conversation'''
prompt1=PromptTemplate(template=template,input_variables=['text','pitch'])
prompt2=PromptTemplate(template=template2,input_variables=['text','wpm'])   
llm=ChatGroq(temperature=0.7, model_name='llama3-70b-8192', api_key=os.environ['GROQ_API_KEY'],)
llm_chain=LLMChain(prompt=prompt1, llm=llm)
llm_chain2=LLMChain(prompt=prompt2, llm=llm)

# Monkey patch symlink_to to perform a copy instead
original_symlink_to = Path.symlink_to

def patched_symlink_to(self, target, target_is_directory=False):
    '''This step may be expendable, test the code without it in the virtual machine'''
    target = Path(target)
    if target_is_directory:
        shutil.copytree(target, self, dirs_exist_ok=True)
    else:
        shutil.copy2(target, self)



Path.symlink_to = patched_symlink_to

# Load Whisper model
whisper_model = whisper.load_model("medium")

# Load SpeechBrain ECAPA-TDNN Speaker Encoder
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir", # Avoids downloading unnecessary files
)
resampler = T.Resample(orig_freq=44100, new_freq=16000)

def get_speaker_embedding(audio_segment, sample_rate=16000):
    # If stereo, take only 1 channel
    if audio_segment.shape[0] > 1:
        audio_segment = audio_segment[0:1, :]
    # Resample if needed
    if sample_rate != 16000:
        audio_segment = resampler(audio_segment)
    # Get speaker embedding
    return speaker_model.encode_batch(audio_segment).squeeze(0).detach().numpy()

def diarize_with_speechbrain(audio_path, num_speakers=2):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    # Step 1: Transcribe with Whisper
    result = whisper_model.transcribe(audio_path, language="en")
    segments = result["segments"]

    # Step 2: Load the full audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Step 3: Extract embeddings for each segment
    embeddings = []
    for seg in segments:
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_audio = waveform[:, start_sample:end_sample]
        emb = get_speaker_embedding(segment_audio, sample_rate)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    # Step 4: Cluster embeddings into N speakers
    labels = KMeans(n_clusters=num_speakers, random_state=42).fit_predict(embeddings)

    # Step 5: Assign and print speaker labels
    diarized_segments = []
    for i, seg in enumerate(segments):
        speaker = f"Speaker_{labels[i]}"
        diarized_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"]
        })

    return diarized_segments
def diarize_speakers(audio_path):

    """
    Function to diarize speakers in an audio file using SpeechBrain's ECAPA-TDNN model.
    
    Args:
        audio_path (str): Path to the audio file.
        num_speakers (int): Number of speakers to identify.
        
    Returns:
        List of dictionaries with speaker segments and their corresponding text.
    """
    diarized=diarize_with_speechbrain(audio_path)
    audio,sr=librosa.load(audio_path, sr=16000)
    user_1_text=''
    user_2_text=''
    user_1_audio=[]
    user_2_audio=[]
    user_1_time=0
    user_2_time=0
    overall_text=''
    for ele in diarized:
        overall_text+=ele['text']
    for ele in diarized:
        if ele['speaker'] == 'Speaker_0':
            user_1_text += ele['text']
            user_1_time += ele.get('end', 0) - ele.get('start', 0)
            user_1_audio.extend(audio[slice(int(ele.get('start', 0) * 16000), int(ele.get('end', 0) * 16000))])
        else:
            user_2_text += ele['text']
            user_2_time += ele['end'] - ele['start']
            user_2_audio.extend(audio[slice(int(ele.get('start', 0) * 16000), int(ele.get('end', 0) * 16000))])
    user_1_pitch = np.mean(librosa.yin(np.array(user_1_audio).flatten(), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr))
    user_2_pitch=np.mean(librosa.yin(np.array(user_2_audio).flatten(), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr))
    user_1_speed=len(user_1_text.split())/user_1_time*60
    user_2_speed=len(user_2_text.split())/user_2_time*60
    output_1=llm_chain.run(text=overall_text,pitch=(user_1_pitch,user_2_pitch))
    output_2=llm_chain2.run(text=overall_text,wpm=user_1_speed)
    return { 
        "user_1_speed": int(user_1_speed),
        "user_2_speed": int(user_2_speed),
        "user_1_centroid": user_1_pitch,
        "user_2_pitch": user_2_pitch,
        "user_1_wordcount": len(user_1_text.split()),
        "user_2_wordcount": len(user_2_text.split()),
        "user_1_sentiment":ensemble_sentiment(user_1_text),
        "user_2_sentiment":ensemble_sentiment(user_2_text),
        "text_summary": {"Conversation_points":output_1,
                         "Conversation_reccommendations": output_2}
    } 


