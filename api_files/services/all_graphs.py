from matplotlib.lines import Line2D
import librosa
import numpy as np
from services.diarizing import diarize_with_speechbrain
import matplotlib.pyplot as plt

def plot_audio_with_speakers(file_path, sr=10, threshold=90):
    speaker_dict=diarize_with_speechbrain(file_path)
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)
    y_db = abs(librosa.amplitude_to_db(y))
    x = np.arange(len(y_db)) / sr

    # Plot setup
    plt.figure(figsize=(15, 5))

    # Plot amplitude with threshold coloring
    for i in range(1, len(y_db)):
        x_segment = [x[i-1], x[i]]
        y_segment = [y_db[i-1], y_db[i]]

        if y_segment[0] <= threshold and y_segment[1] <= threshold:
            plt.plot(x_segment, y_segment, color='green')
        elif y_segment[0] >= threshold and y_segment[1] >= threshold:
            plt.plot(x_segment, y_segment, color='red')
        else:
            slope = (y_segment[1] - y_segment[0]) / (x_segment[1] - x_segment[0])
            x_cross = x_segment[0] + ((threshold - y_segment[0]) / slope)
            if y_segment[0] <= threshold:
                plt.plot([x_segment[0], x_cross], [y_segment[0], threshold], color='green')
                plt.plot([x_cross, x_segment[-1]], [threshold, y_segment[-1]], color='red')
            else:
                plt.plot([x_segment[0], x_cross], [y_segment[0], threshold], color='red')
                plt.plot([x_cross, x_segment[-1]], [threshold, y_segment[-1]], color='green')

    # Speaker bars
    for ele in speaker_dict:
        start, end = ele['start'], ele['end']
        speaker_color = 'blue' if ele['speaker'] == 'Speaker_0' else 'orange'
        plt.plot(np.linspace(start, end, 100), [y_db.min()-3]*100, color=speaker_color, linewidth=10)

    # WPM bars
    for ele in speaker_dict:
        start, end = ele['start'], ele['end']
        wpm = (len(ele['text'].split()) / (end - start)) * 60
        color = 'darkgreen' if wpm > 160 else 'red'
        plt.plot(np.linspace(start, end, 100), [y_db.max()+5]*100, color=color, linewidth=10)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Below Threshold'),
        Line2D([0], [0], color='red', lw=2, label='Above Threshold'),
        Line2D([0], [0], color='blue', lw=2, label='Speaker 1'),
        Line2D([0], [0], color='orange', lw=2, label='Speaker 2')
    ]
    plt.legend(handles=legend_elements)
    plt.ylim(30, y_db.max() + 10)
    plt.grid()
    plt.show()
