import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import RendererAgg

from scipy.io import wavfile
from scipy import signal
import io

mpl.use("agg")
_lock = RendererAgg.lock

# -- Helper function to make audio file

def make_audio_file(bp_data, samplerate):
    # -- window data for gentle on/off
    window = signal.windows.tukey(len(bp_data), alpha=1.0/10)
    win_data = bp_data*window

    # -- Normalize for 16 bit audio
    win_data = np.int16(win_data/np.max(np.abs(win_data)) * 32767 * 0.9)

    fs=samplerate
    virtualfile = io.BytesIO()    
    wavfile.write(virtualfile, int(fs), win_data)
    
    return virtualfile

st.title('soundwave analysis')

# st.write("""
# ## display Rawdata, Bandpassed data , STFT
# """)



uploaded_file = st.sidebar.file_uploader("Choose a wav file",type='wav')
if uploaded_file is None:
    st.write("""## chose a wav file to uploade""")
else:
    # if uploaded_file is not None:
    samplerate, data = wavfile.read(uploaded_file)
    if data.ndim != 1:
        data = data[:, 0]
    else:
        data = data
    st.write("filename:", uploaded_file.name)
    st.write('sample rates is :',samplerate)       

    st.audio(make_audio_file(data,samplerate), format='audio/wav')

    st.subheader('Analysis parameters')
    length=data.shape[0]/samplerate
    data=data/np.max(np.abs(data))

    t_values = st.sidebar.slider('Select a range of times in s', 0.0, length, (0.0, 1.0),step=0.1)
    st.write('Time Values:', t_values)
    start = np.round(t_values[0],decimals=1)
    end = np.round(t_values[1],decimals=1)
    # start=np.int(t_values[0])
    # end = np.int(t_values[1])

    cmpdata=data[int(start*samplerate):int(end*samplerate)]
    time = np.linspace(start, end, cmpdata.shape[0])

    f, t, Zxx = signal.stft(cmpdata, samplerate, nperseg=256)
    t=t+start
    Z= 20*np.log10(np.abs(Zxx)/0.00002)
    delt_f=int((samplerate/2)/(f.shape[0]-1))
    # print(delt_f)
    f_values = st.sidebar.slider('Select a range of frequency in Hz', 0, int(samplerate/2), (500, 3000),step=delt_f)
    st.write('Frequency Values:', f_values)

    start_idex=int(f_values[0]/delt_f)
    end_idex=int(f_values[1]/delt_f)

    amp=np.max(Z[start_idex:end_idex,:])


    st.subheader('Raw data')
    with _lock:
        fig1 = plt.figure()
        plt.plot(time, cmpdata, label="sound pressure")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        st.pyplot(fig1, clear_figure=True)
    
    st.subheader('STFT map')
    with _lock:
        fig2 = plt.figure()
        plt.pcolormesh(t, f[start_idex:end_idex], Z[start_idex:end_idex,:],vmin=amp-20, vmax=amp,shading='auto')
        # plt.pcolormesh(t, f[start_idex:end_idex], Z[start_idex:end_idex,:],vmin=40, vmax=60,shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        st.pyplot(fig2, clear_figure=True)
