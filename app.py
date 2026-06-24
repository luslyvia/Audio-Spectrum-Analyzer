import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import time
import os

# Import mathematical processing libraries
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft
import pywt
from PyEMD import EMD
from vmdpy import VMD

# Import PyTorch deep learning framework and the ResNet1D network structure
import torch
import torch.nn.functional as F
try:
    from resnet1d.resnet1d import ResNet1D
except ImportError:
    st.error("ResNet1D architecture file not found! Please ensure that the 'resnet1d' folder containing 'resnet1d.py' is placed in the same directory as 'app.py'.")

# --- WEB PAGE CONFIGURATION ---
st.set_page_config(page_title="Passive SONAR Spectrum Analyzer", layout="wide")
st.title("Passive SONAR Signal Monitoring & Vehicle Detection System")
st.write("Real-time signal processing pipeline (UPEMD + HHT-VMD) integrated with a Deep 1D ResNet model for aquatic vehicle classification.")

# The 5 original classes corresponding exactly to your model's outputs
CLASSES = ['Ambient Noise', 'Cargo Ship', 'Passenger Ship', 'Tanker', 'Tugboat']

# ---------------------------- MATHEMATICAL SIGNAL PROCESSING ----------------------------
def demon_preprocess(signal: np.ndarray, fs: float, highpass_cutoff: float = 2000, lowpass_cutoff: float = 100) -> np.ndarray:
    nyq = 0.5 * fs
    b_hp, a_hp = butter(4, highpass_cutoff / nyq, btype='high')
    high_freq_noise = filtfilt(b_hp, a_hp, signal)
    rectified = np.abs(high_freq_noise)
    b_lp, a_lp = butter(4, lowpass_cutoff / nyq, btype='low')
    envelope = filtfilt(b_lp, a_lp, rectified)
    return envelope

def btwd_prefilter(signal: np.ndarray, fs: float, thr_m: float, wavelet='db4') -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet, level=5)
    threshold = thr_m * np.std(coeffs[-1])
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    filtered = pywt.waverec(coeffs_thresh, wavelet)
    return filtered[:len(signal)]

def upemd_decompose(signal: np.ndarray, n_phases: int = 4, n_imfs: int = 4) -> np.ndarray:
    """ Optimized n_phases and n_imfs for smoother live local CPU computation """
    emd = EMD()
    fft_signal = np.abs(fft(signal))
    masking_freq = 0.1 * np.argmax(fft_signal) / len(signal)
    imfs_stack = []
    for phase in np.linspace(0, 2*np.pi, n_phases, endpoint=False):
        mask = 0.2 * np.std(signal) * np.cos(2*np.pi*masking_freq*np.arange(len(signal)) + phase)
        imfs = emd(signal + mask, max_imf=n_imfs)
        if len(imfs) < n_imfs:
            padding = np.zeros((n_imfs - len(imfs), len(signal)))
            imfs = np.vstack([imfs, padding])
        imfs_stack.append(imfs[:n_imfs])
    return np.mean(imfs_stack, axis=0)

def btwd_postfilter(signal: np.ndarray, fs: float, thr_aggressive: float = 0.7) -> np.ndarray:
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    threshold = thr_aggressive * np.std(coeffs[-1])
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    filtered = pywt.waverec(coeffs_thresh, 'db4')
    return filtered[:len(signal)]

def reconstruct_signal(imfs: np.ndarray, decisions: np.ndarray, use_postfilter: bool = True, fs=None) -> np.ndarray:
    selected_imfs = [imfs[i] for i in range(len(decisions)) if decisions[i] == 1]
    if use_postfilter and fs is not None:
        selected_imfs = [btwd_postfilter(imf, fs) for imf in selected_imfs]
    if not selected_imfs:
        return np.zeros_like(imfs[0])
    return np.sum(selected_imfs, axis=0)

def hht_vmd(x_rec: np.ndarray, fs: float, alpha: float = 2000, tau: float = 0, K: int = 3, DC: int = 0, init: int = 1, tol: float = 1e-7):
    u, u_hat, omega = VMD(x_rec, alpha, tau, K, DC, init, tol)
    inst_freqs = []
    inst_amps = []
    for mode in u:
        analytic = hilbert(mode)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.diff(phase) / (2*np.pi) * fs
        freq = np.append(freq, freq[-1])
        inst_freqs.append(freq)
        inst_amps.append(amp)
    n_time = len(x_rec)
    freq_axis = np.linspace(0, fs/2, 512)
    hilbert_spectrum = np.zeros((len(freq_axis), n_time))
    for k in range(K):
        for t in range(n_time):
            f = inst_freqs[k][t]
            if 0 < f < fs/2:
                idx = np.argmin(np.abs(freq_axis - f))
                hilbert_spectrum[idx, t] += inst_amps[k][t]**2
    return hilbert_spectrum, np.sum(hilbert_spectrum, axis=1)

def hilbert_huang_pipeline(signal: np.ndarray, fs: float):
    env = demon_preprocess(signal, fs)
    env_filtered = btwd_prefilter(env, fs, thr_m=0.4)
    imfs = upemd_decompose(env_filtered, n_phases=4, n_imfs=4)
    energies = [np.sum(imf**2) for imf in imfs]
    threshold_energy = 0.5 * max(energies)
    decisions = np.array([1 if e > threshold_energy else 0 for e in energies])
    x_rec = reconstruct_signal(imfs, decisions, use_postfilter=True, fs=fs)
    hilbert_spec, marginal_spec = hht_vmd(x_rec, fs, K=3)
    return hilbert_spec, marginal_spec, x_rec, imfs, decisions

# ---------------------------- MODEL PREDICTION PIPELINE ----------------------------
def predict_real_resnet1d(marginal_spec):
    model_path = 'optimized_resnet1d_100ep.pth'
    
    if not os.path.exists(model_path):
        return "Model weight file 'optimized_resnet1d_100ep.pth' not found in working directory!", 0.0, -1

    try:
        # 1. Initialize network based on structural configuration
        model = ResNet1D(
            in_channels=1, base_filters=64, kernel_size=5, stride=2, groups=1, n_block=3, n_classes=5, use_bn=True, use_do=True
        )
        
        # 2. Load trained parameters onto CPU execution environment
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # 3. Z-score normalization (Manual implementation matching the training script)
        mean_val = np.mean(marginal_spec)
        std_val = np.std(marginal_spec) + 1e-12
        norm_spec = (marginal_spec - mean_val) / std_val
        
        # 4. Reshape data into PyTorch tensor dimensions: [Batch_size=1, Channels=1, Length=512]
        input_tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 5. Model execution and inference evaluation
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy()[0]
            
        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)
        
        return predicted_class, confidence, predicted_idx

    except Exception as e:
        return f"AI Processing Exception Error: {str(e)}", 0.0, -2

# ---------------------------- GRAPHICAL USER INTERFACE (UI) ----------------------------
uploaded_file = st.file_uploader("Upload Real-time SONAR Acoustic Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    # 1. Parse and extract primary audio signal properties
    fs_original, signal_original = wavfile.read(uploaded_file)
    if len(signal_original.shape) > 1:
        signal_original = signal_original[:, 0]
        
    # 2. Resample the incoming signal down to exactly 10000 Hz to match model parameters
    TARGET_FS = 10000
    if fs_original != TARGET_FS:
        num_samples = int(len(signal_original) * TARGET_FS / fs_original)
        from scipy.signal import resample
        signal = resample(signal_original, num_samples)
        fs = TARGET_FS
    else:
        signal = signal_original
        fs = fs_original
        
    # 3. Constrain sequence window to exactly 2 seconds (20,000 samples)
    if len(signal) > 20000:
        signal = signal[:20000]
    elif len(signal) < 20000:
        # Zero-padding execution for under-length inputs
        signal = np.pad(signal, (0, 20000 - len(signal)), 'constant')
        
    st.success(f"File loaded and standardized successfully! Original Sampling Rate: {fs_original} Hz -> Standardized: {fs} Hz | Data Sequence Length: {len(signal)} points")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("DSP Processing & Signal Analytics")
        if st.button("Execute Mathematical Calculations & HHT", type="primary"):
            with st.spinner("Processing pipeline: DEMON -> UPEMD -> HHT-VMD..."):
                H_spec, M_spec, x_rec, imfs, dec = hilbert_huang_pipeline(signal, fs)
                
                # Store structural session data state variables
                st.session_state['calculated'] = True
                st.session_state['H_spec'] = H_spec
                st.session_state['M_spec'] = M_spec
                st.session_state['imfs'] = imfs
                st.session_state['signal'] = signal
                st.session_state['fs'] = fs
        
        # Display analysis metrics
        if st.session_state.get('calculated', False):
            tab1, tab2, tab3 = st.tabs(["Raw Audio Time Domain", "IMF Decomposition (UPEMD)", "2D Hilbert Spectrogram Map"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(st.session_state['signal'], color='teal')
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
                
            with tab2:
                current_imfs = st.session_state['imfs']
                fig, axs = plt.subplots(len(current_imfs), 1, figsize=(10, 5), sharex=True)
                for i, imf in enumerate(current_imfs):
                    axs[i].plot(imf, color='orange')
                    axs[i].set_ylabel(f"IMF {i+1}")
                st.pyplot(fig)
                
            with tab3:
                fig, ax = plt.subplots(figsize=(10, 4))
                cax = ax.imshow(st.session_state['H_spec'], aspect='auto', cmap='jet', origin='lower',
                                interpolation='gaussian',  # Smooth the spectrogram map layout
                                extent=[0, len(st.session_state['signal'])/st.session_state['fs'], 0, st.session_state['fs']/2])
                fig.colorbar(cax, label='Energy Intensity')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                st.pyplot(fig)

    with col2:
        st.subheader("Deep Learning Classifier Decision")
        if st.session_state.get('calculated', False):
            if st.button("Trigger 1D ResNet Inference"):
                with st.spinner("Injecting feature array metrics into neural network cells..."):
                    time.sleep(0.5)
                    result_text, confidence, class_idx = predict_real_resnet1d(st.session_state['M_spec'])
                
                # --- MODIFIED SYSTEM CLASSIFICATION LOGIC FOR VESSEL VS NO VESSEL DETECTION ---
                if class_idx == 0:
                    # Index 0 is Ambient Noise (No Vessel)
                    st.success("STATUS: NO VESSEL DETECTED")
                    st.metric(label="System Target Status", value="Ambient Environment")
                    st.metric(label="Model Identification Confidence", value=f"{confidence:.2f}%")
                    st.info("Acoustic wave signatures match ambient background marine noise patterns.")
                elif class_idx > 0:
                    # Indices 1, 2, 3, 4 are aquatic mechanical transport vehicles
                    st.error("STATUS: VESSEL DETECTED")
                    st.metric(label="System Target Status", value="Vessel Presence")
                    st.metric(label="Model Identification Confidence", value=f"{confidence:.2f}%")
                    st.warning("WARNING: High-intensity mechanical propulsion signature matching target vessel detected within array spectrum parameters.")
                else:
                    st.error(result_text)
        else:
            st.info("Please execute the signal processing pipeline on the left column block first.")
else:
    st.info("Systems standing by. Please upload a structured SONAR acoustic signal file (.wav) to initialize telemetry.")