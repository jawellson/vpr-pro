import os
import yaml
import numpy as np
from tqdm import tqdm
from scipy import signal
from yeaudio.audio import AudioSegment
from yeaudio.augmentation import (
    VolumePerturbAugmentor,
    NoisePerturbAugmentor,
    ReverbPerturbAugmentor,
    SpecAugmentor
)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_audio_files_by_folder(root_dir):
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    folder_files = {}
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        folder_files[folder_name] = []
        for root, _, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    folder_files[folder_name].append(os.path.join(root, file))

    return folder_files


def create_augmenters(config):
    augmenters = []
    spec_augmenter = None
    # 音量增强
    if config.get('volume', {}).get('prob', 0) > 0:
        volume_config = config['volume']
        augmenters.append(
            VolumePerturbAugmentor(
                prob=volume_config.get('prob', 0),
                min_gain_dBFS=volume_config.get('min_gain_dBFS', -15),
                max_gain_dBFS=volume_config.get('max_gain_dBFS', 15)
            )
        )
    # 噪声增强
    if config.get('noise', {}).get('prob', 0) > 0:
        noise_config = config['noise']
        augmenters.append(
            NoisePerturbAugmentor(
                noise_dir=noise_config.get('noise_dir', ''),
                prob=noise_config.get('prob', 0),
                min_snr_dB=noise_config.get('min_snr_dB', 10),
                max_snr_dB=noise_config.get('max_snr_dB', 50)
            )
        )
    # 混响增强
    if config.get('reverb', {}).get('prob', 0) > 0:
        reverb_config = config['reverb']
        augmenters.append(
            ReverbPerturbAugmentor(
                reverb_dir=reverb_config.get('reverb_dir', ''),
                prob=reverb_config.get('prob', 0)
            )
        )
    # Spec增强
    if config.get('spec_aug', {}).get('prob', 0) > 0:
        spec_config = config['spec_aug']
        spec_augmenter = SpecAugmentor(
            prob=spec_config.get('prob', 0),
            freq_mask_ratio=spec_config.get('freq_mask_ratio', 0.1),
            n_freq_masks=spec_config.get('n_freq_masks', 1),
            time_mask_ratio=spec_config.get('time_mask_ratio', 0.05),
            n_time_masks=spec_config.get('n_time_masks', 1),
            max_time_warp=spec_config.get('max_time_warp', 0)
        )
    return augmenters, spec_augmenter


def compute_spectrogram(audio_segment):

    nperseg = 256
    noverlap = 128

    f, t, Zxx = signal.stft(audio_segment.samples, 
                            fs=audio_segment.sample_rate, 
                            nperseg=nperseg, 
                            noverlap=noverlap)
    spec = np.abs(Zxx)
    return f, t, Zxx, spec


def inverse_spectrogram(f, t, Zxx, audio_segment):
    _, reconstructed = signal.istft(Zxx, 
                                   fs=audio_segment.sample_rate, 
                                   nperseg=256, 
                                   noverlap=128)

    if len(reconstructed) > len(audio_segment.samples):
        reconstructed = reconstructed[:len(audio_segment.samples)]
    elif len(reconstructed) < len(audio_segment.samples):
        pad_length = len(audio_segment.samples) - len(reconstructed)
        reconstructed = np.pad(reconstructed, (0, pad_length), 'constant')

    return AudioSegment(reconstructed, audio_segment.sample_rate)

def apply_augmentations(audio_file, folder_name, output_dir, augmenters, spec_augmenter=None):
    try:
        speaker_id = int(folder_name)
    except ValueError:
        speaker_id = 0
    try:
        audio_segment = AudioSegment.from_file(audio_file)
    except Exception as e:
        return []
    basename = os.path.basename(audio_file)
    name, ext = os.path.splitext(basename)
    saved_files = []
    output_folder = os.path.join(output_dir, str(speaker_id))
    os.makedirs(output_folder, exist_ok=True)
    augmented_audio = AudioSegment.from_ndarray(audio_segment.samples, audio_segment.sample_rate)
    for augmenter in augmenters:
        augmented_audio = augmenter(augmented_audio)
    if spec_augmenter is not None:
        f, t, Zxx, spec = compute_spectrogram(augmented_audio)
        aug_spec = spec_augmenter(spec)
        phase = np.angle(Zxx)
        Zxx_aug = aug_spec * np.exp(1j * phase)
        augmented_audio = inverse_spectrogram(f, t, Zxx_aug, augmented_audio)

    output_filename = f"{name}_augmented{ext}"
    output_path = os.path.join(output_folder, output_filename)
    augmented_audio.to_wav_file(output_path)
    saved_files.append(output_path)
    return saved_files


def main():
    input_dir = "dataset/wav"
    output_dir = "dataset/aug_wav1"
    config_path = "configs/augmentation.yml"

    config = load_config(config_path)

    folder_files = get_audio_files_by_folder(input_dir)
    total_files = sum(len(files) for files in folder_files.values())

    augmenters, spec_augmenter = create_augmenters(config)
    total_saved = 0
    for folder_name, audio_files in folder_files.items():
        for audio_file in tqdm(audio_files, desc=f"处理文件夹 {folder_name}"):
            saved_files = apply_augmentations(
                audio_file, 
                folder_name,
                output_dir,
                augmenters,
                spec_augmenter
            )
            total_saved += len(saved_files)
if __name__ == "__main__":
    main()
