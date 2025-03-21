import numpy as np
import tensorflow as tf


def waveform_to_log_mel_spectrogram(waveform, params):
  """Compute log mel spectrogram of a 1-D waveform."""
  with tf.name_scope('log_mel_features'):
    window_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
    hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1
    magnitude_spectrogram = tf.abs(tf.signal.stft(
        signals=waveform,
        frame_length=window_length_samples,
        frame_step=hop_length_samples,
        fft_length=fft_length))
    # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]

    # Convert spectrogram into log mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.MEL_BANDS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.SAMPLE_RATE,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET)
    # log_mel_spectrogram has shape [<# STFT frames>, MEL_BANDS]

    return log_mel_spectrogram


def spectrogram_to_patches(spectrogram, params):
  """Break up a spectrogram into a stack of fixed-size patches."""
  with tf.name_scope('feature_patches'):
    hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    spectrogram_sr = params.SAMPLE_RATE / hop_length_samples
    patch_window_length_samples = int(
      round(spectrogram_sr * params.PATCH_WINDOW_SECONDS))
    patch_hop_length_samples = int(
      round(spectrogram_sr * params.PATCH_HOP_SECONDS))
    features = tf.signal.frame(
        signal=spectrogram,
        frame_length=patch_window_length_samples,
        frame_step=patch_hop_length_samples,
        axis=0)
    # features has shape [<# patches>, <# STFT frames in an patch>, MEL_BANDS]

    return features
