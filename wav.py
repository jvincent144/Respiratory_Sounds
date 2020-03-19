import numpy as np
import wave

####
# Necessary functions to read 24-bit wavfiles

####
# Convert from 24-bit to 32-bit and return array of 32-bit integers

# Wavfiles are little-endian, meaning that the most significant byte is on the right, at the highest address
# Little-endian : least significant bytes first

def convert(num_samples, num_channels, samp_width, data):
    four_byte_rep = np.empty((num_samples, num_channels, 4), np.uint8) # samp_width = 4 bytes
    raw_bytes = np.fromstring(data, np.uint8) # byte by byte representation
    # replace fromstring with frombuffer?
    four_byte_rep[:,:,:samp_width] = raw_bytes.reshape(-1, num_channels, samp_width) # copy 3 bytes; byte 4, rightmost byte is still empty
    four_byte_rep[:,:,samp_width:] = (four_byte_rep[:,:,samp_width-1:samp_width] >> 7)*255 # bitshift right 7 bits
    # Why samp_width-1:samp_width? This sets byte 4 to (byte 3) >> 7
    # Shouldn't byte 4 be set to 0?
    # The bitshifting has to do with the little-endian order of the wavfile.
    # ??? Why multiply by 255 ???
    # return four_byte_rep.view('<i4').reshape(four_byte_rep[:-1]) # Can't reshape in the same line
    out = four_byte_rep.view('<i4') # Change bit format to 32 bit and reshape
    out = out.reshape(out.shape[:-1]) # Because the four bytes along this dimension have collapsed into 1 32-bit representation
    return out/np.max(np.abs(out)) # Normalize by maximum magnitude in the recording

def read_wav(path):
    recording = wave.open(path, 'r') # Returns a Wave_read object
    samp_width = recording.getsampwidth() # Returns sample width in bytes
    num_channels = recording.getnchannels() # 1 channel, 2 channel ; mono, stereo
    f_sample = recording.getframerate()
    frames = recording.getnframes()
    read_frames = recording.readframes(frames) # I shouldn't need to rewind the filepointer, but...
    # Returns a bytestring
    wav_array = convert(frames, num_channels, samp_width, read_frames)
    recording.close() # Close the Wave_read object
    return wav_array # Return numpy array