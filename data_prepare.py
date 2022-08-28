import os
import pickle
from scipy.sparse import coo_matrix
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm

DRUM_CLASSES = [
    "kick",
    "snare",
    "high tom",
    "low-mid-tom",
    "high floor tom",
    "open hi-hat",
    "closed hi-hat",
    "crash", 
    "ride"
]
    
DRUM2MIDI = {
    "kick": [36],
    "snare": [38, 40, 37],
    "high tom": [48, 50],
    "low-mid-tom": [45, 47],
    "high floor tom": [43, 58],
    "open hi-hat": [46, 26],
    "closed hi-hat": [42, 22, 44],
    "crash": [49, 55, 57, 52],
    "ride": [51, 59, 53],
    
}
IDX2MIDI = {
    0: 36,  # kick
    1: 38,  # snare
    2: 50,  # high tom
    3: 47,  # low-mid tom
    4: 43,  # high floor tom
    5: 46,  # open hi-hat
    6: 42,  # closed hi-hat
    7: 49,  # Crash Cymbal
    8: 51,  # Ride Cymbal

}

DRUM2IDX = {drum: i for i, drum in enumerate(DRUM_CLASSES)}
MIDI2IDX = {
    midi_num: DRUM2IDX[drum] for drum, midi_nums in DRUM2MIDI.items() for midi_num in midi_nums
}

def get_file_path(root_path):
    train_file, valid_file, test_file = [], [], []
    info = pd.read_csv(root_path + '/info.csv')
    for i in range(len(info)):
        split = info['split'].iloc[i]
        file_name = info['midi_filename'].iloc[i]
        if split=='train':
            train_file.append(file_name)
        elif split=='validation':
            valid_file.append(file_name)
        else:
            test_file.append(file_name)
    return train_file, valid_file, test_file

def change_fs(beats, target_beats=16):
    quarter_length = beats[1] - beats[0]
    changed_length = quarter_length /(target_beats / 4)
    changed_fs = 1 / changed_length
    
    return changed_fs

def bin_to_dec(array):    
    decimal = 0
    length = array.shape[0]
    
    for i, elem in enumerate(array):
        decimal += (np.power(2, length-i-1) * elem)
        
    return int(decimal)

def hot_encoding(roll):    
    last_axis = len(roll.shape) - 1
    I = np.eye(np.power(2, roll.shape[-1]), dtype='bool') 
    dec_index = np.apply_along_axis(bin_to_dec, last_axis, roll)
    
    return I[dec_index]

def windowing(roll, window_size=64, bar=16, cut_ratio=0.9):   
    new_roll = []
    num_windows = roll.shape[0] // window_size
    do_nothing = (np.sum((roll == 0), axis=1) == roll.shape[1])
    
    for i in range(0, num_windows):
        break_flag = False
        start_index = window_size * i
        end_index = window_size * (i + 1)
        check_vacant = do_nothing[start_index:end_index]
        for j in range(0, window_size, bar):
            if np.sum(check_vacant[j:j+bar]) > (bar*cut_ratio):
                break_flag = True
                break
        
        if break_flag: continue
        new_roll.append(np.expand_dims(roll[start_index:end_index], axis=0))
        
    return np.vstack(new_roll)

def quantize_drum(inst, fs, start_time, comp=9):
    fs_time = 1 / fs
    end_time = inst.get_end_time()
    

    
    quantize_time = np.arange(start_time, end_time+fs_time, fs_time)
    drum_roll = np.zeros((quantize_time.shape[0], comp))
    
    for i, note in enumerate(inst.notes):
        if note.pitch not in MIDI2IDX.keys():
            continue
        
        start_index = np.argmin(np.abs(quantize_time - note.start))
        end_index = np.argmin(np.abs(quantize_time - note.end))
        
        if start_index == end_index:
            end_index += 1
        
        range_index = np.arange(start_index, end_index)
        inst_index = MIDI2IDX[note.pitch]
    
        for index in range_index:
            drum_roll[index, inst_index] = 1
        
    return drum_roll


def data_processing(file_lst):    
    data = []
    count = 0 # num of bars
    for f_name in tqdm(file_lst):
        file_path = 'groove/' + f_name
        try:
            pm = pretty_midi.PrettyMIDI(file_path)

            # time signature 4/4 check
            ts = pm.time_signature_changes
            if len(ts)==0:
                time_sig = (4, 4)
            else:
                ts = ts[0]
                time_sig = (ts.numerator, ts.denominator)
            if time_sig != (4, 4): 
                continue

            start_time = pm.get_onsets()[0]
            beats = pm.get_beats(start_time)
            tempo = pm.estimate_tempo()
            fs = change_fs(beats)

            # for each inst
            for inst in pm.instruments:
                if inst.is_drum == True:
                    drum_roll = quantize_drum(inst, fs, start_time)
                    drum_roll = windowing(drum_roll)
                    drum_roll = hot_encoding(drum_roll)

                    for i in range(0, drum_roll.shape[0]):
                        # to reduce size of data, sparse encoding
                        data.append(coo_matrix(drum_roll[i]))
        except:
            continue
    return data

def transform_to_midi(array, fs, comp=9):
    fs_time = 1 / fs
    decimal_idx= np.where(array == 1)[1]
    binary_idx = list(map(lambda x: np.binary_repr(x, comp), decimal_idx))
    
    # initialize
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32, is_drum=True)
    pm.instruments.append(inst)
    
    for i, inst_in_click in enumerate(binary_idx):
        start_time = fs_time * i
        end_time = fs_time * (i + 1)
        
        # add instruments
        for j in range(0, len(inst_in_click)):
            if inst_in_click[j] == '1':
                pitch = IDX2MIDI[j]
                inst.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))
                
    return pm