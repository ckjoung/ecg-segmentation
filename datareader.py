import wfdb
import numpy as np
import torch

def anom_to_label(anom):
    if anom in {
        'Atrial fibrillation',
        'Atrial flutter, typical'
    }:
        return 1
    else:
        return 0

def load_ludb_tensors(ludb_files, leads=None):
    leads = leads or ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    # initialize arrays
    all_waves = []
    all_p = []
    all_qrs = []
    all_t = []
    all_none = []
    all_cls_target = []

    for record_name in ludb_files:
        # load data
        record = wfdb.rdrecord(record_name)
        anom = record.__dict__['comments'][3][8:-1]
        waves = {
            lead: wave
            for lead, wave in zip(
                record.__dict__['sig_name'],
                record.__dict__['p_signal'].T # shape (12,5000)
            )
        }

        # extract annotation
        for lead in leads:
            wave = waves[lead] # shape (5000,)
            all_waves.append(wave)

            # read annotation
            annotation = wfdb.rdann(record_name, extension=lead)
            sample = annotation.__dict__['sample']
            symbol = annotation.__dict__['symbol']

            # initialize annotation dictionary
            ann_dct = {
                'p': np.zeros(5000,),
                'qrs': np.zeros(5000,),
                't': np.zeros(5000,),
            }
            
            # update annotation array
            on = None
            for t,symbol in zip(sample, symbol):
                if symbol == '(': # symbol denotes onset
                    on = t
                elif symbol == ')': # symbol denotes offset
                    off = t
                    if on != None:
                        ann_dct[key] += np.array([0]*on + [1]*(off-on+1) + [0]*(4999-off))
                        on = None
                else: # symbol denotes peak
                    if symbol in {'p','t'}:
                        key = symbol
                    else:
                        assert(symbol == 'N')
                        key = 'qrs'
            
            # create array indicating non-labeled areas
            assert(np.max(ann_dct['p'] + ann_dct['qrs'] + ann_dct['t']) <= 1)
            ann_dct['none'] = np.ones(5000,) - (ann_dct['p'] + ann_dct['qrs'] + ann_dct['t'])

            all_p.append(ann_dct['p'])
            all_qrs.append(ann_dct['qrs'])
            all_t.append(ann_dct['t'])
            all_none.append(ann_dct['none'])
            all_cls_target.append(anom_to_label(anom))
    
    # finalize arrays
    all_waves = np.array(all_waves)
    all_p = np.array(all_p)
    all_qrs = np.array(all_qrs)
    all_t = np.array(all_t)
    all_none = np.array(all_none)
    all_seg_target = np.stack((all_p,all_qrs,all_t,all_none), axis=1) 
    all_cls_target = np.array(all_cls_target)

    # convert to torch tensors
    X_torch = torch.tensor(all_waves, dtype=torch.float32)
    X_torch = X_torch.unsqueeze(dim=1) # add channel dimension 
    y_seg_torch = torch.tensor(all_seg_target, dtype=torch.float32) 
    y_cls_torch = torch.tensor(all_cls_target, dtype=torch.int64)

    return X_torch, y_seg_torch, y_cls_torch