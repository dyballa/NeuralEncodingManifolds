import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def createFlowDataset(categories, topdir, mydirs, orig_shape, input_shape, scl_factor, N_INSTANCES, trial_len, stride):
    scld_shape = tuple((np.array(orig_shape)*scl_factor).astype('int'))
    NDIRS = len(mydirs)
    frames_per_stim = (trial_len//stride)
    
    shift_foos = {'0':lambda im,step: np.roll(im,step,1),
                  '45':lambda im,step: np.roll(np.roll(im,step,1),-step,0),
                  '90':lambda im,step: np.roll(im,-step,0),
                  '135':lambda im,step: np.roll(np.roll(im,-step,1),-step,0),
                  '180':lambda im,step: np.roll(im,-step,1),
                  '225':lambda im,step: np.roll(np.roll(im,-step,1),step,0),
                  '270':lambda im,step: np.roll(im,step,0),
                  '315':lambda im,step: np.roll(np.roll(im,step,0),step,1),
                 }
    
    flow_datasets = {}

    for inst_i in range(N_INSTANCES):   
        print('*INSTANCE',inst_i,end=' ',flush=True)
        for cat in categories: 
            print('.',end='',flush=True)
            stim_arrays = None

            for di,d in enumerate(mydirs):

                image_path = f'{topdir}/{cat}_inst{inst_i}/{d}/0.png'
                img = Image.open(image_path)

                assert orig_shape == img.size

                if scl_factor != 1:
                    img = img.resize(scld_shape, Image.Resampling.LANCZOS)

                #cropping idxs
                w,h = img.size
                assert w == scld_shape[0] and h == scld_shape[1]
                i0, j0 = h//2-input_shape[0]//2, w//2-input_shape[1]//2
                i1, j1 = i0 + input_shape[0], j0 + input_shape[1]

                img_array = np.array(img)[:,:,0] #since grayscale, use only one channel

                for fi in range(0,trial_len,stride):
                    #shift full img
                    shifted_img = shift_foos[d](img_array,fi)
                    #crop from center
                    shifted_img = shifted_img[i0:i1,j0:j1]
                    #save
                    if stim_arrays is None:
                        stim_arrays = np.zeros((NDIRS*frames_per_stim,shifted_img.size))
                    stim_arrays[di*frames_per_stim+fi] = shifted_img.ravel()

            if inst_i not in flow_datasets:
                flow_datasets[inst_i] = stim_arrays
            else:
                flow_datasets[inst_i] = np.concatenate([flow_datasets[inst_i],stim_arrays])

        print()
    return flow_datasets

def from0to1(arr):
    arr = np.asanyarray(arr)
    arr[np.isclose(arr,0)] = 1
    return arr

def subps(nrows,ncols,rowsz=3,colsz=4,d3=False,axlist=False):
    if d3:
        f = plt.figure(figsize=(ncols*colsz,nrows*rowsz))
        axes = [[f.add_subplot(nrows,ncols,ri*ncols+ci+1, projection='3d') for ci in range(ncols)] \
                for ri in range(nrows)]
        if nrows == 1:
            axes = axes[0]
            if ncols == 1:
                axes = axes[0]
    else:
        f,axes = plt.subplots(nrows,ncols,figsize=(ncols*colsz,nrows*rowsz))
    if axlist and ncols*nrows == 1:
        
        axes = [axes]
    return f,axes

def twx():
    ax = plt.subplot(111)
    return ax, ax.twinx()
    
def npprint(a,precision=3):
    with np.printoptions(precision=precision, suppress=True):
        print(a)
    return

