import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat

def createFlowDataset(categories, topdir, mydirs, orig_shape, input_shape, scl_factor, N_INSTANCES, trial_len, stride):
    scld_shape = tuple((np.array(orig_shape)*scl_factor).astype('int'))
    NDIRS = len(mydirs)
    frames_per_stim = int(np.ceil(trial_len/stride))
    
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

                for fii,fi in enumerate(range(0,trial_len,stride)):
                    #shift full img
                    shifted_img = shift_foos[d](img_array,fi)
                    #crop from center
                    shifted_img = shifted_img[i0:i1,j0:j1]
                    #save
                    if stim_arrays is None:
                        stim_arrays = np.zeros((NDIRS*frames_per_stim,shifted_img.size))
                    stim_arrays[di*frames_per_stim+fii] = shifted_img.ravel()


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

def plot_image(orig_image, fig_sz, ax=None, vmin=None, vmax=None,
              axis_off=True):


    image = orig_image.copy()
    
    assert image.min() >= 0
    if image.max() <= 1:
        image = (image*255).astype('int32')
    else:
        image = image.astype('int32')

    imsiz = image.shape[1]
    
    if ax is None:
        plt.figure(figsize=(fig_sz,fig_sz))
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(image, vmin=vmin, vmax=vmax)
        if axis_off:
            ax.axis('off')

def plot_images(images_, fig_sz=2, nrows=None, labels=None, vmin=None, vmax=None):

    images = images_.copy()
    if nrows is not None:
        ncols = int(np.ceil(len(images)/nrows))
    else:
        nrows = int(np.floor(np.sqrt(len(images))))
        ncols = int(np.ceil(np.sqrt(len(images))))
        
    if ncols*nrows < len(images):
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_sz*ncols,fig_sz*nrows))


    for i, ax in enumerate(axes.ravel()):
        if i >= len(images):
            ax.axis('off')
            continue

        img = images[i]
        

        ax.set_title('%d' % i,size=11)
        
        if labels is not None:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax, axis_off=False)
            ax.set_xlabel(labels[i],size=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax)
            
    fig.tight_layout()
    plt.show() 



def predict_images(images_data, fig_sz=None, nrows=None):
    imgs = []
    top1s = []
    for image_data in images_data:
        if type(image_data) == str:
            # Load and resize the image using PIL.
            img = Image.open(image_data)
            img_resized = img.resize(input_shape, Image.Resampling.LANCZOS)
            # Convert the PIL image to a numpy-array with the proper shape.
            img_array = np.array(img_resized)
        else:
            assert image_data.shape == (input_shape[0],input_shape[1],3)
            img_array = image_data.copy()
            
        imgs.append(img_array.astype('uint8'))
        
        img_array = preprocess_input(img_array)

        pred = model.predict(np.expand_dims(img_array, axis=0),verbose=0)

        pred_decoded = decode_predictions(pred)[0]

        code, name, score = pred_decoded[0]
        top1s.append("{0:>6.2%} : {1}".format(score, name))
        
    plot_images(np.array(imgs), fig_sz, nrows, labels=top1s, vmin=0, vmax=255)

def khatri_rao(matrices):
    """Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list of ndarray

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the
        product.

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))





def loadPreComputedCP(tensorname,basedir,specificFs=[],NMODES=3,verbose=True):
    
    """Converts the .mat files produced by tensor decomposition (see `matlab/run_permcp.m`)
    and aggregates the results from multiple choices of F (number of components) and
    initializations into a single Python dict."""
    
    
    preComputed = {}

    #parse the F (# of factors) from the file name
    def parseF(s):
        assert s[-4:] == '.mat'
        extn = 4
        F_str = s[:-extn].split('_F')[-1]
        return int(F_str)
    
    query = '%s/%s*_F*.mat' % (basedir,tensorname)
 
    queryfiles = glob(query)

    F_ = None
    counted_reps = 0
    for r in sorted(queryfiles):
        
        F = parseF(r)

        if specificFs and F not in specificFs:
            continue

        if F != F_:
            if int(verbose) > 1:
                print()
            elif int(verbose) == 1 and counted_reps > 0:
                print(f'({counted_reps})',end=' ')
            
            counted_reps = 0 #reset
            
            if verbose: print(f'F{F}:',end=' ')
            F_ = F
        #if another file from the same F, keep updating the number of reps
        


        matfile = loadmat(r)
        assert matfile['factors'][0,0].shape[1] == NMODES

        nreps = len(matfile['factors'][0])

        factors = {counted_reps+rep:matfile['factors'][0][rep].squeeze() for rep in range(nreps)}
        lambdas = {counted_reps+rep:matfile['lams'][0][rep].squeeze() for rep in range(nreps)}
        objs = {counted_reps+rep:matfile['objs'][0][rep].squeeze() for rep in range(nreps)}


        F_precomp = {'all_factors':factors, 'all_lambdas':lambdas, 'all_objs':objs}


        counted_reps += nreps
        
        if F not in preComputed:
            preComputed[F] = F_precomp.copy()

        else:#merge results
            for dkey in F_precomp.keys():
                preComputed[F][dkey].update(F_precomp[dkey])

    if int(verbose) == 1 and counted_reps > 0:
        print(f'({counted_reps})')
        
    Fs = sorted(preComputed.keys())
    return preComputed,Fs

