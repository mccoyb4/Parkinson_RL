# Functions from Spynoza and Knapen lab 

def savgol_filter(in_file, polyorder=3, deriv=0, window_length=120, tr=None):
    """ Applies a savitsky-golay filter to a nifti-file.
    Fits a savitsky-golay filter to a 4D fMRI nifti-file and subtracts the
    fitted data from the original data to effectively remove low-frequency
    signals.
    Parameters
    ----------
    in_file : str
        Absolute path to nifti-file.
    polyorder : int (default: 3)
        Order of polynomials to use in filter.
    deriv : int (default: 0)
        Number of derivatives to use in filter.
    window_length : int (default: 120)
        Window length in seconds.
    Returns
    -------
    out_file : str
        Absolute path to filtered nifti-file.
    """

    import nibabel as nib
    from scipy.signal import savgol_filter
    import numpy as np
    import os

    data = nib.load(in_file)
    dims = data.shape
    affine = data.affine
    header = data.header

    if tr is None:  # if TR is not set
        tr = data.header['pixdim'][4]

    # TR must be in seconds
    if tr < 0.01:
        tr = np.round(tr * 1000, decimals=3)
    if tr > 20:
        tr = tr / 1000.0

    window = np.int(window_length / tr)

    # Window must be odd
    if window % 2 == 0:
        window += 1

    data = data.get_data().reshape((np.prod(data.shape[:-1]), data.shape[-1]))
    data_filt = savgol_filter(data, window_length=window, polyorder=polyorder,
                              deriv=deriv, axis=1, mode='nearest')

    data_filt = data - data_filt + data_filt.mean(axis=-1)[:, np.newaxis]
    data_filt = data_filt.reshape(dims)
    img = nib.Nifti1Image(data_filt, affine=affine, header=header)
    new_name = os.path.basename(in_file).split('.')[:-2][0] + '_sg.nii.gz'
    #out_file = os.path.abspath(new_name)
    out_file = os.path.join(os.path.dirname(in_file),new_name) # save in same directory as in_file
    print('output file: ', out_file)
    nib.save(img, out_file)
    return out_file

def average_signal(in_files, func='mean', output_filename=None):
    """Takes a list of 4D fMRI nifti-files and averages them.

    Parameters
    ----------
    in_files : list
        Absolute paths to nifti-files.
    func : string ['mean', 'median'] (default: 'mean')
        the function used to calculate the 'average'
    output_filename : str
        path to output filename
    Returns
    -------
    out_file : str
        Absolute path to average nifti-file.
    """

    import nibabel as nib
    import numpy as np
    import os
    import bottleneck as bn

    template_data = nib.load(in_files[0])
    dims = template_data.shape
    affine = template_data.affine
    header = template_data.header
    all_data = np.zeros([len(in_files)] + list(dims))

    for i in range(len(in_files)):
        d = nib.load(in_files[i])
        all_data[i] = d.get_data()

    if func == 'mean':
        av_data = all_data.mean(axis=0)
    elif func == 'median':
        # weird reshape operation which hopeully fixes an issue in which
        # np.median hogs memory and lasts amazingly long
        all_data = all_data.reshape((len(in_files), -1))
        av_data = bn.nanmedian(all_data, axis=0)
        av_data = av_data.reshape(dims)

    img = nib.Nifti1Image(av_data, affine=affine, header=header)

    if output_filename == None:
        new_name = os.path.basename(in_files[0]).split('.')[:-2][
                       0] + '_av.nii.gz'
        #out_file = os.path.abspath(new_name)
        out_file = os.path.join(os.path.dirname(in_file),new_name) # save in same directory as in_file
    else:
        #out_file = os.path.abspath(output_filename)
        out_file = os.path.join(os.path.dirname(in_file),output_filename)
    nib.save(img, out_file)

    return out_file

def percent_signal_change(in_file,avg_signal):
    """
    Converts a time series into a %-signal change series
    Divide by mean of series, multiply by 100, subtract 100
    Returns a zero-centered in percent times series
    """
    psc_file = (np.divide(in_file,avg_signal[:,:,:,np.newaxis],
        out=np.zeros_like(in_file), where=avg_signal[:,:,:,np.newaxis]!=0)* 100)-100
    return psc_file

def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
