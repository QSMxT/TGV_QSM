"""
QSM reconstruction using Total Generalized Variation (TGV-QSM)
Kristian Bredies and Christian Langkammer
Version from December 2014 
www.neuroimaging.at
"""
import sys, getopt
from numpy import *
from qsm_tgv_cython import *
import nibabel as nib

def usage():
    print "-----"
    print "QSM reconstruction based on TGV"
    print "command line options:"
    print "  python %s -a <magnitude.nii.gz>"
    print " 		-p <phase.nii.gz>"
    print " 		-m <brainmask.nii.gz>"
    print " 		-o <outputname.nii.gz> % sys.argv[0]"
    print "		-t echo time in seconds"
    print "		-f fieldstrength in Tesla"
    print " "

def dyp(u, step=1.0):
    """Returns forward differences of a 2D/3D array with respect to x."""
    if (u.ndim == 3):
        return concatenate((u[:,1:,:] - u[:,0:-1,:], \
                            zeros([u.shape[0],1,u.shape[2]], u.dtype)), 1)/step
    else:
        return hstack((u[:,1:]-u[:,0:-1],zeros([u.shape[0],1],u.dtype)))/step

def dym(u, step=1.0):
    """Return backward differences of a 2D/3D array with respect to x."""
    if (u.ndim == 3):
        return (concatenate((u[:,:-1,:],zeros([u.shape[0],1,u.shape[2]], u.dtype)), 1) \
               - concatenate((zeros([u.shape[0],1,u.shape[2]], u.dtype), u[:,:-1,:]), 1))/step
    else:
        return (hstack((u[:,:-1],zeros([u.shape[0],1],u.dtype))) \
               - hstack((zeros([u.shape[0],1],u.dtype),u[:,:-1])))/step

def dxp(u, step=1.0):
    """Returns forward differences of a 2D/3D array with respect to y."""
    if (u.ndim == 3):
        return concatenate((u[1:,:,:] - u[0:-1,:,:], \
                            zeros([1,u.shape[1],u.shape[2]], u.dtype)), 0)/step
    else:
        return vstack((u[1:,:]-u[0:-1,:],zeros([1,u.shape[1]],u.dtype)))/step

def dxm(u, step=1.0):
    """Return backward differences of a 2D/3D array with respect to y."""
    if (u.ndim == 3):
        return (concatenate((u[:-1,:,:],zeros([1,u.shape[1],u.shape[2]], u.dtype)), 0) \
               - concatenate((zeros([1,u.shape[1],u.shape[2]], u.dtype), u[:-1,:,:]), 0))/step
    else:
        return (vstack((u[:-1,:],zeros([1,u.shape[1]],u.dtype))) \
               - vstack((zeros([1,u.shape[1]],u.dtype),u[:-1,:])))/step

def dzp(u, step=1.0):
    """Returns forward differences of a 3D array with respect to z."""
    return concatenate((u[:,:,1:] - u[:,:,0:-1], \
                        zeros([u.shape[0],u.shape[1],1], u.dtype)), 2)/step

def dzm(u, step=1.0):
    """Return backward differences of a 3D array with respect to z."""
    return (concatenate((u[:,:,:-1],zeros([u.shape[0],u.shape[1],1], u.dtype)), 2) \
           - concatenate((zeros([u.shape[0],u.shape[1],1], u.dtype),u[:,:,:-1]), 2))/step

def read_magnitude_image(fname):
    """Returns image data and resolution for a given file

    fname : file name of data to load"""

    pha_data = nib.load(fname)
    hdr = pha_data.get_header()

    data = array(pha_data.get_data())
    res = diag(hdr.get_base_affine())[0:3]

    return(data, res)

def read_phase_image(fname, mode = 0):
    """Returns image data and resolution for a given file

    fname : file name of data to load"""

    pha_data = nib.load(fname)
    hdr = pha_data.get_header()

    data = pha_data.get_data()
    if (mode == 0):
        data = array(data)/4096.0*pi
    res = diag(hdr.get_base_affine())[0:3]

    return(data, res)

def save_nifti(fname, data, res):
    affine = eye(4)
    for i in xrange(3):
        affine[i,i] = res[i]
        affine[i,3] = -0.5*(res[i]*(data.shape[i]-1))

    img = nib.Nifti1Image(data, affine)
    img.to_filename(fname)


def get_grad_phase(phase, res):
    phi = exp(1.0j*phase)
    dx = imag(dxp(phi, res[0])/phi)
    dy = imag(dyp(phi, res[1])/phi)
    dz = imag(dzp(phi, res[2])/phi)
    grad_phase = concatenate((dx[...,newaxis], dy[...,newaxis],
                              dz[...,newaxis]), axis=-1)
    return(grad_phase)

def get_laplace_phase(phase, res):
    grad_phi = get_grad_phase(phase, res)
    laplace_phi = dxm(grad_phi[...,0], res[0]) \
                  + dym(grad_phi[...,1], res[1]) \
                  + dzm(grad_phi[...,2], res[2])

    return(laplace_phi)

def get_laplace_phase2(phase, res):
    phi = exp(1.0j*phase)
    laplace_phi = dxm(dxp(phi, res[0]), res[0]) + \
                  dym(dyp(phi, res[1]), res[1]) + \
                  dzm(dzp(phi, res[2]), res[2])
    laplace_phi = imag(laplace_phi/phi)

    return(laplace_phi)

def get_best_local_h1(dx, axis=0):
    F_shape = list(dx.shape)
    F_shape[axis] -= 1
    F_shape.append(9)

    F = zeros(F_shape, dtype=dx.dtype)
    for i in xrange(3):
        for j in xrange(3):
            if (axis == 0):
                F[...,i+3*j] = (dx[:-1,...] - 2*pi*(i-1))**2 + (dx[1:,...] + 2*pi*(j-1))**2
            if (axis == 1):
                F[...,i+3*j] = (dx[:,:-1,...] - 2*pi*(i-1))**2 + (dx[:,1:,...] + 2*pi*(j-1))**2
            if (axis == 2):
                F[...,i+3*j] = (dx[:,:,:-1,...] - 2*pi*(i-1))**2 + (dx[:,:,1:,...] + 2*pi*(j-1))**2

    G = F.argmin(axis=-1)
    I = (G % 3) - 1
    J = (G / 3) - 1

    return(I,J)



def get_laplace_phase3(phase, res):
    #pad phase
    phase = concatenate((phase[0,...][newaxis,...], phase, phase[-1,...][newaxis,...]), axis=0)
    phase = concatenate((phase[:,0,...][:,newaxis,...], phase, phase[:,-1,...][:,newaxis,...]), axis=1)
    phase = concatenate((phase[:,:,0,...][:,:,newaxis,...], phase, phase[:,:,-1,...][:,:,newaxis,...]), axis=2)

    dx = (phase[1:,1:-1,1:-1] - phase[:-1,1:-1,1:-1])
    dy = (phase[1:-1,1:,1:-1] - phase[1:-1,:-1,1:-1])
    dz = (phase[1:-1,1:-1,1:] - phase[1:-1,1:-1,:-1])

    (Ix,Jx) = get_best_local_h1(dx, axis=0)
    (Iy,Jy) = get_best_local_h1(dy, axis=1)
    (Iz,Jz) = get_best_local_h1(dz, axis=2)

    laplace_phi = (-2.0*phase[1:-1,1:-1,1:-1]
                   + (phase[:-2,1:-1,1:-1] + 2*pi*Ix)
                   + (phase[2:,1:-1,1:-1] + 2*pi*Jx))/(res[0]**2)

    laplace_phi += (-2.0*phase[1:-1,1:-1,1:-1]
                    + (phase[1:-1,:-2,1:-1] + 2*pi*Iy)
                    + (phase[1:-1,2:,1:-1] + 2*pi*Jy))/(res[1]**2)

    laplace_phi += (-2.0*phase[1:-1,1:-1,1:-1]
                    + (phase[1:-1,1:-1,:-2] + 2*pi*Iz)
                    + (phase[1:-1,1:-1,2:] + 2*pi*Jz))/(res[2]**2)

    return(laplace_phi)


def erode_mask(mask):
    mask = (mask != 0)
    mask0 = mask.copy()
    mask[1:,...] *= mask0[:-1,...]
    mask[:-1,...] *= mask0[1:,...]
    mask[:,1:,...] *= mask0[:,:-1,...]
    mask[:,:-1,...] *= mask0[:,1:,...]
    mask[:,:,1:,...] *= mask0[:,:,:-1,...]
    mask[:,:,:-1,...] *= mask0[:,:,1:,...]

    return(mask)


############# main #############
def main():    
    #file_magnitude = 'epi3d_test_magni.nii.gz'
    #file_phase = 'epi3d_test_phase.nii.gz'
    #file_mask = 'epi3d_test_mask.nii.gz'
    #file_output = 'epi3d_test_QSM'
    #TE = 0.021
    #FieldStrength = 2.89

    try:
        myopts, args = getopt.getopt(sys.argv[1:], 'a:p:m:o:t:f:')
    except getopt.GetoptError as e:
        print (str(e))
        usage()
        sys.exit(2)
        
    for opt, arg in myopts:
        if opt == '-a':
            file_magnitude = arg
        elif opt == '-p':
            file_phase = arg
        elif opt == '-m':
            file_mask = arg
        elif opt == '-o':
            file_output = arg.replace('.nii.gz', '')
        elif opt == '-t':
            TE = float(arg)
        elif opt == '-f':
            FieldStrength = float(arg)
        else:
            assert False, "unhandled option"
            usage()
            sys.exit(2)

    print "----------------------------------"
    print "TVG-QSM (Dec 2014)"
    print "----------------------------------"
    print "loading files..."
    print "magni: " + file_magnitude
    print "phase: " + file_phase
    print "mask:  " + file_mask
    print "TE:    %f" % TE
    print "Field: %f" % FieldStrength
    print "save:  " + file_output + ".nii.gz"
    print "----------------------------------"

    (magnitude, res) = read_magnitude_image(file_magnitude)
    (phase, res) = read_phase_image        (file_phase, 1)
    (mask, res) = read_magnitude_image     (file_mask)
    mask = mask > 0  
    mask_orig = mask.copy()
    
    print "processing laplacian of %s ..." % file_phase
    laplace_phi0 = get_laplace_phase3(phase, res)
    #save_nifti(file_output + "_phase_laplacian.nii.gz", laplace_phi0, res)

    print "processing QSM %s ..." % file_phase
    ### parameter alpha in factors of 0.0005
    #for fac in [1.0, 2.0, 0.5]:
    for fac in [1.0]:
        #erode n pixels
        mask = mask_orig.copy()
        for i in xrange(5):
            mask = erode_mask(mask)

        ### iteration steps
        #for array_of_iterations in [100, 500, 1000, 10000]
        for array_of_iterations in [1000]:
            print "factor %f" %fac
            phi_tgv = qsm_tgv(laplace_phi0, mask, res, alpha=(0.0015*fac, 0.0005*fac), iter=array_of_iterations, vis=False)
            chi_tgv = phi_tgv/(2*pi*TE)/(FieldStrength*42.5781)
            save_nifti(file_output + "_TGV_QSM_fac%f_" % fac + "%d.nii.gz" % array_of_iterations, chi_tgv, res)
            print "==> saved " + file_output + "_TGV_QSM_fac%f_" % fac + "%d.nii.gz" % array_of_iterations			
          
if __name__ == "__main__":
    main()
