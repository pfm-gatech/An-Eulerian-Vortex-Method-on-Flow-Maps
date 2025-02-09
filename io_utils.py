import numpy as np
import imageio.v2 as imageio
import os
import shutil
import glob
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import vtk

# remove everything in dir
def remove_everything_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# for writing images
def to_numpy(x):
    return x.detach().cpu().numpy()
    
def to8b(x):
    return (255*np.clip(x,0,1)).astype(np.uint8)

def comp_vort(vel_img): # compute the curl of velocity
    W, H, _ = vel_img.shape
    dx = 1./H
    u = vel_img[...,0]
    v = vel_img[...,1]
    dvdx = 1/(2*dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
    dudy = 1/(2*dx) * (u[1:-1, 2:] - u[1:-1, :-2])
    vort_img = dvdx - dudy
    return vort_img


def write_image(img_xy, outdir, i):
    #pass
    img = np.flip(img_xy.transpose([1,0,2]), 0)
    # take the predicted c map
    img8b = to8b(img)
    if img8b.shape[-1] == 1:
        img8b = np.concatenate([img8b, img8b, img8b], axis = -1)
    # print(img8b.shape)
    save_filepath = os.path.join(outdir, '{:03d}.jpg'.format(i))
    imageio.imwrite(save_filepath, img8b)

def write_field(img, outdir, i, vmin = 0, vmax = 1):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)

def write_quiver(uv_pred_xy, outdir, i, scale = 1, skip = 1):
    if scale is None:
        relative = True 
    else:
        relative = False

    W, H, D = uv_pred_xy.shape
    x = np.arange(0, W, skip)
    y = np.arange(0, H, skip)
    if x[-1] < W-1:
        x = np.append(x, np.array([W-1]))
    if y[-1] < H-1:
        y = np.append(y, np.array([H-1]))

    _X, _Y = np.meshgrid(x, y)

    Y = _Y / H
    X = _X / H
    vel_field = uv_pred_xy.transpose([1,0,2]) # tranpose into Y-X
    speed = np.linalg.norm(vel_field, axis = -1)
    x_to_y = W/H
    y_size = 7
    x_size = x_to_y * y_size + 1
    fig = plt.figure(num=2, figsize=(x_size, y_size), clear=True)
    ax = fig.add_subplot()
    fig.subplots_adjust(0.05,0.07,0.98,0.98)
    scale = 1./scale
    if relative:
        scale = None

    plt.quiver(X, Y, \
               vel_field[...,0][_Y, _X], vel_field[...,1][_Y, _X], scale=scale, width = 1.e-4 * x_size)
         
    plt.gca().set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([0., x_to_y])
    plt.ylim([0., 1.])
    #adding text inside the plot
    plt.text(1.3, 1.3, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)

def write_vtk(numpy_array, outdir, i, name):
    data = numpy_array.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName(name)
    imageData.GetPointData().SetScalars(vtkDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_vtk_smoke(numpy_array, outdir, i, name):
    # print(numpy_array.shape)
    data_0 = numpy_array[:, :, :, 0].squeeze()
    data_1 = numpy_array[:, :, :, 1].squeeze()
    data_2 = numpy_array[:, :, :, 2].squeeze()
    data_3 = numpy_array[:, :, :, 3].squeeze()
    data_4 = numpy_array[:, :, :, 4].squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data_0.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray_0 = numpy_support.numpy_to_vtk(data_0.ravel(order = "F"), deep=True)
    vtkDataArray_0.SetName(name + "0")
    imageData.GetPointData().SetScalars(vtkDataArray_0)

    vtkDataArray_1 = numpy_support.numpy_to_vtk(data_1.ravel(order = "F"), deep=True)
    vtkDataArray_1.SetName(name + "1")
    imageData.GetPointData().AddArray(vtkDataArray_1)

    vtkDataArray_2 = numpy_support.numpy_to_vtk(data_2.ravel(order = "F"), deep=True)
    vtkDataArray_2.SetName(name + "2")
    imageData.GetPointData().AddArray(vtkDataArray_2)

    vtkDataArray_3 = numpy_support.numpy_to_vtk(data_3.ravel(order = "F"), deep=True)
    vtkDataArray_3.SetName(name + "3")
    imageData.GetPointData().AddArray(vtkDataArray_3)

    vtkDataArray_4 = numpy_support.numpy_to_vtk(data_4.ravel(order = "F"), deep=True)
    vtkDataArray_4.SetName(name + "4")
    imageData.GetPointData().AddArray(vtkDataArray_4)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_w_and_smoke_original(w_numpy, smoke_numpy, outdir, i):
    data = w_numpy.squeeze()
    smoke_data = smoke_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # add smoke
    smokeDataArray = numpy_support.numpy_to_vtk(smoke_data.ravel(order = "F"), deep=True)
    smokeDataArray.SetName("smoke")
    imageData.GetPointData().AddArray(smokeDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()