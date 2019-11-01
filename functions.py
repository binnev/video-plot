#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:54:08 2019

@author: rmn
"""

import numpy as np, pandas as pd, json, glob, os, subprocess as sp
from matplotlib import gridspec, pyplot as plt, image as mpimg
from pathlib import Path


def get_index_from_progress(sequence, N_steps, step=None, output="index"):
    """ function to calculate the index of a sequence to match the progress of
    a stepwise process. Let's say you want to scroll through an iterable
    (sequence) at the same rate as the process, such that the first step of
    the process gets the first value in the sequence, and the last step of the
    process gets the last value in the sequence, even if the length of the
    iterable is different to the number of steps in the process. This function
    calculates the index required to match sequence to process. E.g. in step
    100 of a process with N_steps = 300, we are a third of the way along
    the process. If sequence has, say, 500 values, then we need to skip some
    in order to keep pace with the 300 step process. This function calculates
    which entries in the sequence to use, making sure that the start and end
    point are included. I use this function to synchronise different data
    series with different sample rates to produce videos with a set number of
    frames.

    INPUTS:
    - sequence = the iterable which you want to match with the process. Expects
      a 1D vector.
    - step = the current step. Expects an integer. This function expects step
      to vary from 0..N_steps-1 e.g. the output of range(N_steps).
    - N_steps = the total number of steps.
    - output = flag to signify whether to output the index or the value.
      default is "index"
      - "index" = return the index of sequence to match the current step
      - "value" = return the value of sequence to match the current step
      - "indices" = return a vector of ALL the indices from sequence which
        match ALL the steps in the process. For use up front instead of in a
        loop
      - "values" = return a vector of ALL the values from sequence which match
        ALL the steps in the process.
    """

    indices = np.linspace(0, len(sequence)-1, N_steps).round().astype(int)

    if output == "index":
        return indices[step]
    elif output == "value":
        return sequence[indices[step]]
    elif output == "indices":
        return list(indices)
    elif output == "values":
        return [sequence[i] for i in indices]
    else:
        raise Exception("output flag '{}' not recognised".format(output))


# define base class for plot elements
class PlotElement:
    def __init__(self, number_of_frames, kind, data, ax, handle=None,
                 **kwargs):

        # mandatory arguments are bound to self attributes here
        self.number_of_frames = number_of_frames  # self-explanatory
        self.kind = kind  # the kind of plot element this is, e.g. Image
        self.data_source = data  # the information to be plotted
        self.ax = ax  # handle to the axes on which we want to plot

        # more general optional args are stored in a dict here
        self.kwargs = kwargs

        # call the sample method to downsample the self.data
        self.__sample__(number_of_frames)
        # call the shortcuts method to bind further (subclass-specific)
        # self.attributes
        self.__shortcuts__()

        """ if the user didn't pass a handle to an existing plot object, we
        need to create the object (e.g. plt.plot([], [], **formatting));
        this is done in the self.setup() method. """
        if handle is None:
            # call the setup method (in this parent class = do nothing)
            self.__setup__()
        else:  # if the user DID pass a handle to an existing plot object
            # don't call the setup method; just store a ref to handle
            self.handle = handle

    # sample the data (internal method)
    def __sample__(self, number_of_frames):
        data = []
        # get the indices from the first data sequence. There is always one
        indices = get_index_from_progress(self.data_source[0],
                                          number_of_frames,
                                          output="indices",)
        self.indices = indices
        # then sample all data sequences passed
        for d in self.data_source:
            data.append(get_index_from_progress(d, number_of_frames,
                                                output="values",))
        self.data = tuple([np.array(sequence) for sequence in data])

    def __shortcuts__(self):  # function to assign self.attr attributes
        pass

    def __setup__(self, *args):
        pass  # this method will be overwritten by the derived classes

    def update(self, *args):
        pass  # this method will be overwritten by the derived classes


class TextStatic(PlotElement):
    def __init__(self, number_of_frames, kind, data, ax, handle=None,
                 formatting={},  # formatting kwargs
                 position=None,  # position of the text string (x, y)
                 string_template="{: >6.2f}",  # default format of string
                 ):

        if position is None:
            # set position based on 2d or 3d axes object
            position = (0.01, .95, 0) if hasattr(ax, "zaxis") else (0.01,
                                                                    0.95)
        PlotElement.__init__(self, number_of_frames, kind, data, ax,
                             handle=handle, formatting=formatting,
                             position=position,
                             string_template=string_template, )

    def __shortcuts__(self):
        # because I defined init() for this class, I KNOW that there will
        # be an entry for "formatting" in the kwargs dict.
        f_default = dict(transform=self.ax.transAxes, ha="left", va="top",
                         family="monospace")
        # add any kwargs passed by user, overriding defaults if necessary
        f_default.update(self.kwargs["formatting"])
        self.formatting = f_default  # store updated format dict in self
        self.position = self.kwargs["position"]
        self.string_template = self.kwargs["string_template"]

    def __setup__(self):
        # get the first value of each data series passed
        string_values = [d[0] for d in self.data]
        s = self.string_template.format(*string_values)
        p = self.position
        f = self.formatting
        self.handle = self.ax.text(*p, s=s, **f)

    def update(self, frame):
        # get the value of each data sequence corresponding to current
        # frame
        string_values = [d[frame] for d in self.data]
        s = self.string_template.format(*string_values)
        self.handle.set_text(s)


class Plot(PlotElement):
    def __init__(self, number_of_frames, kind, data, ax, handle=None,
                 formatting=dict()):

        PlotElement.__init__(self, number_of_frames, kind, data, ax,
                             handle=handle, formatting=formatting)

    def __shortcuts__(self):
        self.dimensions = len(self.data)
        # establish whether this data is to be plotted as 1D/2D/3D
        if self.dimensions == 1:  # if only one data sequence was passed
            # set the indices as x data, and the data sequence as y data
            self.data = [self.indices, self.data[0]]
        if self.dimensions > 3:
            raise Exception("You passed more than 3 data sequences. I "
                            "don't know how to handle this!")

        # merge the default formatting with the user inputted formatting
        f_default = dict(marker="x", ms="10", lw=0, c="k")
        f_default.update(self.kwargs["formatting"])
        self.formatting = f_default  # store updated format dict in self

    def __setup__(self):
        # sidestep the [x,],[y,],[z,] nonsense by initialising the plot
        # element with as many empty lists as there are dimensions
        self.handle, = self.ax.plot(*[[] for i in range(self.dimensions)],
                                    **self.formatting)

    def update(self, frame):
        # get the current value of the first two (x & y) data sequences
        xy = [seq[frame] for seq in self.data[:2]]
        self.handle.set_data(*xy)
        if self.dimensions == 3:  # if this is 3d data
            z = self.data[2][frame]
            self.handle.set_3d_properties(z)


class Image(PlotElement):
    def __init__(self, number_of_frames, kind, data, ax,
                 # optional kwargs:
                 handle=None,
                 formatting={},  # formatting kwargs
                 crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
                 ):

        PlotElement.__init__(self, number_of_frames, kind, data, ax,
                             handle=handle, formatting=formatting,
                             crop_left=crop_left, crop_right=crop_right,
                             crop_top=crop_top, crop_bottom=crop_bottom,
                             )

    def __shortcuts__(self):
        f_default = dict()
        f_default.update(self.kwargs["formatting"])
        self.formatting = f_default
        self.crop_left = self.kwargs["crop_left"]
        self.crop_right = self.kwargs["crop_right"]
        self.crop_top = self.kwargs["crop_top"]
        self.crop_bottom = self.kwargs["crop_bottom"]

    def __setup__(self):
        # set up the dummy image
        f = self.formatting
        datum = self.data[0][0]  # 0 for 1st data seq; 0 for first frame
        if type(datum) == str:
            img = mpimg.imread(datum)
        else:
            img = datum
        width = img.shape[1]
        height = img.shape[0]
#        self.ax.set(xlim=(self.crop_left, width-self.crop_right),
#                    ylim=(height-self.crop_bottom, self.crop_top))
        self.handle = self.ax.imshow(img, **self.formatting)

    def update(self, frame):
        datum = self.data[0][frame]
        # if data is string, assume it's points to an image file and load it
        if type(datum) == str:
            img = mpimg.imread(path)
        # otherwise, assume it's already an imag
        else:
            img = datum
#        self.handle.set_data(img)  # this doesn't work with colourmaps
        self.handle.remove()
        self.handle = self.ax.imshow(img, **self.formatting)


class FollowerAxes(PlotElement):
    def __init__(self, number_of_frames, kind, data, ax,
                 # optional kwargs:
                 width=None, height=None):
        super().__init__(number_of_frames, kind, data, ax, width=width,
                         height=height)

    def __shortcuts__(self):
        data = self.data
        self.x = data[0]
        self.y = data[1]
        self.width = self.kwargs["width"]
        self.height = self.kwargs["height"]
        if (self.width is None) and (self.height is None):
            raise Exception("you need to pass either 'height' or 'width'; "
                            "you can't pass neither.")

        w, h = self.width, self.height
        if self.width is not None:
            self.xlim = [(x-w/2, x+w/2) for x in self.x]
        else:
            self.xlim = None

        if self.height is not None:
            self.ylim = [(y-h/2, y+h/2) for y in self.y]
        else:
            self.ylim = None

    def __setup__(self):
        # this class acts only on the axes, so doesn't need to instantiate
        # an object
        pass

    def update(self, frame):
        if self.xlim is not None:
            plt.setp(self.ax, xlim=self.xlim[frame])
        if self.ylim is not None:
            plt.setp(self.ax, ylim=self.ylim[frame])


def VideoPlot(fig, number_of_frames, *plot_elements,
              output_path=None, output_filename="animation",
              output_filetype="mp4", savefig_kwargs={},
              video_frame_format="jpg",
              ffmpeg_input_options="-framerate 10",
              ffmpeg_output_options='-c:v libx264 -pix_fmt yuvj420p -r 30 -vf scale=trunc(iw/2)*2:trunc(ih/2)*2',
              ffmpeg_global_options=("-y"),
              ask_overwrite_permission=True,
              delete_images=False):
    """
    Function to create animated video plots. The user prepares a figure (or
    subplots) and plots any static (non-animated) stuff on this. The user
    passes this, and a list of instructions for dynamic (animated) plot
    elements to this function. This function then updates the figure and saves
    the figure each time; creating a series of images which will become the
    video. This function assumes that the data has been synchronised by the
    user -- i.e. the data sequences passed in share the same start/end point,
    and can therefore be safely sampled relative to the start/end point.

    The user specifies how many video frames are required, and this function
    samples all data sequences such that there are as many values as there are
    frames in the video. If an input data sequence has more datapoints than the
    number of video frames, then the sample will be an evenly spaced subset of
    the original sequence. If the input data sequence has fewer datapoints than
    the number of video frames, then some datapoints will be used twice in
    order to match the length of the video.

    This function can handle the following kinds of plot element:
    - TextStatic
    - Plot
    - Image
    - FollowerAxes
    Each of which have their own class to handle the plotting. They are derived
    from a base class "PlotElement" which defines how the data sequences are
    sampled. The user passes in dictionaries of instructions, and these are
    instantiated as plot element classes according to what kind they are. This
    is covered in more detail later.

    INPUTS:
    - fig : handle to the figure object which will be modified. This is
      necessary because of the fig.savefig() calls to save the image frames.
    - number_of_frames : number of frames in the output video. Used for
      sampling the input data sequences.
    - *plot_elements : any further positional args are taken to be plot
      elements. The result is a list [plot_elements] containing dicts, each of
      which contains instructions for that plot element. More detail on this
      later.
    - output_path="" : path to the folder in which the outputs are to be saved.
      defaults to the current python working directory.
    - output_filename="animation" : the main descriptor which the user wants to
      use as the filename. Minus the file extension. E.g. "1.75mm_2_P_4".
    - output_filetype="mp4" : self explanatory. This just gets passed straight
      to ffmpeg.
    - savefig_kwargs={} : kwargs dict which is passed to savefig via **kwargs
    - ffmpeg_input_options="-framerate 10" : pass optional ffmpeg input options
      here. The framerate specified here is the rate at which the original
      images will appear in the video -- i.e. the effective framerate which the
      viewer will see. If you want to change the framerate of the videos, you
      should change this one, not the output framerate.
    - ffmpeg_output_options="-c:v libx264 -pix_fmt yuv420p -r 10" : optional
      ffmpeg outputs. The output framerate specified here controls the number
      of frames in the output video. ffmpeg will duplicate/drop frames to
      achieve this. It is important that the output framerate is high enough;
      having it too low makes the videos unplayable (VLC player can't deal with
      fps<10). If you want to change the framerate, it's probably best to leave
      this parameter alone and change the input framerate instead. So long as
      the output framerate is higher than the input framerate, you're good.
      The -vf option deals with input images that have an odd number of
      pixels in height or width. It basically pads an extra black pixel if
      necessary.
    - ask_overwrite_permission=True : flag to check if the function should ask
      the user's permission before overwriting any existing image/video files.

    PLOT ELEMENTS INPUT FORMAT:
        Plot elements can be entered as individual positional args, or by
        emptying a list of plot elements using the *() syntax.
        Each plot element is essentially a dictionary of instructions. Example
        dicts for each type of plot element are given below, along with any
        default values which the classes use if no optional args are passed

    - PlotElement : dict(kind, data, ax, handle)
      This is the generic parent class. You won't actually use this, but it
      does handle the common arguments, so they are explained here.
      - kind : str describing the type of plot element e.g. "TextStatic"
      - data : tuple containing data sequences e.g. (x,); (x, y); (x, y, z)
        where x, y, z are vectors of data values. These vectors can be as long
        as you like; they will be sampled by the PlotElement class.
      - ax : handle to the axes object on which this plot element is to be
        plotted
      - handle=None : handle to an existing initialised object. E.g. if the
        user wants to set up their plot object outside the function, and just
        use the function to update the data values for each frame. This saves
        the user from having to pass in a formatting dict. Defaults to None.

    - TextStatic : dict(kind, data, ax, handle, formatting, position,
                        string_template)
      Text object for which the position doesn't change, but the value of the
      text is updated every frame, from data.
      - kind : "TextStatic"
      - data : data tuple, as described above. Can have as many entries as you
        want, so long as each corresponds to a "{}" in the string_template!
      - formatting={} : dict of kwargs to pass to the matplotlib.text object
      - position=None : position of the text string: (x, y[, z]). Fixed.
      - string_template="{: >6.2f}" : the formatting template for the text
        display. You need to put in as many "{}" fields as there are entries in
        the data arg. E.g. for data=(a,b,c,d); "{}, {}, {}, {}".format(*data)
        currently only supports the positional input (*data); not **kwarg input

    - Plot : dict(kind, data, ax, handle, formatting)
      All-purpose moving plot.
      - kind : "Plot"
      - data : (x[, y[, z]]). The user specifies what data is to be used
        and Plot automatically counts the number of data sequences to work out
        if a 1D/2D/3D plot is required. The data sequences can be sequences of
        single values (for a moving marker) or sequences of vectors of values
        (for moving lines). Data sequences are passed as a tuple:
            data=(x, )        # 1D
            data=(x, y, )     # 2D
            data=(x, y, z)    # 3D
        For a marker, each sequence has the format:
            x = [x1, x2, x3, ... , xi]  # i = number_of_frames
        For a moving line, each sequence in x is itself a sequence:
            xn = [xi1, xi2, xi3, ... , xij]  # j = number of points in the line
      - formatting : dict of kwargs to be passed to the plt.plot command

    - Image : dict(kind, data, ax, handle, formatting, crop_left, crop_right,
                   crop_top, crop_bottom)
      A sequence of images to become a video.
      - kind : "Image"
      - data : Images are passed as a sequence of *paths* to sequential images:
        ("c:/images/img1", "c:/images/img2", "c:/images/img3", ...).
        This means they can be loaded on the fly. Pre-loading many images
        proved to be way slower. The downsampling of data sequences to match
        the number of video frames helps with this. For example, the user can
        pass in a list of paths to 1000 images, and specify that they want a
        video with 100 frames. By passing in the list of paths, and only
        loading 100 of them, this operation is 10x faster than loading all 1000
        images and passing them in to be sampled.
      - formatting : a dict of kwargs to be passed to the matplotlib.imshow
      - crop variables : integer values.

    - FollowerAxes : dict(kind, data, ax)
      Modifies the x/y limits of an axis to "follow" an x,y sequence
      - kind : "FollowerAxes"
      - data :
          data[0] = x (vector)
          data[1] = y (vector)
          data[2] = width
          data[3] = height

          x and y are vectors describing the location of the point to follow.
          This is the same type of data you would pass to the Plot element.

          width and height describe the width and height of the box around the
          point. For a constant width, pass in a single value inside a list,
          e.g.:
              width = (0.5,)
          For a changing width (e.g. zooming in throughout the video), pass a
          list of values the same format as x and y.

    """

    # convert output_path to Path object, or set to current working directory if None
    output_path = Path(output_path) if (output_path is not None) else Path.cwd()

    # instantiate class for each type of plot element
    temp = []
    for e in plot_elements:
        if e["kind"] == "TextStatic":
            temp.append(TextStatic(number_of_frames, **e))
        if e["kind"] == "Plot":
            temp.append(Plot(number_of_frames, **e))
        if e["kind"] == "Image":
            temp.append(Image(number_of_frames, **e))
        if e["kind"] == "FollowerAxes":
            temp.append(FollowerAxes(number_of_frames, **e))
    plot_elements = temp

    overwrite_fig = None

    # this function determines what happens during the animation. What updates.
    def animate(frame, video_frame_format):

        # refer to the overwrite_fig from *outside* this function
        nonlocal overwrite_fig
        # create the figure name using the frame number
        fig_name = output_filename+"_{:03d}.{}".format(frame, video_frame_format)
        fig_file = output_path / fig_name

        # if this file already exists
        if fig_file.is_file():
            # if the user has told me to ask for permission
            if ask_overwrite_permission is True:
                # if overwrite_fig hasn't been set by the user yet
                if overwrite_fig is None:
                    # prompt the user
                    inp = input("The file {} already exists. Shall I overwrite"
                                " it?\n"
                                "enter = overwrite files (this will also "
                                "overwrite subsequent frames)\n"
                                "'v' = leave the existing video frames alone, "
                                "but try to make a video from them later\n"
                                "'n' or anything else = don't overwrite and "
                                "stop program \n".format(fig_file))
                    if inp == "":  # if yes (enter),
                        # set overwrite toggle=True and exit to save the figure
                        overwrite_fig = True
                    elif inp == "v":  # if v for video
                        print("OK, I won't overwrite the existing figures but "
                              "I'll still try and make the video.\n")
                        # set overwrite_fig to False, but don't raise an Exc.
                        overwrite_fig = False
                    elif inp == "n":  # if no,
                        # don't overwrite and stop the program
                        return "stop"  # return stop signal to calling script
                        print("You don't want me to overwrite the "
                              "existing video so I'm going to stop.\n")
                    else:  # if input not recognised
                        # don't overwrite and stop the program
                        raise Exception("input '{}' not recognised".format(inp))
                if overwrite_fig is False:
                    return None  # exit the func without saving figure

        # update the plot elements and save the figures
        for elm in plot_elements:  # scroll through plot elements
            elm.update(frame)  # call each element's update() method

        print("saving {} ({}/{})".format(fig_file.as_posix(), frame+1, number_of_frames))
        fig.savefig(fig_file.as_posix(), **savefig_kwargs)

    # animate the plot
    for frame in range(number_of_frames):
        a = animate(frame, video_frame_format)
        if a == "stop":  # if the user has said no to overwriting
            # exit the function right here. Don't even ask about the videos.
            return None

    # stitch the images together into a video with ffmpeg
    video_name = output_filename+"."+output_filetype
    video_file = output_path / video_name
    if video_file.is_file():
        if ask_overwrite_permission is True:
            inp = input("The file {} already exists. Shall I overwrite it?\n"
                        " enter = overwrite \n"
                        " 'n' = don't overwrite and stop the "
                        "program\n".format(video_file))
            if inp == "":  # if yes,
                pass  # do nothing; video will be exported.
            elif inp == "n":  # if no,
                # don't overwrite and stop the program
                print("You don't want me to overwrite the "
                      "existing video so I'm going to stop.\n")
                return None  # exit the func here without making video
            else:  # if input not recognised
                raise Exception("input '{}' not recognised".format(inp))

    print("making video "+video_name)
    image_mask = output_filename+f"_%03d.{video_frame_format}"

    # construct ffmpeg command
    cmd = ["ffmpeg",
           *ffmpeg_global_options.split(" "),
           *ffmpeg_input_options.split(" "),
           "-i", (output_path/image_mask).as_posix(),
           *ffmpeg_output_options.split(" "),
           video_file.as_posix(),
           ]

    print(" ".join(cmd))

    # execute command
    try:
        output = sp.check_output(cmd)
    except sp.CalledProcessError as e:
        raise Exception("command {} encountered an "
                        "error code {}:\n{}".format(e.cmd, e.returncode,
                                                    e.output))

    # delete image files if user asked for it
    if delete_images:
        for frame in range(number_of_frames):
            fig_name = output_filename+"_{:03d}.{}".format(frame, video_frame_format)
            fig_file = output_path / fig_name
            if fig_file.is_file():
                os.remove(fig_file)
