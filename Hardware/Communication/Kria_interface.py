# imports
import pynq              
import struct

### functions to transfer between datatypes ###

# float -> binary representation -> integer
def float_to_int(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

# integer -> binary representation -> integer
def int_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

# 2 16-bit integers -> 1 32-bit integer
def combine_int16(a, b):
    return (b & 0xFF) | ((a & 0xFF) << 16)

# 1 32-bit integer -> 2 16-bit integers
def split_int16(A):
    b = A & 0xFF
    a = (A >> 16) & 0xFF
    return a, b

# 4 8-bit integers -> 1 32-bit integer
def combine_int8(a, b, c, d):
    return (d & 0xFF) | ((c & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24)

# 1 32-bit integer -> 4 4-bit integers
def split_int8(A):
    d = A & 0xFF
    c = (A >> 8) & 0xFF
    b = (A >> 16) & 0xFF
    a = (A >> 24) & 0xFF
    return a, b, c, d
    

### Hardware ###

# Idk of this works tbh
class KRIA():
    # initialise overlay, memory and handles
    # overlay is a string with the filename of the .bit file of the overlay
    def __init__(self, overlay):
        kria = pynq.Overlay(overlay)                            

        # Base addresses and address range of the Block Ram
        # bram.write(offset, value): writes values at the address with the specified offset, offset is multiple of 4, value should be an integer
        # bram.read(offset): reads values at the address with the specified offset
        self.bram_inputs  = pynq.MMIO(0xB000_0000,0x2_0000)
        self.bram_outputs = pynq.MMIO(0xB002_0000,0x2_0000)

        # Control signals written to and read from PL, are configured as input or output as seen from PS
        # output.write(0,1): assert signal
        # output.write(0,0): deassert signal
        # input.read(): returns value of the signal
        self.reset = kria.restart                   # output        
        self.load_parameters = kria.load_parameters # output
        self.start = kria.start                     # output
        self.done  = kria.done                      # input



    # reset all FSMs and registers (not BRAM)
    def resetAll(self):
        self.reset.write(0,1)
        self.reset.write(0,0)



    # write new parameters of the encoder via BRAM to registers in PL
    # num_parameter: number of total parameters, all kernels and biases
    # parameters: array/list of all these parameters, should be flat.       ToDo: figure out which parameter is what for connections in hardware
    def writeParameters(self, numParameters, parameters, quantizationEnabled):
        offset = 0

        # if 16 bits per parameter, 2 parameters can be send at a time
        # if 32 bits per parameter, 1 parameter at a time
        # floats are tranformed into integers (and combined if necessary) before being written
        # ToDo: check if the new format is compatible with the current transfer method to int
        if quantizationEnabled:         
            for i in range(int(numParameters/2)):
                value = combine_int16(float_to_int(parameters[i]),float_to_int(parameters[i+1]))
                self.bram_inputs.write(offset,value)
                offset = offset + 4
        else:
            for i in range(numParameters):
                value = float_to_int(parameters[i])
                self.bram_inputs.write(offset,value)
                offset = offset + 4
        
        # Tell PL to load values into registers
        self.load_parameters.write(0,1)         
        self.load_parameters.write(0,0)



    # load images into the BRAM, used to load a batch or an entire buffer
    # images: array/list of the images that should be written to the BRAM, should be flat
    def load_buffer(self, numParameters, bufferSize, imageDim, images, quantizationEnabled):
        # number of transfers, 4 pixels fit into 1 word
        num_words = int(imageDim * imageDim * bufferSize / 4)

        # the number of parameters decides where the images are stored (kernel and images are in the same BRAM)
        if quantizationEnabled:
            offset = int(numParameters * 2)
        else:
            offset = int(numParameters * 4)

        # integers are combined into groups of 4 and written to the BRAM
        for i in range(num_words):
            value = combine_int8(images[i],images[i+1],images[i+2],images[i+3])
            self.bram_inputs.write(offset, value)
            offset = offset + 4



    # renews a section of the data stored in the BRAM
    # epoch: necessary to select which part to update, updates based on FIFO system, numImages should be a divider of bufferSize    
    def update_buffer(self, epoch, bufferSize, imageDim, numImages, images):     
        # number of transfers, 4 pixels into 1 word
        num_words = int(imageDim * imageDim * numImages / 4)

        # determine which section should be replaced and calculate the corresponding offset
        selection = epoch % (bufferSize/numImages)
        offset = selection * numImages
        
        # integers are combined into groups of 4 and written to the BRAM
        for i in range(num_words):
            value = combine_int8(images[i],images[i+1],images[i+2],images[i+3])
            self.bram_inputs.write(offset,value)
            offset = offset + 4



    # performs augmentations and convolutions on the images in the buffer with the loaded kernels
    # outputSize: determines how many values have to be transferred back to PS
    def HardwareAugmentConvolve(self, outputSize):     
        # give signal to start the process                      
        self.start.write(0,1)
        self.start.write(0,0)

        # check if hardware is done with all images
        while(self.done.read()==0):
            pass

        # read the output
        offset = 0
        output = []
        for i in range(int(outputSize/4)):
            output.extend(split_int8(self.bram_outputs.read(offset)))
            offset = offset + 4
        return output



    # loads a bunch of images into PL, then convolves them and reads them
    # written with the idea that augmentations are not done so only convolutions occur and data is not standard stored in BRAM
    def HardwareConvolve(self,numParameters,batchSize,imageDim, images,QuantizationEnabled,outputSize):
        # load the images into the buffer
        self.load_buffer(numParameters,batchSize,imageDim, images,QuantizationEnabled)

        # give signal to start convolution
        self.start.write(0,1)
        self.start.write(0,0)
        while(self.done.read()==0):
            pass
        offset = 0
        output = []
        for i in range(int(outputSize/4)):
            output.extend(split_int8(self.bram_outputs.read(offset)))
            offset = offset + 4
        return output



    








    

        
    
