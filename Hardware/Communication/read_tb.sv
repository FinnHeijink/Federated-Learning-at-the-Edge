`timescale 1ns/1ps
module read_tb
    #(  parameter DATA_WIDTH = 32,                                                  // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 32,                                                  // The size of the BRAM addresses in bits
        parameter KERNEL_ADDR = 32'h0000_0000,                                      // The base address of the BRAM for the kernel data
        parameter IMAGE_ADDR = 32'h0000_0024,                                       // The base address of the BRAM for the image data    
        parameter PIXEL_SIZE = 8,                                                   // Number of bits for 1 pixel                
        parameter KERNEL_SIZE = 9,                                                  // Size of the kernel in terms of 32 bits!
        parameter NUM_IMAGES = 3                                                    // Number of images read at once
    )();
    
         logic clk;
         logic reset;

         logic [ADDR_WIDTH-1:0] bram_addr;                                    // Only changing signals are the requested address and the  data
         logic [DATA_WIDTH-1:0] bram_data;                                     // Others are assumed to stay constant

         logic read_kernel;                                                    // Signal from PS, read the kernel
         logic [KERNEL_SIZE-1:0][DATA_WIDTH-1:0] kernel;                      // Kernel register
        
         logic read_image;                                                     // Signal from PS, read the image
         logic [PIXEL_SIZE-1:0] pixel;                                        // Pass along the read pixel to the filter
         logic pixel_valid;                                                   // Indicate that a new pixel has been read
        
         logic interrupt;
         
         read_module test1 (clk, reset, bram_addr, bram_data, read_kernel, kernel, read_image, pixel, pixel_valid, interrupt);
         bram_sim    test2 (clk, reset, bram_addr, bram_data, 1'b1);
    
    always
         #5 clk = ~clk;  // period 10ns (100 MHz)
    initial
        clk = 0;
    
    initial begin
                reset = 1; read_kernel = 0; read_image = 0; interrupt =0;      
     #20;       reset = 0; read_kernel = 1;
     #60;       read_kernel = 0;
     #200;      read_image = 1;
     #60;       read_image = 0;
     #305;      interrupt = 1;
     #60        interrupt = 0;
     
    end
endmodule
    