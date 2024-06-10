`timescale 1ns/1ps
module ResizedCrop_tb
    #()();
    
        logic clk;
        logic reset;
                
        logic start;                              // start with a new image
        logic image_done;                        // an image is done with reading, memory can be overwritten
            
        logic [7:0] pixel_o;                     // pixel of the resized cropped image
        logic pixel_valid;                       // the pixel is valid
        
        logic [10:0] bram_address;               // address where the next pixel is located, one word one pixel
        logic  [7:0] bram_data;                   // data read from the bram
    
    ResizedCrop L1 (clk,reset,start, image_done,pixel_o,pixel_valid,bram_address,bram_data);
    BRAM_sim_internal L2(bram_address,bram_data,clk,reset);
    
    always
         #5 clk = ~clk;  // period 10ns (100 MHz)
    initial
        clk = 0;
    
    initial begin
                reset = 1; start = 0;   
     #20;       reset = 0; 
     #20;       start = 1;
     #20;       start = 0;
     #10000;     start = 1;
     #20;       start = 0;
     #9000;      start = 1;
     #20; start = 0;
     
    end
endmodule