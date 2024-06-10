`timescale 1ns/1ps
module RotationWrite_tb
    #()();
    
        logic clk;
        logic reset;
                
        logic start;                              // start with a new image
        logic image_done;                        // an image is done with reading, memory can be overwritten
            
        logic [7:0] pixel_o;                     // pixel of the resized cropped image
        logic pixel_valid;                       // the pixel is valid
        
        logic [31:0] bram_address;               // address where the next pixel is located, one word one pixel
        logic [11:0] bram_addr_out;
        logic [7:0] bram_data_out;
        logic  [31:0] bram_data;                   // data read from the bram
        logic [0:0] w_enable;
    
    Rotation L1 (clk,reset,start,image_done,pixel_o,pixel_valid,bram_address,bram_data);
    BRAM_sim_biiiig L2(bram_address,bram_data,clk,reset);
    write_augmented L3(clk,reset,bram_addr_out,bram_data_out,w_enable,pixel_o,pixel_valid);
    
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
     #20;       start = 0;
     
    end
endmodule