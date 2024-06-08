`timescale 1ns/1ps
module communication_tb
    #(  parameter DATA_WIDTH = 32,                                                  // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 32,                                                  // The size of the BRAM addresses in bits
        parameter KERNEL_ADDR = 32'hA000_0000,                                      // The base address of the BRAM for the kernel data
        parameter IMAGE_ADDR = 32'hA000_0024,                                       // The base address of the BRAM for the image data    
        parameter PIXEL_SIZE = 8,                                                   // Number of bits for 1 pixel                
        parameter KERNEL_SIZE = 9,                                                  // Size of the kernel in terms of 32 bits!
        parameter IMAGE_SIZE = 16,  //196                                           // Size of the image in terms of 32 bits!
        parameter NUM_IMAGES = 3                                                    // Number of images read at once
    )();
    
    logic clk, reset, read_kernel, read_image, pixel_valid, interrupt, conv_done; 
    logic [3:0] write_enable;
    logic [7:0] pixel;
    logic [31:0] addr_in, data_in, addr_out, data_out;
    logic [8:0][31:0] kernel;
    
    
    read_module test1 (clk, reset, addr_in, data_in, read_kernel, kernel, read_image, pixel, pixel_valid, interrupt);
    bram_sim    test2 (clk, reset, addr_in, data_in, 1'b1);
    write_module test3 (clk, reset, addr_out, data_out, write_enable, pixel, pixel_valid, conv_done);

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

     
    end
endmodule
    