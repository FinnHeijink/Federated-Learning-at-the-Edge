`timescale 1ns/1ps
module write_tb
    #(  parameter DATA_WIDTH = 32,                                                  // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 32,                                                  // The size of the BRAM addresses in bits
        parameter OUTPUT_ADDR = 32'hA000_0000,                                      // The base address of the BRAM for the kernel data
        parameter PIXEL_SIZE = 8                                                    // Number of bits for 1 pixel                
    )();
    
        logic clk;
        logic reset;
        
        logic [ADDR_WIDTH-1:0] bram_addr;                                    // Only changing signals are the requested address, data to write and write enable
        logic [DATA_WIDTH-1:0] bram_data;                                    // Others are assumed to stay constant
        logic [3:0] write_enable;
        
        logic [PIXEL_SIZE-1:0] pixel;                                         // filter which has to be written in BRAM
        logic pixel_valid;                                                    // Indication that the pixel signal is a valid pixe
        logic conv_done;
         
        write_module test1 (clk, reset, bram_addr, bram_data, write_enable, pixel, pixel_valid, conv_done);
    
    always
         #5 clk = ~clk;  // period 10ns (100 MHz)
    initial
        clk = 0;
    
    initial begin
                reset = 1; pixel = 8'b0; pixel_valid = 0; conv_done = 0;      
     #20;       reset = 0; 
     #60;       pixel = 8'h11; pixel_valid = 1;
     #10;       pixel = 8'h22;
     #10;       pixel = 8'h33;
     #10;       pixel = 8'h44;
     #10;       pixel = 8'h55;
     #10;       pixel = 8'h66;
     #10;       pixel = 8'h77;
     #10;       pixel = 8'h88;
     #10;       pixel = 8'h99; pixel_valid = 0; // geen 99 in testbench zien als het goed is
     #10;       pixel = 8'hAA; 
     #10;       pixel = 8'hBB;
     #10;       pixel = 8'hCC; pixel_valid = 1;
     #10;       pixel = 8'hDD;
     #10;       pixel = 8'hEE;
     #10;       pixel = 8'hFF; 
     #10;       pixel = 8'h00;
     #10;       pixel = 8'h11; pixel_valid = 0;
     #10;       pixel = 8'h22;
     #10;       pixel = 8'h33;
     #10;       pixel = 8'h44; pixel_valid = 1;
     #10;       pixel = 8'h55;
     #10;       pixel = 8'h66;
     #10;       pixel = 8'h77;
     #10;       pixel = 8'h88;
     #10;       pixel = 8'h99;
     #10;       pixel = 8'hAA; 
     #10;       pixel = 8'hBB; conv_done = 1; pixel_valid =0;
     
     
    end
endmodule
    