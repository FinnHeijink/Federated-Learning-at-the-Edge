module write_augmented_tb();
  
        logic clk;
        logic reset;
        logic [11-1:0] bram_addr;                                    // Only changing signals are the requested address, data to write and write enable
        logic [8-1:0] bram_data;                                    // Others are assumed to stay constant
        logic [1-1:0] w_enable;
        logic [8-1:0] pixel;                                        // Output filter which has to be written in BRAM
        logic pixel_valid; 
  
    write_blur test1 (clk,reset,bram_addr,bram_data,w_enable,pixel,pixel_valid);
    
        always
         #5 clk = ~clk;  // period 10ns (100 MHz)
    initial
        clk = 0;
    
    initial begin
        reset = 1; pixel = 8'h00; pixel_valid = 0;
    #20;       reset = 0; 

    #75;       pixel_valid = 1; pixel = 8'h01;
    #10;        pixel = 9'h02;
     #10;       pixel = 8'h03;
     #10;       pixel = 8'h04;
     #10;       pixel = 8'h05;
     #10;       pixel = 8'h06;
     #10;       pixel = 8'h07;
     #10;       pixel = 8'h08;
     #10;       pixel = 8'h09;
     #10;       pixel = 8'h0A;
     #10;       pixel = 8'h0B;
     #10;       pixel = 8'h0C;
     #10;       pixel = 8'h0D;
     #10;       pixel = 8'h0E;
     #10;       pixel = 8'h10;
     #10;       pixel = 8'h11;
     #10;       pixel = 8'h12; pixel_valid = 0;
     #30;       pixel_valid = 1; pixel = 8'h13;
     #10;       pixel = 8'h14;
     #10;       pixel = 8'h15;
     #10;       pixel = 8'h16;
     #10;       pixel = 8'h17;
     #10;       pixel = 8'h18;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     #10;       pixel = 8'h19;
     
     
    end
endmodule