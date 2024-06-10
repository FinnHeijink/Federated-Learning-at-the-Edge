module LFSR_crop_tb
    ();
    
    logic clk, reset;
    logic [1:0] scale;
    
    LFSR_crop L1 (clk,reset,scale);
    
    always
        #5 clk = ~clk;  // period 10ns (100 MHz)
    initial
        clk = 0;
    
    initial begin
                reset = 1;       
     #20;       reset = 0; 
     end
endmodule