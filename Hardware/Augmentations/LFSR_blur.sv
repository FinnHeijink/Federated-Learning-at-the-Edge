module LFSR_blur                                    // pseudo-random number generator to randomise Gaussian blur
    #()
    (
        input logic clk,        
        input logic reset,
        input logic enable,                         // LFSR should generate a new output
        output logic [1:0] select                   // pseudo-random output of the LFSR
    );
    
    logic [10:0] d,q;                               // 11-bit LFSR
    
    always_ff @(posedge clk, posedge reset)         
    if (reset)
        d <= 11'b01010100111;                       // start value of the LFSR
    else if (enable)
        d <= q;                                     // shift and add a new bit
    else
        d <= d;                                     // do not change under normal circumstances
    
    always_comb begin
        q = {d[2]~^d[0],d[10:1]};                   // the new bit is an xnor output of the 11th and 9th flipflop outputs, according to the xlnx LFSR document
        select = d[1:0];                            // the pseudo random output consists of the outputs of the last 3 flipflops
    end
endmodule