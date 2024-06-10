module LFSR_crop                                // pseudo-random number generator to randomise resized crop
    #()
    (
        input logic clk,                        
        input logic reset,
        input logic enable,                     // LFSR should generate a new output
        output logic [1:0] scale                // pseudo-random output of the LFSR
    );
    
    logic [9:0] d,q;                            // 10-bit LFSR
    
    always_ff @(posedge clk, posedge reset)      
    if (reset)
    d <= 10'b0000101001;                        // start value of the LFSR
    else if (enable)
    d <= q;                                     // shift and add a new bit
    else
    d <= d;                                     // do not change under normal circumstances
    
    always_comb begin
        q = {d[3]~^d[0],d[9:1]};                // the new bit is an xnor output of the 10th and 7th flipflop outputs, according to the xlnx LFSR document
        scale = d[1:0];                         // the pseudo random output consists of the outputs of the last 3 flipflops
    end
endmodule