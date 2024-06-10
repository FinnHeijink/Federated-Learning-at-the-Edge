module LFSR_rotation                                // pseudo-random number generator to randomise rotation
    #()
    (
        input logic clk,
        input logic reset,
        input logic enable,                         // LFSR should generate a new output
        output logic [1:0] degrees                  // pseudo-random output of the LFSR
    );
    
    logic [8:0] d,q;                                // 9-bit LFSR
    
    always_ff @(posedge clk, posedge reset)        
    if (reset)
    d <= 10'b100100001;                             // start value of the LFSR
    else if (enable)
    d <= q;                                         // shift and add a new bit
    else
    d <= d;                                         // do not change under normal circumstances
    
    always_comb begin
        q = {d[4]~^d[0],d[8:1]};                    // the new bit is an xnor output of the 9th and 5th flipflop outputs, according to the xlnx LFSR document
        degrees = d[1:0];                           // the pseudo random output consists of the outputs of the last 3 flipflops
    end
endmodule