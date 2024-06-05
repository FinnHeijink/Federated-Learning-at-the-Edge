module LSFR_crop
    #()
    (
        input logic clk,
        input logic reset,
        output logic [1:0] scale
    );
    
    logic [9:0] d,q;
    
    always_ff @(posedge clk, posedge reset)        // updates when an image is done in crop/rescale module
    if (reset)
    d <= 10'b0000101001;            // start value of the lsfr
    else
    d <= q;       // shift and add a value, taps are decided by xilinx document about LSFR
    
    always_comb begin
        q = {d[3]~^d[0],d[9:1]};
        scale = d[1:0];
    end
endmodule