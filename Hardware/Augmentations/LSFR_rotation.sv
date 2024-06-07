module LSFR_rotation
    #()
    (
        input logic clk,
        input logic reset,
        input logic enable,
        output logic [1:0] degrees
    );
    
    logic [8:0] d,q;
    
    always_ff @(posedge clk, posedge reset)        // updates when an image is done in crop/rescale module
    if (reset)
    d <= 10'b100100001;            // start value of the lsfr
    else if (enable)
    d <= q;       // shift and add a value, taps are decided by xilinx document about LSFR
    else
    d <=d;
    
    always_comb begin
        q = {d[4]~^d[0],d[8:1]};
        degrees = d[1:0];
    end
endmodule