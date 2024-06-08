module check_kernel
    #(  parameter DATA_WIDTH = 32,
        parameter KERNEL_SIZE = 9
    )(
    input logic [KERNEL_SIZE-1:0][DATA_WIDTH-1:0] in,
    input logic [DATA_WIDTH-1:0] selecter,
    output logic [DATA_WIDTH-1:0] result
);
    always_comb begin
       result = in[selecter];
    end
endmodule 