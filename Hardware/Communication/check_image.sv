module check_image
    #(  parameter DATA_WIDTH = 32,
        parameter IMAGE_SIZE = 4 //16/4
    )(
    input logic [IMAGE_SIZE-1:0][DATA_WIDTH-1:0] in,
    input logic [DATA_WIDTH-1:0] selecter,
    output logic [DATA_WIDTH-1:0] result
);
    always_comb begin
       result = in[selecter];
    end
endmodule 