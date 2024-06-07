module check_output(
    output logic pixel_valid,
    output logic [7:0] pixel,
    input logic clk,
    input logic reset
);

    logic [3:0] count;
    
    always_ff @(posedge(clk))
        if (reset) begin
            count <= 0;
            pixel_valid <= 1;
        end else begin
            count <= count + 1;
            pixel_valid <= !pixel_valid;
        end    
        
    always_comb begin        
        case (count)
        4'd0:  pixel = 8'h00;  // Assigning a unique 8-bit value for each case
        4'd1:  pixel = 8'h11;
        4'd2:  pixel = 8'h22;
        4'd3:  pixel = 8'h33;
        4'd4:  pixel = 8'h44;
        4'd5:  pixel = 8'h55;
        4'd6:  pixel = 8'h66;
        4'd7:  pixel = 8'h77;
        4'd8:  pixel = 8'h88;
        4'd9:  pixel = 8'h99;
        4'd10: pixel = 8'hAA;
        4'd11: pixel = 8'hBB;
        4'd12: pixel = 8'hCC;
        4'd13: pixel = 8'hDD;
        4'd14: pixel = 8'hEE;
        4'd15: pixel = 8'hFF;
        default: pixel = 8'h00; // Default case to handle any unexpected values
    endcase
end
endmodule