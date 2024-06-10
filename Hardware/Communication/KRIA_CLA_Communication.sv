module top_KRIA_CLA_Communication();

    logic clock, read_kernel, read_image, reset, pixel_valid;
    logic [3:0] write_enable;
    logic [7:0] pixel;
    logic [8:0][31:0] kernel;
    logic [31:0] addr_in, data_in, addr_out, data_out, not_used, result_k, result_i, selecter;
    logic [17:0][7:0] image;

    KRIA_CLA_Communication_wrapper L1
       (.BRAM_inputs_addr(addr_in),
        .BRAM_inputs_clk(clock),
        .BRAM_inputs_din(32'b0),
        .BRAM_inputs_dout(data_in),
        .BRAM_inputs_en(1'b1),
        .BRAM_inputs_rst(1'b0),
        .BRAM_inputs_we(3'b0),
        .BRAM_outputs_addr(addr_out),
        .BRAM_outputs_clk(clock),
        .BRAM_outputs_din(data_out),
        .BRAM_outputs_dout(not_used),
        .BRAM_outputs_en(1'b1),
        .BRAM_outputs_rst(1'b0),
        .BRAM_outputs_we(write_enable),
        .clock(clock),
        .read_image_tri_o(read_image),
        .read_kernel_tri_o(read_kernel),
        .restart_tri_o(reset),
        .result_i_tri_i(result_i),
        .result_k_tri_i(result_k),
        .selecter_tri_o(selecter));
        
    read_module L2
       (.clk(clock),
        .reset(reset),
        .bram_addr(addr_in),                            
        .bram_data(data_in),                                    
        .read_kernel(read_kernel),                                               
        .kernel(kernel),                 
        .read_image(read_image),                                                     
        .pixel(pixel),                                  
        .pixel_valid(pixel_valid),
        .interrupt(1'b0));
    
    write_module 
       #(.OUTPUT_ADDR(32'h00000000)) L3
       (.clk(clock),
        .reset(reset),
        .bram_addr(addr_out),                                    
        .bram_data(data_out),                                   
        .write_enable(write_enable),
        .pixel(pixel),                                         
        .pixel_valid(pixel_valid));
    
    check_kernel L4
       (.in(kernel),
        .selecter(selecter),
        .result(result_k));

    
    //check_output
    //(.clk(clock), .reset(reset), .pixel(pixel), .pixel_valid(pixel_valid));

endmodule