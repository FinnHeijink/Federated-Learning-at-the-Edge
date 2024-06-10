module GaussianBlurKRIA();
   
        // system signals
        logic clk, reset, pixel_in_valid,int_w_enable, interrupt, blur_done, start_image;
        logic [7:0] pixel_in, int_bram_data_in, int_bram_data_out, not_used2;                           
        logic [10:0] int_bram_addr_in, int_bram_addr_out;  
        logic [31:0] bram_input_addr, bram_input_data, not_used1;
        logic [107:0] kernel;

    
    read_module(
        .clk(clk),
        .reset(reset),
        .bram_addr(bram_input_addr),                                   
        .bram_data(bram_input_data),                                 
        .read_kernel(1'b0),                                                   
        .kernel(kernel),                 
        .read_image(start_image),                                                 
        .pixel(pixel_in),                                      
        .pixel_valid(pixel_in_valid),                                                  
        .interrupt(interrupt));
    
    GaussianBlur(
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),                          
        .pixel_in_valid(pixel_in_valid),                    
        .bram_addr(int_bram_addr_in),                         
        .bram_data(int_bram_data_in),                      
        .w_enable(int_w_enable),                       
        .interrupt(interrupt),                         
        .image_done(blur_done)                         
    );
    
    ResizedCropBS_wrapper
   (.bram_addr_tri_o(int_bram_addr_out),
    .bram_data_tri_i(int_bram_data_out),
    .bram_input_addr(bram_input_addr),
    .bram_input_clk(clk),
    .bram_input_din(32'b0),
    .bram_input_dout(bram_input_data),
    .bram_input_en(1'b1),
    .bram_input_rst(1'b0),
    .bram_input_we(4'b0),
    .bram_output_addr(32'b0),
    .bram_output_clk(clk),
    .bram_output_din(32'b0),
    .bram_output_dout(not_used1),
    .bram_output_en(4'b0),
    .bram_output_rst(1'b0),
    .bram_output_we(4'b0),
    .clock(clk),
    .internal_bram_in_addr(int_bram_addr_in),
    .internal_bram_in_clk(clk),
    .internal_bram_in_din(int_bram_data_in),
    .internal_bram_in_dout(not_used2),
    .internal_bram_in_en(1'b1),
    .internal_bram_in_we(int_w_enable),
    .internal_bram_out_addr(int_bram_addr_out),
    .internal_bram_out_clk(clk),
    .internal_bram_out_din(8'b0),
    .internal_bram_out_dout(int_bram_data_out),
    .internal_bram_out_en(1'b1),
    .internal_bram_out_we(1'b0),
    .restart_tri_o(reset),
    .start_tri_o(start_image));
endmodule