module InternalBramKRIA
    #(
    )
    ();

    logic [31:0] bram_addr_in, bram_addr_out, bram_data_in, bram_data_out, not_used;
    logic [10:0] bram_addr_int_in, bram_addr_int_out;
    logic [7:0] bram_data_int_in, bram_data_int_out, not_used_int, rot_pixel;
    logic [3:0] w_enable_out;
    logic clk, reset, w_enable_int, start_crop, start_rot, rot_pixel_valid;

    ResizedCropBS_wrapper
       (.bram_input_addr(bram_addr_in),
        .bram_input_clk(clk),
        .bram_input_din(32'b0),
        .bram_input_dout(bram_data_in),
        .bram_input_en(1'b1),
        .bram_input_rst(1'b0),
        .bram_input_we(4'b0),
        .bram_output_addr(bram_addr_out),
        .bram_output_clk(clk),
        .bram_output_din(bram_data_out),
        .bram_output_dout(not_used),
        .bram_output_en(1'b1),
        .bram_output_rst(1'b0),
        .bram_output_we(w_enable_out),
        .clock(clk),
        .internal_bram_in_addr(bram_addr_int_in),
        .internal_bram_in_clk(clk),
        .internal_bram_in_din(bram_data_int_in),
        .internal_bram_in_dout(not_used_int),
        .internal_bram_in_en(1'b1),
        .internal_bram_in_we(w_enable_int),
        .internal_bram_out_addr(bram_addr_int_out),
        .internal_bram_out_clk(clk),
        .internal_bram_out_din(8'b0),
        .internal_bram_out_dout(bram_data_int_out),
        .internal_bram_out_en(1'b1),
        .internal_bram_out_we(1'b0),
        .restart_tri_o(reset),
        .start_tri_o(start_rot),
        .bram_addr_tri_o(bram_addr_int_out),
        .bram_data_tri_i(bram_data_int_out)
        );
        
    write_augmented
        (.clk(clk),
         .reset(reset),
         .bram_addr(bram_addr_int_in),                                    
         .bram_data(bram_data_int_in),                                    
         .w_enable(w_enable_int),
         .pixel(rot_pixel),                                         
         .pixel_valid(rot_pixel_valid));
        
    Rotation
        (.clk(clk),
         .reset(reset),      
         .start(start_rot),                              
         .image_done(start_crop),                       
         .pixel_o(rot_pixel),                     
         .pixel_valid(rot_pixel_valid),                       
         .bram_address(bram_addr_in),               
         .bram_data(bram_data_in));
    
//    ResizedCrop
//        (.clk(clk),
//         .reset(reset),      
//         .start(start_crop),                              
//         .image_done(crop_done),                        
//         .pixel_o(crop_pixel),          
//         .pixel_valid(crop_pixel_valid),                      
//         .bram_address(bram_addr_int_out),     
//         .bram_data(bram_data_int_out));
    
//    write_module
//        (.clk(clk),
//         .reset(reset),
//         .bram_addr(bram_addr_out),                                    
//         .bram_data(bram_data_out),                                    
//         .write_enable(w_enable_out),
//         .pixel(crop_pixel),                                        
//         .pixel_valid(crop_pixel_valid));
         
//    BRAM_sim_internal 
//        (.address(bram_addr_int_out),
//         .data_send(bram_data_int_out),
//         .clk(clk),
//         .reset(reset));

    
endmodule