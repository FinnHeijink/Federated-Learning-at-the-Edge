module ResizedCropKRIA
    #(
    )
    ();
    
    // connection signals
    
    logic clk, reset;                                   // system signals
    logic crop_start, crop_done;             // control signals crop module
    //logic interrupt
    logic [7:0] pixel;                                  // pixel of cropped image to be written into memory
    logic pixel_valid;                                  // this pixel is valid
    logic [31:0] bram_address_in, bram_address_out;     // the addresses to access the input and output bram memory
    logic [31:0] bram_data_in, bram_data_out, not_used; // the data read from/written in the brams
    logic [3:0] w_enable;                               // enable writing option output bram
    
    
    // port maps
    
    ResizedCropBS_wrapper
       (.bram_input_addr(bram_address_in),
        .bram_input_clk(clk),
        .bram_input_din(32'b0),
        .bram_input_dout(bram_data_in),
        .bram_input_en(1'b1),
        .bram_input_rst(1'b0),
        .bram_input_we(4'b0),
        .bram_output_addr(bram_address_out),
        .bram_output_clk(clk),
        .bram_output_din(bram_data_out),
        .bram_output_dout(not_used),
        .bram_output_en(1'b1),
        .bram_output_rst(1'b0),
        .bram_output_we(w_enable),
        .clock(clk),
        .restart_tri_o(reset),
        .start_tri_o(crop_start));
        

    
    ResizedCrop
   (.clk(clk),
    .reset(reset),
    .start(crop_start),                             
    //.interrupt(1'b0),                          
    .image_done(crop_done),                      
    .pixel_o(pixel),                   
    .pixel_valid(pixel_valid),                    
    .bram_address(bram_address_in),               
    .bram_data(bram_data_in));
    
    write_module 
  #(.OUTPUT_ADDR(32'h00000000))     
   (.clk(clk),
    .reset(reset),
    .bram_addr(bram_address_out),                                    
    .bram_data(bram_data_out),                                    
    .write_enable(w_enable),
    .pixel(pixel),                                         
    .pixel_valid(pixel_valid));
    
endmodule

