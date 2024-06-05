module ResizedCropKRIA
    #()
    ();
    
    // connection signals
    
    logic clk, reset;                                   // system signals
    logic crop_start, interrupt, crop_done;             // control signals crop module
    logic [7:0] pixel;                                  // pixel of cropped image to be written into memory
    logic pixel_valid;                                  // this pixel is valid
    logic [31:0] bram_address_in, bram_address_out;     // the addresses to access the input and output bram memory
    logic [31:0] bram_data_in, bram_data_out;           // the data read from/written in the brams
    logic [3:0] w_enable;                               // enable writing option output bram
    logic conv_done;                                    // not used yet, controlsignal of write module to restart or something
    
    
    // port maps
    
    ResizedCropBS_wrapper
   (.bram_address_in(bram_address_in),
    .bram_address_out(bram_address_out),
    .bram_clk_in(clk),
    .bram_clk_out(clk),
    .bram_data_in(bram_data_in),
    .bram_data_out(bram_data_out),
    .bram_enable_in(1'b1),                              // standard enabled
    .bram_enable_out(1'b1),                             // standard enabled
    .bram_wenable_out(w_enable),
    .clock(clk),
    .restart_tri_o(reset),
    .start_tri_o(crop_start));
    
    ResizedCrop
   (.clk(clk),
    .reset(reset),
    .start(crop_start),                             
    .interrupt(interrupt),                          
    .image_done(crop_done),                      
    .pixel_o(pixel),                   
    .pixel_valid(pixel_valid),                    
    .bram_address(bram_address_in),               
    .bram_data(bram_data_in));
    
    write_module    
   (.clk(clk),
    .reset(reset),
    .bram_addr(bram_address_out),                                    
    .bram_data(bram_data_out),                                    
    .write_enable(w_enable),
    .pixel(pixel),                                         
    .pixel_valid(pixel_valid),                                             
    .conv_done(conv_done));
    
endmodule

