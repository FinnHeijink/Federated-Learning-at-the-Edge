module Augmentation                                             // perform all augmentations
    #(
        ADDR_WIDTH_INT = 11,                                    // Addr width of augmentation BRAMs
        PIXEL_WIDTH = 8,                                        // size of one pixel
        NUM_PIXELS = 784,                                       // number of pixels in an image
        BRAM_INT_BASE1 = 11'h000,                               // base address of the first image in the augmentation BRAMs
        BRAM_INT_BASE2 = 11'h310                                // base address of the second image in the augmentation BRAMs
    )
    (
        // system signals
        input logic                     clk,
        input logic                     reset,
        
        // communication with read module
        input logic [PIXEL_WIDTH-1:0]   pixel_in,               // pixel from read module out of buffer
        input logic                     pixel_in_valid,         // pixel from read module is valid
        
        // communication with internal memory
        output logic [ADDR_WIDTH_INT-1:0] blur_bram_addr_in,    // blurred image written to this address
        output logic [PIXEL_WIDTH-1:0]  blur_bram_data_in,      // blurred image pixel being written
        output logic                    blur_bram_w_enable,     // buffer for blurred images can be written to
        output logic [ADDR_WIDTH_INT-1:0] blur_bram_addr_out,   // blurred image read from this address
        input logic [PIXEL_WIDTH-1:0]   blur_bram_data_out,     // blurred image pixel being read
        output logic [ADDR_WIDTH_INT-1:0] rot_bram_addr_in,     // rotated image written to this address
        output logic [PIXEL_WIDTH-1:0]  rot_bram_data_in,       // rotated image pixel being written
        output logic                    rot_bram_w_enable,      // buffer for rotated images can be written to
        output logic [ADDR_WIDTH_INT-1:0] rot_bram_addr_out,    // rotated image read from this address
        input logic [PIXEL_WIDTH-1:0]   rot_bram_data_out,      // rotated image pixel being read
        
        // communication with actual CNN
        output logic [PIXEL_WIDTH-1:0]  pixel_out,              // pixel  of the augmented image
        output logic                    pixel_out_valid,        // pixel is valid
        
        // control  
        output logic                    interrupt               // interrupt signal to the read signal
    );
    
    // internal signals
    logic [PIXEL_WIDTH-1:0] rot_pixel;                          // output pixels of rotation module
    logic                   rot_pixel_valid;                    // valid indications of this pixels
    logic                   rot_done, blur_done, crop_done;     // signals to indicate that a augmentation for an image is done
    logic                   write_done;                         // writing from rot to bram is done
    
    // port maps

    GaussianBlur                                                // Apply a Gaussian Blur (or does not) and writes into the internal bram memory
       (.clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),                          
        .pixel_in_valid(pixel_in_valid),                 
        .bram_addr(blur_bram_addr_in),                       
        .bram_data(blur_bram_data_in),                        
        .w_enable(blur_bram_w_enable),                        
        .interrupt(interrupt),                        
        .image_done(blur_done));
         
    Rotation                                                    // rotate image
        (.clk(clk),
         .reset(reset),      
         .start(blur_done),                              
         .image_done(rot_done),                       
         .pixel_o(rot_pixel),                     
         .pixel_valid(rot_pixel_valid),                       
         .bram_address(blur_bram_addr_out),               
         .bram_data(blur_bram_data_out));
      
    write_augmented                                             // write rotated image in internal memory
       (.clk(clk),
        .reset(reset),
        .bram_addr(rot_bram_addr_in),                    
        .bram_data(rot_bram_data_in),                   
        .w_enable(rot_bram_w_enable),
        .pixel(rot_pixel),                         
        .pixel_valid(rot_pixel_valid),                                
        .image_done(write_done)); 
         
    ResizedCrop                                                 // crop and resize image
        (.clk(clk),
         .reset(reset),      
         .start(rot_done),                              
         .image_done(crop_done),                        
         .pixel_o(pixel_out),          
         .pixel_valid(pixel_out_valid),                      
         .bram_address(rot_bram_addr_out),     
         .bram_data(rot_bram_data_out));
    
endmodule