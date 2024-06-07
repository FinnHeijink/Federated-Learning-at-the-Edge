module ResizedCrop
    #(  PIXEL_SIZE = 8,
        NUM_PIXELS = 784,
        BASE_ADDR1 = 32'hB000_0000,
        BASE_ADDR2 = 32'hB000_1000
    )
    (
        input logic clk,
        input logic reset,
                
        input logic start,                              // start with a new image
        input logic interrupt,                          // no new pixels should be send because their is a stall further in the line
        output logic image_done,                        // an image is done with reading, memory can be overwritten
            
        output logic [7:0] pixel_o,                     // pixel of the resized cropped image
        output logic pixel_valid,                       // the pixel is valid
        
        output logic [31:0] bram_address,               // address where the next pixel is located, one word one pixel
        input logic  [31:0] bram_data                   // data read from the bram 
                
        
    );
    
        typedef enum {idle, pre1_image1, pre2_image1, pre1_image2, pre2_image2, image1, image2, done, interruption} state_type;
        state_type state, next_state;
        
        logic [9:0] pixel_count, next_pixel_count;      // keeps track of which pixel is evaluated
        logic [31:0] next_bram_address;                 // used to calculate the next bram address
        logic [11:0] bram_offset16, 
        bram_offset20, bram_offset24;                   // the offset of the relevant pixel with respect to the base address
        logic [1:0] scale;                              // the dimensions of the cropped image before resizing
                                                        // 00 -> 28 bits, 01 -> 24 bits, 10 -> 20 bits, 11 -> 16 bits
        logic image,next_image;             // decides which image in memory is read 0 or 1
        
        always_ff @(posedge clk, posedge reset) begin
        if (reset) begin                                // module should start in the idle state
            state <= idle;
            pixel_count <= 'd0;
            bram_address <= 0;
            image <= 0;                                 // start with image 1 (0), continue to image 2 (1), continue alternating, memory has room for two images
        end else if (interrupt) begin                   // go directly to interruption state if an interrupt signal is received
            state <= interruption;
            pixel_count <= pixel_count;                 // current pixel_count and bram_address should we kept so no pixel is skipped
            bram_address <= bram_address;
            image <= next_image;
        end else begin
            state <= next_state;
            pixel_count <= next_pixel_count;
            bram_address <= next_bram_address;
            image <= next_image; 
        end
        end 
        
        
        always_comb begin
            case (state) 
                idle: begin                                                 // idle state: wait until a new image has been loaded by the previous augmentations
                    next_pixel_count = 0;
                    next_bram_address = 0;
                    next_image = image;
                    if ((start == 1) & (image == 0)) begin                  // a new image is complete in the memory, go to the state corresponding to the right image
                        next_state = pre1_image1;
                    end else if ((start == 1) & (image == 1)) begin
                        next_state = pre1_image2;
                    end else begin
                        next_state = idle;
                    end
                    pixel_valid = 0;
                    pixel_o = 0;
                    image_done = 0;                    
                end
            
                pre1_image1: begin                                          // make preparations to read image 1, it takes 2 cycles for the data at next_address to appear at the bram_data input
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)                                         // the pixels that are sent depend on the cropping scale, LUT gives address offset of the pixel which should be sent
                        next_bram_address = BASE_ADDR1;                     
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR1 + bram_offset24;
                    else if (scale == 2)
                        next_bram_address = BASE_ADDR1 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR1 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    next_image = image;
                    next_state = pre2_image1;
                    pixel_valid = 0;
                    pixel_o = 0;
                    image_done = 0;
                end
                
                pre2_image1: begin                                          // make preparations to read image 1
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)                                         // the pixels that are sent depend on the cropping scale, LUT gives address offset of the pixel which should be sent
                        next_bram_address = bram_address + 4;
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR1 + bram_offset24;
                    else if (scale == 2)
                        next_bram_address = BASE_ADDR1 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR1 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    next_image = image;
                    next_state = image1;
                    pixel_valid = 0;
                    pixel_o = 0;
                    image_done = 0;
                    
                end
            
                image1: begin                                               // read image 1
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)
                        next_bram_address = bram_address + 4;               // combine the addresses in a similar manner
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR1 + bram_offset24;
                    else if (scale == 2)
                        next_bram_address = BASE_ADDR1 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR1 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    
                    next_image = image;
                    image_done = 0;
                        
                    if (pixel_count == NUM_PIXELS+1) begin                    // when all pixels are read go to done
                        next_state = done;
                    end else begin
                        next_state = image1;
                    end
                    
                    pixel_o = bram_data[7:0];                               // only 1 pixel is stored on an address, in the 8 LSB
                    pixel_valid = 1;                                        // pixel is valid
                end
            
                pre1_image2: begin                                          // same as image 1 version
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)
                        next_bram_address = BASE_ADDR2;
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR2 + bram_offset24;
                    else if (scale == 2)
                        next_bram_address = BASE_ADDR2 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR2 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    next_image = image;
                    next_state = pre2_image2;
                    pixel_valid = 0;
                    pixel_o = 0;
                    image_done = 0;
                end
                    
                pre2_image2: begin                                          // same as image 1 version
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)
                        next_bram_address = bram_address + 4;
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR2 + bram_offset24;
                    else if (scale == 22)
                        next_bram_address = BASE_ADDR2 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR2 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    next_image = image;
                    next_state = image2;
                    pixel_valid = 0;
                    pixel_o = 0;
                    image_done = 0;
                end
            
                image2: begin                                               // same as image 1 version
                    next_pixel_count = pixel_count + 1;
                    if (scale == 0)
                        next_bram_address = bram_address + 4;
                    else if (scale == 1)
                        next_bram_address = BASE_ADDR2 + bram_offset24;
                    else if (scale == 2)
                        next_bram_address = BASE_ADDR2 + bram_offset20;
                    else if (scale == 3)
                        next_bram_address = BASE_ADDR2 + bram_offset16;
                    else
                        next_bram_address = 32'b0;
                    
                    next_image = image;
                    image_done = 0;
                    
                    if (pixel_count == NUM_PIXELS+1) begin
                        next_state = done;
                    end else begin
                        next_state = image2;
                    end
                    
                    pixel_o = bram_data[7:0];
                    pixel_valid = 1;
                end
                
                                
                done: begin                                                 // when an image is done the values can be reset
                    next_pixel_count = 0;
                    next_bram_address = 0;
                    next_image = ~image;                                 // the image in memory location is switched
                    next_state = idle;                                      // go back to idle to wait on the next image
                    image_done = 1;                                         // image is done
                    pixel_o = 0;                                            // no pixel is send
                    pixel_valid = 0;
                end
                
                interruption: begin                                         // the module should stop sending new values
                    next_pixel_count = pixel_count;                         // all values are stored as they were during the interruption
                    next_bram_address = bram_address;
                    next_image = image;
                    image_done = 0;
                    if ((interrupt ==0) & (image == 0)) begin               // continue with the right image when the interruption is done
                        next_state = image1;
                        if (scale == 0)                                     // when the interruption is solved the next address should already be calculated to ensure the next pixel is not send twice
                            next_bram_address = bram_address + 4;                     
                        else if (scale == 1)
                            next_bram_address = BASE_ADDR1 + bram_offset24;
                        else if (scale == 2)
                            next_bram_address = BASE_ADDR1 + bram_offset20;
                        else if (scale == 3)
                            next_bram_address = BASE_ADDR1 + bram_offset16;
                        else
                            next_bram_address = 32'b0;
                    end else if ((interrupt ==0) & (image == 1)) begin
                        next_state = image2;
                        if (scale == 0)                                      
                            next_bram_address = bram_address + 4;                     
                        else if (scale == 1)
                            next_bram_address = BASE_ADDR2 + bram_offset24;
                        else if (scale == 2)
                            next_bram_address = BASE_ADDR2 + bram_offset20;
                        else if (scale == 3)
                            next_bram_address = BASE_ADDR2 + bram_offset16;
                        else
                            next_bram_address = 32'b0;
                    end else begin
                    next_state = interruption;
                    end
                    pixel_o = 0;                                            // no valid pixels are sent
                    pixel_valid = 0;
                end
                
                default: begin
                    next_pixel_count = pixel_count;
                    next_bram_address = bram_address;
                    next_image = image;
                    image_done = 0;
                    next_state = idle;
                    pixel_o = 0;
                    pixel_valid = 0;
                end
            endcase
        end
        
        LUT_size16 L1(.pixel_counter(pixel_count),.address_offset(bram_offset16)); // gives the offset of the relevant pixel when the cropped image dimensions are 16 bits
        LUT_size20 L2(.pixel_counter(pixel_count),.address_offset(bram_offset20)); // gives the offset of the relevant pixel when the cropped image dimensions are 20 bits
        LUT_size24 L3(.pixel_counter(pixel_count),.address_offset(bram_offset24)); // gives the offset of the relevant pixel when the cropped image dimensions are 24 bits
        LSFR_crop L4(.clk(clk),.reset(reset),.enable(image_done),.scale(scale));                // gives the scale with which every image is cropped     
endmodule

