module write_augmented                                              // modified version of original write module, one pixel per address and only 2 image locations    
    #(  parameter DATA_WIDTH = 8,                                   // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 11,                                  // The size of the BRAM addresses in bits
        parameter OUTPUT_ADDR = 11'h0,                              // The base address of the BRAM for the first image location
        parameter PIXEL_SIZE = 8,                                   // Number of bits for 1 pixel   
        parameter PIXEL_PER_WORD = 1                                // Number of pixels written at the same time to the BRAM
    )(     
        // System signals
        input logic clk,
        input logic reset,
        
        // BRAM signals
        output logic [ADDR_WIDTH-1:0] bram_addr,                    // Only changing signals are the requested address, data to write and write enable
        output logic [DATA_WIDTH-1:0] bram_data,                    // Others are assumed to stay constant
        output logic [PIXEL_PER_WORD-1:0] w_enable,
        
        // Hardware Output
        input logic [PIXEL_SIZE-1:0] pixel,                         // Output pixel of the relevant augmentation module to be written into the BRAM
        input logic pixel_valid,                                    // Indication that the pixel signal is valid 
        
        output logic image_done);                                   // Indication that an image has been entirely written
        
        // Internal signals
        logic [ADDR_WIDTH-1:0] next_bram_addr;                      // output values for the next clock cycle, writing has 1 clock cycle delay
        logic [PIXEL_SIZE-1:0] next_bram_data;                      //
        logic [PIXEL_PER_WORD-1:0] next_w_enable;                   //
        logic image, next_image;                                    // keeps track to which of the two image locations is written
        logic [9:0] pixel_count,next_pixel_count;                   // keeps track of the number of pixels written to the BRAM                  
        
        typedef enum {idle, writing} state_types;
        state_types state, next_state;
        
        // State registers
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                state <= idle;
            end
            else begin
                state <= next_state;
            end
        
        //counters
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                image<=0;
                pixel_count<= 0;
            end
            else begin
                image <= next_image;
                pixel_count<= next_pixel_count;
            end
            
        // BRAM signals
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                bram_addr <= OUTPUT_ADDR-1;
                bram_data <= 0;
                w_enable <= 1'b0;
            end else begin
                bram_addr <= next_bram_addr;
                bram_data <= next_bram_data;
                w_enable <= next_w_enable;
                
            end
        
        always_comb begin
            // values generally stay equal or are zero
            next_w_enable = 1'b0;
            next_bram_addr = bram_addr;
            next_bram_data = bram_data;
            next_state = state;
            next_image = image;
            next_pixel_count = pixel_count;
            image_done = 0;
        
            unique case (state)
                idle: begin                                         // Nothing is happening
                    next_bram_addr = OUTPUT_ADDR-1;                 // -1 so the first valid pixel is written to OUTPUT_ADDR
                    next_bram_data = 0;
                    next_state = writing;                           // immediate go to writing state
                end

                writing: begin                                      // Write pixels into BRAM, continue indefinetely
                    if (pixel_valid) begin                          // Pixel valid? Ready signals for next clock cycle to write
                        next_bram_data = pixel;
                        next_bram_addr=bram_addr + 1;
                        next_w_enable = 1'b1;                                          
                        next_pixel_count = pixel_count + 1;
                        if (pixel_count == 783) begin               
                            next_pixel_count =0;                    // reset pixel_counter when all pixels of an image are read
                            next_image = ~image;                    // update the image tracker
                            image_done =1;                          // an image has been entirely written
                            if (image == 1) begin
                            next_bram_addr = OUTPUT_ADDR-1;         // go back to the start location of the first image location         
                            end
                        end     
                    end
                end
                
                default: begin
                    next_w_enable = 1'b0;
                    next_bram_addr = bram_addr;
                    next_bram_data = bram_data;
                    next_state = state;
                    next_image = image;
                    next_pixel_count = pixel_count;
                    image_done = 0;
                end
            endcase
        end                    
endmodule